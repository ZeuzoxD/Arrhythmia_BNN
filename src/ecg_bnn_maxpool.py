import torch
import torch.nn.functional as F
import numpy as np
import argparse
import os
from pathlib import Path
import random

# Dataset labels for 5-class ECG classification
LABELS_5 = ['N', 'S', 'V', 'F', 'Q']

# ---------------------------
#  Binary Ops
# ---------------------------

def sign_binarize(x):
    """Binarize values to {+1, -1}"""
    return torch.sign(x).clamp(min=-1, max=1)

def real_conv1d(x, weight, stride=1, padding=0):
    """Standard convolution for first layer (real input × binary weights)"""
    if padding > 0:
        x = F.pad(x, (padding, padding), value=1.0)  # Pad with +1
    return F.conv1d(x, weight, stride=stride)

def xnor_popcount_conv1d(x_binary, weight_binary, stride=1, padding=0):
    """XNOR + POPCOUNT convolution for binary inputs × binary weights"""
    B, C_in, L = x_binary.shape
    C_out, _, K = weight_binary.shape

    if padding > 0:
        x_binary = F.pad(x_binary, (padding, padding), value=1.0)

    # Unfold input for convolution
    x_unf = x_binary.unfold(dimension=2, size=K, step=stride)  # [B,C_in,L_out,K]
    x_unf = x_unf.permute(0, 2, 1, 3)  # [B,L_out,C_in,K]

    # Expand dims for broadcasting
    x_exp = x_unf.unsqueeze(2)  # [B,L_out,1,C_in,K]
    w_exp = weight_binary.unsqueeze(0).unsqueeze(0)  # [1,1,C_out,C_in,K]

    # XNOR: +1 if same sign, -1 otherwise
    xnor = (x_exp == w_exp).int()  # 1 where match, 0 otherwise
    popcount = xnor.sum(dim=(3, 4))  # [B,L_out,C_out]

    total_bits = C_in * K
    out_int = (2 * popcount - total_bits).to(torch.int32)

    return out_int.permute(0, 2, 1)  # [B,C_out,L_out]

# ---------------------------
#  Integer max-pool helper
# ---------------------------

def int_maxpool1d(x_int, kernel_size=7, stride=2):
    """
    Integer maxpool via unfold + amax.
    x_int: [B, C, L] int32/int64
    returns: [B, C, L_out] same dtype (int)
    """
    B, C, L = x_int.shape
    if L < kernel_size:
        pad = torch.full((B, C, kernel_size - L), -2**30, dtype=x_int.dtype, device=x_int.device)
        x_int = torch.cat([x_int, pad], dim=2)
        L = x_int.shape[2]
    x_unf = x_int.unfold(2, kernel_size, stride)  # [B, C, L_out, kernel_size]
    return x_unf.amax(dim=3)

# ---------------------------
#  Threshold + Final Layer (Int)
# ---------------------------

def compute_fixed_point_thresholds(bn_weight, bn_bias, bn_mean, bn_var, prelu_weight,
                                   scale_factor=2**10, eps=1e-5):
    """
    Compute fixed-point thresholds for hardware implementation.
    Returns integer thresholds scaled by scale_factor.
    """
    bn_weight = bn_weight.float().view(-1)
    bn_bias = bn_bias.float().view(-1)
    bn_mean = bn_mean.float().view(-1)
    bn_var = bn_var.float().view(-1)

    # Handle PReLU weight
    if isinstance(prelu_weight, (float, int)):
        a = torch.full_like(bn_mean, float(prelu_weight))
    else:
        a = prelu_weight.float().view(-1)
        if a.numel() == 1:
            a = a.expand_as(bn_mean)
        elif a.numel() != bn_mean.numel():
            a = a.expand_as(bn_mean)

    s = torch.sqrt(bn_var + eps)
    gamma_safe = bn_weight.clone()
    small_mask = gamma_safe.abs() < 1e-6
    gamma_safe[small_mask] = torch.sign(gamma_safe[small_mask]) * 1e-6

    # Thresholds (float first)
    delta_plus = bn_mean - bn_bias * s / gamma_safe

    bn0 = gamma_safe * (0.0 - bn_mean) / s + bn_bias
    a_nonzero = (a.abs() >= 1e-12)
    large_pos = torch.full_like(delta_plus, 1e9)
    large_neg = torch.full_like(delta_plus, -1e9)

    delta_minus = torch.where(
        a_nonzero,
        delta_plus / a,
        torch.where(bn0 >= 0.0, large_neg, large_pos)
    )

    # Convert to fixed-point integers
    delta_plus_int = (delta_plus * scale_factor).round().int()
    delta_minus_int = (delta_minus * scale_factor).round().int()
    a_sign = torch.sign(a).int()

    return delta_plus_int, delta_minus_int, a_sign, scale_factor

def hardware_threshold_activation(x, thresholds_tuple, pool_kernel=7, pool_stride=2):
    """Hardware-friendly threshold activation using integer arithmetic (integer maxpool)."""
    delta_plus_int, delta_minus_int, a_sign, scale_factor = thresholds_tuple

    # Round input to int (safe if x is float) and convert to int32
    x_int = torch.round(x.float()).to(torch.int32)

    # Integer MaxPool
    x_pooled = int_maxpool1d(x_int, kernel_size=pool_kernel, stride=pool_stride)  # int32

    # Scale (to match threshold scale)
    x_scaled = (x_pooled.to(torch.int64) * int(scale_factor))  # int64
    B, C, L = x_scaled.shape

    dp = delta_plus_int.view(1, C, 1).expand(B, C, L).to(torch.int64)
    dm = delta_minus_int.view(1, C, 1).expand(B, C, L).to(torch.int64)
    a_s = a_sign.view(1, C, 1).expand(B, C, L)

    # Comparisons done in integer domain
    mask_pos = x_pooled >= 0
    cond_pos = mask_pos & (x_scaled >= dp)

    mask_neg = ~mask_pos
    cond_neg_when_a_pos = mask_neg & (x_scaled >= dm)
    cond_neg_when_a_neg = mask_neg & (x_scaled <= dm)
    a_nonneg = a_s >= 0
    cond_neg = torch.where(a_nonneg, cond_neg_when_a_pos, cond_neg_when_a_neg)

    out = torch.where(cond_pos | cond_neg,
                      torch.ones_like(x_pooled, dtype=torch.float32),
                      -torch.ones_like(x_pooled, dtype=torch.float32))
    return out

def hardware_final_layer(x, final_params, pool_kernel=7, pool_stride=2, scale_factor=2**10):
    """Hardware-friendly final affine layer using integer maxpool (keeps final rescale)."""
    alpha_pos_int, alpha_neg_int, beta_int = final_params

    # Round to ints then integer maxpool
    x_int = torch.round(x.float()).to(torch.int32)
    x_pooled = int_maxpool1d(x_int, kernel_size=pool_kernel, stride=pool_stride)  # int32

    # Scale
    x_scaled = (x_pooled.to(torch.int64) * int(scale_factor))  # int64

    B, C, L = x_scaled.shape
    a_pos = alpha_pos_int.view(1, C, 1).to(torch.int64).expand(B, C, L)
    a_neg = alpha_neg_int.view(1, C, 1).to(torch.int64).expand(B, C, L)
    b = beta_int.view(1, C, 1).to(torch.int64).expand(B, C, L)

    mask_pos = x_pooled >= 0
    out_pos = (a_pos * x_scaled + b * int(scale_factor))
    out_neg = (a_neg * x_scaled + b * int(scale_factor))

    out = torch.where(mask_pos, out_pos, out_neg).float()
    out = out / (scale_factor * scale_factor)

    return out

# ---------------------------
#  Model
# ---------------------------

class ECG_BNN_Int:
    def __init__(self, num_classes=5, scale_factor=2**12):
        self.num_classes = num_classes
        self.scale_factor = scale_factor
        self.block_configs = [
            [1, 8, 7, 2, 5, 7, 2],      # Block 1
            [8, 16, 7, 1, 5, 7, 2],     # Block 2
            [16, 32, 7, 1, 5, 7, 2],    # Block 3
            [32, 32, 7, 1, 5, 7, 2],    # Block 4
            [32, 64, 7, 1, 5, 7, 2],    # Block 5
            [64, num_classes, 7, 1, 5, 7, 2],  # Block 6
        ]
        self.binary_weights = []
        self.integer_thresholds = []
        self.final_params = None

    def load_weights_and_convert(self, state_dict_path):
        print(f"Loading weights from: {state_dict_path}")
        sd = torch.load(state_dict_path, map_location='cpu')

        # Binarized conv weights
        self.binary_weights = []
        for i in range(6):
            conv_key = f'block{i+1}.conv.weight'
            w = sd[conv_key]
            self.binary_weights.append(sign_binarize(w))
            print(f"Block {i+1}: Binarized weights {w.shape}")

        # Integer thresholds for blocks 1–5
        self.integer_thresholds = []
        for i in range(5):
            block_name = f'block{i+1}'
            out_ch = self.block_configs[i][1]
            bn_w = sd.get(f'{block_name}.bn.weight', torch.ones(out_ch))
            bn_b = sd.get(f'{block_name}.bn.bias', torch.zeros(out_ch))
            bn_m = sd.get(f'{block_name}.bn.running_mean', torch.zeros(out_ch))
            bn_v = sd.get(f'{block_name}.bn.running_var', torch.ones(out_ch))
            pre_w = sd.get(f'{block_name}.prelu.weight', torch.tensor([0.01]))
            thresholds = compute_fixed_point_thresholds(
                bn_w, bn_b, bn_m, bn_v, pre_w, self.scale_factor)
            self.integer_thresholds.append(thresholds)
            print(f"Block {i+1}: Integer thresholds ready")

        # Final affine params
        bname = 'block6'
        bn_w = sd.get(f'{bname}.bn.weight', torch.ones(self.num_classes)).float()
        bn_b = sd.get(f'{bname}.bn.bias', torch.zeros(self.num_classes)).float()
        bn_m = sd.get(f'{bname}.bn.running_mean', torch.zeros(self.num_classes)).float()
        bn_v = sd.get(f'{bname}.bn.running_var', torch.ones(self.num_classes)).float()
        pre_w = sd.get(f'{bname}.prelu.weight', torch.tensor([0.01]))

        if isinstance(pre_w, (float, int)):
            a = torch.full_like(bn_m, float(pre_w))
        else:
            a = pre_w.float().view(-1)
            if a.numel() == 1:
                a = a.expand_as(bn_m)

        s = torch.sqrt(bn_v + 1e-5)
        alpha = bn_w / s
        beta_affine = bn_b - alpha * bn_m

        alpha_pos = alpha
        alpha_neg = alpha * a

        alpha_pos_int = (alpha_pos * self.scale_factor).round().int()
        alpha_neg_int = (alpha_neg * self.scale_factor).round().int()
        beta_int = (beta_affine * self.scale_factor).round().int()

        self.final_params = (alpha_pos_int, alpha_neg_int, beta_int)
        print("Final affine params ready (int)")

    def preprocess_input(self, ecg_signal):
        ecg_signal = torch.from_numpy(ecg_signal).float()
        ecg_signal = ecg_signal.unsqueeze(0).unsqueeze(0)
        mean = ecg_signal.mean(dim=-1, keepdim=True)
        std = ecg_signal.std(dim=-1, keepdim=True)
        return (ecg_signal - mean) / (std + 1e-8)

    def forward(self, x):
        x = self.preprocess_input(x)

        # Block 1 (real conv + int activation)
        cfg = self.block_configs[0]
        x = real_conv1d(x, self.binary_weights[0], stride=cfg[3], padding=cfg[4])
        x = hardware_threshold_activation(x, self.integer_thresholds[0], cfg[5], cfg[6])

        # Blocks 2–5 (binary conv + int activation)
        for i in range(1,5):
            cfg = self.block_configs[i]
            x = xnor_popcount_conv1d(x, self.binary_weights[i], stride=cfg[3], padding=cfg[4])
            x = hardware_threshold_activation(x, self.integer_thresholds[i], cfg[5], cfg[6])

        # Block 6 (final affine int)
        cfg = self.block_configs[5]
        x = xnor_popcount_conv1d(x, self.binary_weights[5], stride=cfg[3], padding=cfg[4])
        x = hardware_final_layer(x, self.final_params, cfg[5], cfg[6], self.scale_factor)

        x = x.sum(dim=2)
        idx = torch.argmax(x, dim=1).item()
        return idx

    def predict(self, ecg_signal):
        pred_idx = self.forward(ecg_signal)
        return pred_idx, LABELS_5[pred_idx]

# ---------------------------
#  Utils + Main
# ---------------------------

def load_ecg_npy(path):
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        arr = np.array(arr.tolist())
    arr = np.asarray(arr).squeeze()
    if arr.ndim != 1 or arr.shape[0] != 3600:
        raise ValueError(f"Expected 1D array of length 3600, got shape {arr.shape}")
    return arr

def sample_test_files(data_root, num_samples=5):
    random.seed(33)
    files = []
    for class_dir in sorted(Path(data_root).glob('*')):
        if class_dir.is_dir():
            files.extend(class_dir.glob('*.npy'))
    if len(files) == 0:
        raise FileNotFoundError(f"No .npy files in {data_root}")
    if num_samples >= len(files):
        return files
    return random.sample(files, num_samples)

def main():
    parser = argparse.ArgumentParser(description="ECG-BNN Inference (integer thresholds + int maxpool)")
    parser.add_argument("--weights", type=str, default="ECG_BNN_state_dict.pth")
    parser.add_argument("--data-root", type=str, default="ECG_Dataset/ECG-5")
    parser.add_argument("--num-samples", type=int, default=10)
    args = parser.parse_args()

    if not os.path.exists(args.weights):
        print(f"Model weights not found: {args.weights}"); return 1
    if not os.path.exists(args.data_root):
        print(f"Dataset not found: {args.data_root}"); return 1

    model = ECG_BNN_Int(num_classes=5, scale_factor=2**12)
    model.load_weights_and_convert(args.weights)

    test_files = sample_test_files(args.data_root, args.num_samples)
    print(f"Selected {len(test_files)} test files")

    correct = 0
    for i, fp in enumerate(test_files, 1):
        print(f"\n--- Test {i}: {Path(fp).name} ---")
        ecg = load_ecg_npy(fp)
        gt_idx = int(Path(fp).parent.name.split()[0]) - 1
        gt_label = LABELS_5[gt_idx]
        pred_idx, pred_label = model.predict(ecg)
        ok = pred_idx == gt_idx
        if ok: correct += 1
        print(f"GT: {gt_label}  Pred: {pred_label}  {'✓' if ok else '✗'}")

    accuracy = correct / len(test_files) * 100
    print(f"\nAccuracy: {accuracy:.1f}% ({correct}/{len(test_files)})")
    return 0

if __name__ == "__main__":
    exit(main())
