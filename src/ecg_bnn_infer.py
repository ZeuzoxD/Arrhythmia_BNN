import torch
import torch.nn.functional as F
import numpy as np
import argparse, os, random
from pathlib import Path

LABELS = ['N', 'S', 'V', 'F', 'Q']

def pad_1d(x, pad, val=1):
    if pad == 0: return x
    B, C, L = x.shape
    left = torch.full((B, C, pad), val, dtype=x.dtype, device=x.device)
    right = torch.full((B, C, pad), val, dtype=x.dtype, device=x.device)
    return torch.cat([left, x, right], dim=2)

def maxpool1d_int(x, k=7, s=2):
    return x.unfold(2, k, s).amax(dim=3)

def binarize_to_bits(x):
    return (x > 0).to(torch.uint8)

def binary_conv1d(x, w, stride=1, padding=0):
    B, Cin, L = x.shape
    Cout, _, K = w.shape
    if padding > 0:
        x = pad_1d(x, padding, val=1)
    x_unf = x.unfold(2, K, stride)          # [B,Cin,Lout,K]
    B, Cin, Lout, K = x_unf.shape

    x_bin = binarize_to_bits(x_unf)
    w_bin = binarize_to_bits(w).unsqueeze(0)

    x_bin = x_bin.permute(0, 2, 1, 3)
    x_bin = x_bin.unsqueeze(1)
    w_bin = w_bin.unsqueeze(2)

    xnor = (x_bin == w_bin)
    popcount = xnor.sum(dim=(-1, -2), dtype=torch.int32)

    out = 2 * popcount - (Cin * K)
    return out.to(torch.int32)

def threshold_activation(x, params, pool_k=7, pool_s=2):
    dp, dm, a_sign, scale = params
    x_int = torch.round(x.float()).to(torch.int32)

    x_pool = maxpool1d_int(x_int, pool_k, pool_s)
    x_scaled = x_pool * int(scale)


    dp = dp.to(torch.int32).view(1,-1,1)
    dm = dm.to(torch.int32).view(1,-1,1)
    a_sign = a_sign.to(torch.int32).view(1,-1,1)

    pos_mask = x_pool >= 0
    cond_pos = pos_mask & (x_scaled >= dp)

    neg_mask = ~pos_mask
    cond_neg = torch.where(
        a_sign >= 0,

        neg_mask & (x_scaled >= dm),
        neg_mask & (x_scaled <= dm)
    )
    out = torch.where(cond_pos | cond_neg, torch.ones_like(x_pool, dtype=torch.int8), -torch.ones_like(x_pool, dtype=torch.int8))
    return out

def final_layer(x, params, pool_k=7, pool_s=2, scale=1):
    a_pos, a_neg, b = params
    x_pool = maxpool1d_int(x.to(torch.int32), pool_k, pool_s)
    x_scaled = x_pool * int(scale)
    a_pos = a_pos.to(torch.int32).view(1,-1,1)
    a_neg = a_neg.to(torch.int32).view(1,-1,1)
    b = b.to(torch.int32).view(1,-1,1)
    return torch.where(x_pool >= 0, a_pos * x_scaled + b, a_neg * x_scaled + b)


class ECG_BNN:
    def __init__(self, params_dir="ecg_params", num_classes=5):
        self.params_dir = params_dir
        self.num_classes = num_classes
        self.blocks = [
            [1, 8, 7, 2, 5, 7, 2],
            [8, 16, 7, 1, 5, 7, 2],
            [16, 32, 7, 1, 5, 7, 2],
            [32, 32, 7, 1, 5, 7, 2],

            [32, 64, 7, 1, 5, 7, 2],
            [64, num_classes, 7, 1, 5, 7, 2],
        ]
        self._load_params()

    def _load_params(self):
        self.weights = [torch.load(f"{self.params_dir}/conv{i+1}_weight.pt").to(torch.int8) for i in range(6)]
        self.thresholds = []
        for i in range(5):
            dp, dm, a_sign, scale = torch.load(f"{self.params_dir}/block{i+1}_thresholds.pt")
            self.thresholds.append((dp.to(torch.int32), dm.to(torch.int32), a_sign.to(torch.int32), int(scale)))
        a_pos, a_neg, b = torch.load(f"{self.params_dir}/final_params.pt")
        self.final_params = (a_pos.to(torch.int32), a_neg.to(torch.int32), b.to(torch.int32))

    def preprocess(self, sig):

        sig = torch.from_numpy(sig).float().unsqueeze(0).unsqueeze(0)
        return (sig - sig.mean(dim=-1, keepdim=True)) / (sig.std(dim=-1, keepdim=True) + 1e-8)


    def forward(self, sig):
        x = self.preprocess(sig)
        cfg = self.blocks[0]
        x = F.pad(x, (cfg[4], cfg[4]), value=1.0) if cfg[4] > 0 else x
        x = F.conv1d(x, self.weights[0].float(), stride=cfg[3])
        x = threshold_activation(x, self.thresholds[0], cfg[5], cfg[6])

        for i in range(1,5):
            cfg = self.blocks[i]
            x = binary_conv1d(x, self.weights[i], stride=cfg[3], padding=cfg[4])
            x = threshold_activation(x, self.thresholds[i], cfg[5], cfg[6])
        cfg = self.blocks[5]
        x = binary_conv1d(x, self.weights[5], stride=cfg[3], padding=cfg[4])
        x = final_layer(x, self.final_params, cfg[5], cfg[6])
        logits = x.sum(dim=2)
        max = torch.argmax(logits, dim=1).item()
        return max, logits

    def predict(self, sig):
        idx, logits = self.forward(sig)
        return idx, LABELS[idx], logits

def load_ecg(path):
    arr = np.load(path, allow_pickle=True)
    if arr.dtype == object: arr = np.array(arr.tolist())
    return np.asarray(arr).squeeze()


def sample_files(root, n=50):
    files = []
    for d in sorted(Path(root).glob('*')):
        if d.is_dir(): files.extend(d.glob('*.npy'))
    random.seed(33)
    return random.sample(files, min(n, len(files)))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--params", type=str, default="ecg_params")
    p.add_argument("--data", type=str, default="ECG_Dataset/ECG-5")
    p.add_argument("--n", type=int, default=50)
    args = p.parse_args()

    model = ECG_BNN(params_dir=args.params)
    files = sample_files(args.data, args.n)
    correct = 0

    for i, f in enumerate(files, 1):
        sig = load_ecg(f)
        gt = int(Path(f).parent.name.split()[0]) - 1
        idx, lbl, _ = model.predict(sig)
        if idx == gt: correct += 1
        print(f"{i}: {Path(f).name}: GT={LABELS[gt]} Pred={lbl} {'✓' if correct else '✗'}")

    print(f"Accuracy: {correct/len(files)*100:.2f}% ({correct}/{len(files)})")

if __name__ == "__main__":
    main()
