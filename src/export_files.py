import torch
import os
import argparse

from ecg_bnn_maxpool import sign_binarize, compute_fixed_point_thresholds

def export_params(weights_path, out_dir="ecg_params", scale_factor=2**10, num_classes=5):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print(f"Exporting to {out_dir}/ ...")

    sd = torch.load(weights_path, map_location="cpu")

    # --------------------
    # 1) Binarized Conv Weights
    # --------------------
    for i in range(6):
        conv_key = f'block{i+1}.conv.weight'
        w = sd[conv_key]
        w_bin = sign_binarize(w).to(torch.int8)   # store as -1/+1 int8
        torch.save(w_bin, os.path.join(out_dir, f"conv{i+1}_weight.pt"))
        print(f"Saved conv{i+1}_weight.pt  {tuple(w_bin.shape)}")


    # --------------------
    # 2) Thresholds (blocks 1–5)
    # --------------------

    for i in range(5):
        block_name = f'block{i+1}'
        out_ch = sd[f'{block_name}.bn.weight'].shape[0]

        bn_w = sd.get(f'{block_name}.bn.weight', torch.ones(out_ch))
        bn_b = sd.get(f'{block_name}.bn.bias', torch.zeros(out_ch))
        bn_m = sd.get(f'{block_name}.bn.running_mean', torch.zeros(out_ch))
        bn_v = sd.get(f'{block_name}.bn.running_var', torch.ones(out_ch))
        pre_w = sd.get(f'{block_name}.prelu.weight', torch.tensor([0.01]))


        thresholds = compute_fixed_point_thresholds(bn_w, bn_b, bn_m, bn_v, pre_w, scale_factor)

        torch.save(thresholds, os.path.join(out_dir, f"block{i+1}_thresholds.pt"))
        print(f"Saved block{i+1}_thresholds.pt")


    # --------------------
    # 3) Final affine params (block6)
    # --------------------
    bname = 'block6'
    bn_w = sd.get(f'{bname}.bn.weight', torch.ones(num_classes)).float()

    bn_b = sd.get(f'{bname}.bn.bias', torch.zeros(num_classes)).float()
    bn_m = sd.get(f'{bname}.bn.running_mean', torch.zeros(num_classes)).float()
    bn_v = sd.get(f'{bname}.bn.running_var', torch.ones(num_classes)).float()
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

    alpha_pos_int = (alpha_pos * scale_factor).round().int()
    alpha_neg_int = (alpha_neg * scale_factor).round().int()

    beta_int = (beta_affine * scale_factor).round().int()

    final_params = (alpha_pos_int, alpha_neg_int, beta_int)
    torch.save(final_params, os.path.join(out_dir, "final_params.pt"))
    print("Saved final_params.pt")

    print("✅ Export finished")


def main():
    parser = argparse.ArgumentParser(description="Export ECG-BNN params")
    parser.add_argument("--weights", type=str, default="ECG_BNN_state_dict.pth")
    parser.add_argument("--out-dir", type=str, default="ecg_params")
    parser.add_argument("--scale", type=int, default=2**12)
    args = parser.parse_args()

    export_params(args.weights, args.out_dir, args.scale)


if __name__ == "__main__":
    main()
