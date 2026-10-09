"""
Network Weight Interpolation for Super-Resolution (Pillar 2/3 Synergy).

Interpolates parameter space between:
- Theta_GAN: Pretrained APISR generator (high edge sharpness, adversarial textures)
- Theta_L1: Domain-adapted generator (broadcast compression deblocking, flat cel smoothing)

Formula:
  theta_interp = alpha * theta_GAN + (1 - alpha) * theta_L1

Pioneered in ESRGAN (Wang et al., ECCV 2018) to balance perceptual quality and artifact suppression
without runtime latency overhead (runs at the exact same speed as a single model).
"""

import os
import sys
import torch

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)


def interpolate_weights(
    path_gan: str = "model_zoo/2x_APISR_RRDB_GAN_generator.pth",
    path_l1: str = "checkpoints/apisr_reanime600_best.pth",
    alpha: float = 0.7,
    save_path: str = "checkpoints/apisr_interpolated_alpha07.pth",
) -> str:
    print("=" * 65)
    print("AIAnime — Network Weight Interpolation")
    print(f"Alpha (GAN weight): {alpha:.2f} | (1 - Alpha) (Domain-Adapted): {1.0 - alpha:.2f}")
    print("=" * 65)

    # 1. Load GAN state dict
    ckpt_gan = torch.load(path_gan, map_location="cpu", weights_only=False)
    state_gan = ckpt_gan["model_state_dict"] if isinstance(ckpt_gan, dict) and "model_state_dict" in ckpt_gan else ckpt_gan

    # 2. Load Domain-Adapted L1 state dict
    ckpt_l1 = torch.load(path_l1, map_location="cpu", weights_only=True)
    state_l1 = ckpt_l1["model_state_dict"] if isinstance(ckpt_l1, dict) and "model_state_dict" in ckpt_l1 else ckpt_l1

    # 3. Perform linear interpolation across all matching parameter tensors
    interpolated_state = {}
    matched_keys = 0
    for k in state_gan.keys():
        if k in state_l1:
            w_gan = state_gan[k].float()
            w_l1 = state_l1[k].float()
            # Linear interpolation in parameter space
            w_interp = alpha * w_gan + (1.0 - alpha) * w_l1
            interpolated_state[k] = w_interp.to(state_gan[k].dtype)
            matched_keys += 1
        else:
            interpolated_state[k] = state_gan[k]

    print(f"Successfully interpolated {matched_keys} parameter tensors.")

    # 4. Save interpolated weights
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(interpolated_state, save_path)
    file_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"Saved interpolated weights: {save_path} ({file_mb:.2f} MB)")
    print("=" * 65)

    return save_path


if __name__ == "__main__":
    alpha_val = float(sys.argv[1]) if len(sys.argv) > 1 else 0.7
    save_file = f"checkpoints/apisr_interpolated_alpha{int(alpha_val * 100):02d}.pth"
    interpolate_weights(alpha=alpha_val, save_path=save_file)
