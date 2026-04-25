"""Core inference functionality for T1w to T2 TSE translation."""

import torch
import torch.nn as nn
from tqdm import tqdm


class AutoregressiveFlowMatcher(nn.Module):
    """Autoregressive Flow Matching for slice-by-slice MRI inference."""

    def __init__(self, model, sigma_min=0.001):
        super().__init__()
        self.model = model
        self.sigma_min = sigma_min

    @torch.no_grad()
    def sample(self, condition, prev_slice, num_steps=50, device="cuda"):
        """ODE integration using Euler method.

        Args:
            condition: Conditioning input (T1w slice), shape (1, 1, H, W)
            prev_slice: Previous generated slice for autoregressive context
            num_steps: Number of Euler integration steps
            device: torch device
        """
        batch_size = condition.shape[0]
        x = torch.randn_like(condition).to(device)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full((batch_size,), i * dt, device=device)
            model_input = torch.cat([x, condition, prev_slice], dim=1)
            timesteps = (t * 999).long()
            v = self.model(model_input, timesteps)
            x = torch.clamp(x + v * dt, -3, 3)

        return x

    @torch.no_grad()
    def sample_rk4(self, condition, prev_slice, num_steps=50, device="cuda"):
        """ODE integration using 4th-order Runge-Kutta method.

        More accurate than Euler but requires 4x model evaluations per step.
        """
        batch_size = condition.shape[0]
        x = torch.randn_like(condition).to(device)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t_cur = i * dt
            t_b = torch.full((batch_size,), t_cur, device=device)
            k1 = self.model(torch.cat([x, condition, prev_slice], dim=1), (t_b * 999).long())

            t_mid = t_cur + 0.5 * dt
            t_mid_b = torch.full((batch_size,), t_mid, device=device)
            k2 = self.model(torch.cat([x + 0.5 * k1 * dt, condition, prev_slice], dim=1), (t_mid_b * 999).long())
            k3 = self.model(torch.cat([x + 0.5 * k2 * dt, condition, prev_slice], dim=1), (t_mid_b * 999).long())

            t_next = min((i + 1) * dt, 1.0)
            t_next_b = torch.full((batch_size,), t_next, device=device)
            k4 = self.model(torch.cat([x + k3 * dt, condition, prev_slice], dim=1), (t_next_b * 999).long())

            x = torch.clamp(x + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6.0, -3, 3)

        return x


class DirectInference(nn.Module):
    """Single-pass inference for baseline and pix2pix models."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    @torch.no_grad()
    def sample(self, condition, prev_slice, device="cuda"):
        zeros = torch.zeros_like(condition)
        model_input = torch.cat([zeros, condition, prev_slice], dim=1)
        timesteps = torch.zeros(condition.shape[0], dtype=torch.long, device=device)
        return self.model(model_input, timesteps)


def direct_inference(inferencer, mprage_volume, args, device):
    """Slice-by-slice inference for baseline/pix2pix models.

    Args:
        inferencer: DirectInference instance
        mprage_volume: Tensor (num_slices, H, W)
        args: Namespace with .no_fp16, .no_auto
        device: torch.device

    Returns:
        generated_volume: Tensor (num_slices, H, W)
        mprage_volume: Input tensor (returned for convenience)
    """
    num_slices, height, width = mprage_volume.shape
    generated_volume = torch.zeros_like(mprage_volume)
    prev_slice = torch.zeros((1, 1, height, width), device=device)

    use_fp16 = not args.no_fp16
    autoregressive = not args.no_auto

    print(f"Generating {num_slices} slices...")

    for slice_idx in tqdm(range(num_slices), desc="Generating slices"):
        mprage_slice = mprage_volume[slice_idx : slice_idx + 1].to(device).unsqueeze(0)

        with torch.no_grad():
            if use_fp16 and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    gen = inferencer.sample(mprage_slice, prev_slice, device)
            else:
                gen = inferencer.sample(mprage_slice, prev_slice, device)

            generated_volume[slice_idx] = gen.squeeze(0).squeeze(0).float().cpu()
            prev_slice = gen.detach().float() if autoregressive else torch.zeros((1, 1, height, width), device=device)

    return generated_volume, mprage_volume


def tse_flow_matching_inference(flow_matcher, mprage_volume, args, device):
    """Slice-by-slice autoregressive flow matching for T1w→T2 TSE translation.

    Args:
        flow_matcher: AutoregressiveFlowMatcher instance
        mprage_volume: Tensor (num_slices, H, W) — normalised and permuted
        args: Namespace with .steps, .no_fp16, .rk4, .no_auto
        device: torch.device

    Returns:
        generated_volume: Tensor (num_slices, H, W)
        mprage_volume: Input tensor (returned for convenience)
    """
    num_slices, height, width = mprage_volume.shape
    generated_volume = torch.zeros_like(mprage_volume)
    prev_slice = torch.zeros((1, 1, height, width), device=device)

    use_fp16 = not args.no_fp16
    use_euler = not getattr(args, "rk4", False)
    autoregressive = not args.no_auto

    print(f"Generating {num_slices} slices using flow matching (TSE)...")

    for slice_idx in tqdm(range(num_slices), desc="Generating slices"):
        mprage_slice = mprage_volume[slice_idx : slice_idx + 1].to(device).unsqueeze(0)

        with torch.no_grad():
            if use_fp16 and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    gen = (
                        flow_matcher.sample(mprage_slice, prev_slice, args.steps, device)
                        if use_euler
                        else flow_matcher.sample_rk4(mprage_slice, prev_slice, args.steps, device)
                    )
            else:
                gen = (
                    flow_matcher.sample(mprage_slice, prev_slice, args.steps, device)
                    if use_euler
                    else flow_matcher.sample_rk4(mprage_slice, prev_slice, args.steps, device)
                )

            generated_volume[slice_idx] = gen.squeeze(0).squeeze(0).cpu()
            prev_slice = gen.detach() if autoregressive else torch.zeros((1, 1, height, width), device=device)

    return generated_volume, mprage_volume
