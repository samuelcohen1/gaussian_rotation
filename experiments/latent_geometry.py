"""Baseline spatial transforms for Stable Diffusion latents.

Image-space transforms live in ``scripts/run_geometric_registration_experiment.py``.
This module is the latent-space counterpart used to measure equivariance.

Assumptions, none of which imply exact equivariance:

- The VAE compresses each spatial axis by ``vae_scale_factor`` (8 for SD1.5).
  A pixel shift of ``n`` becomes a latent shift of ``n / vae_scale_factor``.
- Positive horizontal shift moves content right. Positive vertical shift moves
  content down. Vacated locations are filled with 0. That 0 is not the VAE
  encoding of a black pixel.
- Integer latent shifts are exact copies. A fractional shift uses nearest
  neighbor, matching the image-space translation.
- Rotation is counterclockwise about the center, independently on each channel.
  It uses a cubic spline with constant 0 fill. It is not the same kernel as
  PIL bicubic, and it does not mix latent channels.
- The equivariance error is
  ``||z' - T(z)||_2 / (||z'||_2 + eps)``.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

EPS = 1e-8


def selected_step_indices(num_steps: int) -> list[int]:
    """First, middle, and last inversion steps. Fewer if the trajectory is short."""
    if num_steps < 1:
        raise ValueError("num_steps must be positive")
    if num_steps == 1:
        return [0]
    if num_steps == 2:
        return [0, 1]
    return [0, num_steps // 2, num_steps - 1]


def undo_image_transform(
    latent: np.ndarray,
    kind: str,
    magnitude: float,
    vae_scale_factor: int,
) -> np.ndarray:
    """Apply the opposite of an image-space shift or rotation on a latent.

    A right or down image shift is undone by a left or up latent shift of
    ``magnitude / vae_scale_factor`` pixels. A counterclockwise image rotation
    is undone by a clockwise latent rotation of the same angle. Baseline is
    copied. Accepts ``(C, H, W)`` or ``(1, C, H, W)`` and returns the same rank.
    """
    batched = np.asarray(latent).ndim == 4
    spatial = np.asarray(latent)[0] if batched else np.asarray(latent)
    if kind == "baseline":
        corrected = apply_latent_transform(spatial, "baseline", 0.0, vae_scale_factor)
    elif kind in {"horizontal_translation", "vertical_translation", "rotation"}:
        corrected = apply_latent_transform(spatial, kind, -float(magnitude), vae_scale_factor)
    else:
        raise ValueError(f"unknown transformation: {kind}")
    corrected = corrected.astype(np.float32, copy=False)
    if batched:
        return corrected[None]
    return corrected


def apply_latent_transform(
    latent: np.ndarray,
    kind: str,
    magnitude: float,
    vae_scale_factor: int,
) -> np.ndarray:
    """Apply one spatial transform to a latent of shape (C, H, W)."""
    z = np.asarray(latent, dtype=np.float32)
    if z.ndim != 3:
        raise ValueError(f"expected (C, H, W), got {z.shape}")
    if kind == "baseline":
        if magnitude != 0:
            raise ValueError("baseline magnitude must be 0")
        return z.copy()
    if kind == "horizontal_translation":
        return _shift_channels(z, dx=_latent_shift(magnitude, vae_scale_factor), dy=0.0)
    if kind == "vertical_translation":
        return _shift_channels(z, dx=0.0, dy=_latent_shift(magnitude, vae_scale_factor))
    if kind == "rotation":
        return _rotate_channels(z, degrees=float(magnitude))
    raise ValueError(f"unknown transformation: {kind}")


def equivariance_error(z_transformed: np.ndarray, z_mapped: np.ndarray, eps: float = EPS) -> float:
    """``||z' - T(z)||_2 / (||z'||_2 + eps)``."""
    left = np.asarray(z_transformed, dtype=np.float64).ravel()
    right = np.asarray(z_mapped, dtype=np.float64).ravel()
    return float(np.linalg.norm(left - right) / (np.linalg.norm(left) + eps))


def normalized_difference(z_transformed: np.ndarray, z_original: np.ndarray, eps: float = EPS) -> float:
    """``||z' - z||_2 / (||z'||_2 + eps)``. Same denominator as the equivariance error."""
    return equivariance_error(z_transformed, z_original, eps=eps)


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = EPS) -> float:
    left = np.asarray(a, dtype=np.float64).ravel()
    right = np.asarray(b, dtype=np.float64).ravel()
    denom = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / (denom + eps))


def fit_shared_pca(latents: list[np.ndarray], n_components: int = 3) -> dict:
    """Fit one PCA on channel vectors drawn from every provided latent.

    Each latent is (C, H, W). Every spatial site is one observation. The same
    mean and components are later applied to every map, including maps that
    were not in this list only if the caller chooses to do that. This helper
    records the min and max of the projected training points so RGB scaling
    is shared too.
    """
    if len(latents) < 1:
        raise ValueError("need at least one latent")
    blocks = []
    channels = None
    for latent in latents:
        z = np.asarray(latent, dtype=np.float64)
        if z.ndim != 3:
            raise ValueError(f"expected (C, H, W), got {z.shape}")
        if channels is None:
            channels = z.shape[0]
        elif z.shape[0] != channels:
            raise ValueError("all latents must have the same channel count")
        blocks.append(z.reshape(channels, -1).T)
    observations = np.concatenate(blocks, axis=0)
    mean = observations.mean(axis=0)
    centered = observations - mean
    # Column observations are tiny (4 channels), so the full SVD is cheap.
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_components]
    projected = centered @ components.T
    return {
        "mean": mean.astype(np.float64),
        "components": components.astype(np.float64),
        "vmin": float(projected.min()),
        "vmax": float(projected.max()),
    }


def project_latent_rgb(latent: np.ndarray, model: dict) -> np.ndarray:
    """Map one (C, H, W) latent to an RGB image in [0, 1] with a shared PCA."""
    z = np.asarray(latent, dtype=np.float64)
    channels, height, width = z.shape
    flat = z.reshape(channels, -1).T
    rgb = (flat - model["mean"]) @ model["components"].T
    span = model["vmax"] - model["vmin"]
    if span == 0:
        scaled = np.zeros_like(rgb)
    else:
        scaled = (rgb - model["vmin"]) / span
    return np.clip(scaled.reshape(height, width, 3), 0.0, 1.0)


def _latent_shift(pixels: float, vae_scale_factor: int) -> float:
    if vae_scale_factor < 1:
        raise ValueError("vae_scale_factor must be positive")
    return float(pixels) / float(vae_scale_factor)


def _shift_channels(latent: np.ndarray, dx: float, dy: float) -> np.ndarray:
    return np.stack([_shift_2d(channel, dx, dy) for channel in latent], axis=0).astype(np.float32)


def _shift_2d(channel: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Move content by (dx, dy) latent pixels. Positive dx is right, positive dy is down."""
    if abs(dx - round(dx)) < 1e-6 and abs(dy - round(dy)) < 1e-6:
        return _shift_integer(channel, int(round(dx)), int(round(dy)))
    shifted = ndimage.shift(channel, shift=(dy, dx), order=0, mode="constant", cval=0.0)
    return shifted.astype(np.float32)


def _shift_integer(channel: np.ndarray, dx: int, dy: int) -> np.ndarray:
    out = np.zeros_like(channel)
    height, width = channel.shape
    src_x0 = max(0, -dx)
    src_x1 = min(width, width - dx)
    src_y0 = max(0, -dy)
    src_y1 = min(height, height - dy)
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return out
    dst_x0 = src_x0 + dx
    dst_y0 = src_y0 + dy
    dst_x1 = src_x1 + dx
    dst_y1 = src_y1 + dy
    out[dst_y0:dst_y1, dst_x0:dst_x1] = channel[src_y0:src_y1, src_x0:src_x1]
    return out


def _rotate_channels(latent: np.ndarray, degrees: float) -> np.ndarray:
    if abs(degrees) % 360 < 1e-8:
        return latent.copy()
    rotated = [
        ndimage.rotate(
            channel,
            angle=degrees,
            reshape=False,
            order=3,
            mode="constant",
            cval=0.0,
            prefilter=True,
        )
        for channel in latent
    ]
    return np.stack(rotated, axis=0).astype(np.float32)
