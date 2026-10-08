"""Geometric perturbation versus unsigned latent-magnitude correlation.

For each sample, one Gaussian latent is denoised to an image. That same image is
then copied through every condition: no transform, horizontal shifts, vertical
shifts, and rotations. Each copy is DDIM-inverted, and the score is the Pearson
correlation of |z| and |z_hat|.

Padding: pixels that enter the frame are black, RGB (0, 0, 0). Translations are
integer nearest-neighbor shifts, so a 1 px move is exact. Positive horizontal
magnitude moves image content to the right. Positive vertical magnitude moves
it down. Rotations are bicubic, counterclockwise about the center, and are
cropped back to the original resolution (PIL expand=False).

Latent tensors are not saved. One image is reused across the conditions of a
sample, so the only intended difference among those rows is the transform.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TRANSLATION_PIXELS = (1, 2, 5, 10)
ROTATION_DEGREES = (1.0, 2.0, 5.0, 10.0)
FILL = (0, 0, 0)
EXAMPLE_SAMPLE = 0
EXAMPLE_MAGNITUDES = {
    "baseline": 0.0,
    "horizontal_translation": 5.0,
    "vertical_translation": 5.0,
    "rotation": 5.0,
}


def conditions() -> list[tuple[str, float]]:
    rows = [("baseline", 0.0)]
    rows += [("horizontal_translation", float(px)) for px in TRANSLATION_PIXELS]
    rows += [("vertical_translation", float(px)) for px in TRANSLATION_PIXELS]
    rows += [("rotation", float(deg)) for deg in ROTATION_DEGREES]
    return rows


def apply_geometric(image: Image.Image, kind: str, magnitude: float) -> Image.Image:
    """Return a same-size RGB image. New pixels are black. See module docstring."""
    image = image.convert("RGB")
    if image.size[0] != image.size[1]:
        raise ValueError(f"expected a square image, got {image.size}")
    if kind == "baseline":
        if magnitude != 0:
            raise ValueError("baseline magnitude must be 0")
        return image.copy()
    if kind == "horizontal_translation":
        return _translate(image, dx=int(magnitude), dy=0)
    if kind == "vertical_translation":
        return _translate(image, dx=0, dy=int(magnitude))
    if kind == "rotation":
        return image.rotate(
            magnitude,
            resample=Image.Resampling.BICUBIC,
            expand=False,
            fillcolor=FILL,
        )
    raise ValueError(f"unknown transformation: {kind}")


def _translate(image: Image.Image, dx: int, dy: int) -> Image.Image:
    # Output pixel (x, y) is read from (x - dx, y - dy), so content moves right/down.
    return image.transform(
        image.size,
        Image.Transform.AFFINE,
        (1, 0, -dx, 0, 1, -dy),
        resample=Image.Resampling.NEAREST,
        fillcolor=FILL,
    )


def magnitude_correlation(z: np.ndarray, z_hat: np.ndarray) -> float:
    """Pearson correlation of unsigned latent magnitudes."""
    if z.shape != z_hat.shape:
        raise ValueError(f"latent shape {z.shape} != recovered shape {z_hat.shape}")
    if z.size < 2:
        raise ValueError("need at least two coordinates for a correlation")
    left = np.abs(np.asarray(z, dtype=np.float64)).reshape(-1)
    right = np.abs(np.asarray(z_hat, dtype=np.float64)).reshape(-1)
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("latent magnitudes contain non-finite values")
    result = pearsonr(left, right)
    corr = float(result.statistic) if hasattr(result, "statistic") else float(result[0])
    if not np.isfinite(corr) or corr < -1.0 or corr > 1.0:
        raise ValueError(f"correlation {corr} is outside [-1, 1]")
    return corr


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/geometric_registration/run_001"),
    )
    parser.add_argument("--model-id", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--prompt", default="a photograph of an astronaut riding a horse")
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    args = parser.parse_args(argv)
    if args.num_samples < 1:
        parser.error("--num-samples must be positive")
    if args.num_inference_steps < 1:
        parser.error("--num-inference-steps must be positive")
    if args.image_size % 8 != 0:
        parser.error("--image-size must be a multiple of 8")
    return args


def _dtype_from_name(name: str):
    import torch

    table = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
    if name not in table:
        raise SystemExit(f"dtype must be one of {', '.join(table)}")
    return table[name]


def _complete_samples(csv_path: Path, n_conditions: int) -> tuple[pd.DataFrame, set[int]]:
    if not csv_path.is_file():
        return pd.DataFrame(columns=["sample_id", "transformation_type", "magnitude", "correlation"]), set()
    frame = pd.read_csv(csv_path)
    counts = frame.groupby("sample_id").size()
    done = set(int(i) for i, n in counts.items() if n == n_conditions)
    return frame[frame.sample_id.isin(done)].copy(), done


def _save_example(image: Image.Image, example_dir: Path, sample_id: int, kind: str, magnitude: float) -> None:
    if sample_id != EXAMPLE_SAMPLE or magnitude != EXAMPLE_MAGNITUDES.get(kind):
        return
    example_dir.mkdir(parents=True, exist_ok=True)
    label = "baseline" if kind == "baseline" else f"{kind}_{magnitude:g}"
    image.save(example_dir / f"sample{sample_id:03d}_{label}.png")


def _generate_samples(args, pending: list[int], dtype):
    import torch

    from experiments.generate_and_invert import (
        _load_generation_pipeline,
        _release_pipeline,
        _sample_latent,
    )

    pipe = _load_generation_pipeline(args.model_id, dtype, args.device)
    if float(pipe.scheduler.init_noise_sigma) != 1.0:
        raise SystemExit("DDIM init_noise_sigma is not 1; refusing to rescale the supplied latent")
    master = np.random.default_rng(args.seed)
    sample_seeds = master.integers(0, np.iinfo(np.int32).max, size=args.num_samples)
    generated = []
    for sample_id in pending:
        z = _sample_latent(pipe, int(sample_seeds[sample_id]), args.image_size, args.image_size, dtype)
        result = pipe(
            prompt=args.prompt,
            latents=z.to(args.device),
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            height=args.image_size,
            width=args.image_size,
        )
        if len(result.images) != 1:
            raise SystemExit(f"sample {sample_id} produced {len(result.images)} images")
        image = result.images[0]
        if image.size != (args.image_size, args.image_size):
            raise SystemExit(f"sample {sample_id} image size {image.size}")
        generated.append((sample_id, int(sample_seeds[sample_id]), z.detach().cpu().float(), image))
        print(f"generated sample {sample_id}", flush=True)
    _release_pipeline(pipe)
    return generated, sample_seeds


def _invert_samples(args, generated, dtype, spec, example_dir: Path, csv_path: Path, done_frame: pd.DataFrame) -> None:
    import torch

    from experiments.generate_and_invert import (
        _load_inversion_pipeline,
        _release_pipeline,
        _recovered_latent,
    )

    inverse = _load_inversion_pipeline(args.model_id, dtype, args.device)
    frame = done_frame
    try:
        for sample_id, _seed, z, image in generated:
            z_np = z.numpy()
            rows = []
            for kind, magnitude in spec:
                transformed = apply_geometric(image, kind, magnitude)
                if transformed.size != image.size:
                    raise SystemExit(f"{kind} {magnitude} changed the image size")
                _save_example(transformed, example_dir, sample_id, kind, magnitude)
                inverted = inverse.invert(
                    prompt=args.prompt,
                    image=transformed,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    inpaint_strength=1.0,
                    num_reg_steps=0,
                )
                z_hat = _recovered_latent(inverted.latents, z.shape)
                z_hat = z_hat.detach().to(device="cpu", dtype=torch.float32).numpy()
                rows.append(
                    {
                        "sample_id": sample_id,
                        "transformation_type": kind,
                        "magnitude": magnitude,
                        "correlation": magnitude_correlation(z_np, z_hat),
                    }
                )
            frame = pd.concat([frame, pd.DataFrame(rows)], ignore_index=True)
            frame.to_csv(csv_path, index=False)
            print(f"inverted sample {sample_id}", flush=True)
    finally:
        _release_pipeline(inverse)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    import torch

    from experiments.generate_and_invert import _git_commit

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda was requested but torch.cuda.is_available() is false")
    dtype = _dtype_from_name(args.dtype)
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    example_dir = output_dir / "examples"
    spec = conditions()
    csv_path = output_dir / "correlations.csv"
    done_frame, done = _complete_samples(csv_path, len(spec))
    pending = [i for i in range(args.num_samples) if i not in done]
    print(f"{len(done)} samples already complete, {len(pending)} remaining", flush=True)

    if pending:
        generated, sample_seeds = _generate_samples(args, pending, dtype)
        _invert_samples(args, generated, dtype, spec, example_dir, csv_path, done_frame)
    else:
        master = np.random.default_rng(args.seed)
        sample_seeds = master.integers(0, np.iinfo(np.int32).max, size=args.num_samples)

    frame = pd.read_csv(csv_path) if csv_path.is_file() else done_frame

    metadata = {
        "seed": args.seed,
        "num_samples": args.num_samples,
        "model_id": args.model_id,
        "image_size": args.image_size,
        "latent_shape": [1, 4, args.image_size // 8, args.image_size // 8],
        "dtype": args.dtype,
        "device": args.device,
        "prompt": args.prompt,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "scheduler": "DDIMScheduler",
        "inverse_scheduler": "DDIMInverseScheduler",
        "inversion_api": "StableDiffusionDiffEditPipeline.invert",
        "inversion_strength": 1.0,
        "padding": "constant black RGB (0, 0, 0)",
        "translation_resample": "nearest",
        "rotation_resample": "bicubic",
        "rotation_direction": "counterclockwise about the image center",
        "translation_direction": "positive horizontal moves content right; positive vertical moves content down",
        "correlation": "pearsonr(|z|, |z_hat|)",
        "conditions": [{"transformation_type": k, "magnitude": m} for k, m in spec],
        "sample_seeds": [int(s) for s in sample_seeds],
        "git_commit": _git_commit(),
        "results_csv": csv_path.name,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Samples: {frame.sample_id.nunique()}")
    print(f"Rows: {len(frame)}")
    print(f"Results: {csv_path}")
    print(f"Metadata: {output_dir / 'metadata.json'}")
    print(f"Examples: {example_dir}")


if __name__ == "__main__":
    main()
