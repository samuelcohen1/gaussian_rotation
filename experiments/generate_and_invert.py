"""Generate one image from a known Gaussian latent and invert it back to a latent.

This script does not decode watermarks, compute BER, or compare the two latents.
Those checks happen later, on the saved tensors.

Generation uses ``StableDiffusionPipeline`` with an explicit ``latents=`` tensor.
Inversion uses ``StableDiffusionDiffEditPipeline.invert`` with ``DDIMInverseScheduler``,
which is Diffusers' documented DDIM inversion API. ``inpaint_strength=1`` runs every
inverse step. The API returns the whole trajectory stacked on dimension 1, reversed,
so index 0 is the fully noised latent. Only that tensor is saved.
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
from pathlib import Path

import torch
from PIL import Image
from diffusers import (
    DDIMInverseScheduler,
    DDIMScheduler,
    StableDiffusionDiffEditPipeline,
    StableDiffusionPipeline,
)

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_ID = "runwayml/stable-diffusion-v1-5"
DEFAULT_PROMPT = "a photograph of an astronaut riding a horse"
# 1.0 runs the inverse scheduler for every step, so the saved latent is the
# fully noised end of the inversion, not a partial DiffEdit trajectory.
DEFAULT_INVERSION_STRENGTH = 1.0


def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    commit = out.strip()
    return commit or None


def _dtype(name: str) -> torch.dtype:
    table = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
    try:
        return table[name]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(f"dtype must be one of {', '.join(table)}") from exc


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate exactly one image from a known Gaussian latent and invert it."
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/diffusion_inversion/run_001"),
    )
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", type=_dtype, default=torch.float16)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument(
        "--inversion-strength",
        type=float,
        default=DEFAULT_INVERSION_STRENGTH,
        help="Passed as inpaint_strength to DiffEdit invert. 1.0 is a full inversion.",
    )
    args = parser.parse_args(argv)
    if args.num_inference_steps < 1:
        parser.error("--num-inference-steps must be positive")
    if not 0.0 < args.inversion_strength <= 1.0:
        parser.error("--inversion-strength must be in (0, 1]")
    if args.image_size % 8 != 0:
        parser.error("--image-size must be a multiple of 8")
    return args


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _sample_latent(pipe: StableDiffusionPipeline, seed: int, height: int, width: int, dtype: torch.dtype) -> torch.Tensor:
    """Draw the Gaussian latent that generation will actually consume.

    The generator is on CPU so the same seed yields the same tensor on any GPU.
    DDIM's init_noise_sigma is 1, so the pipeline does not rescale this sample.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    channels = pipe.unet.config.in_channels
    height_lat = height // pipe.vae_scale_factor
    width_lat = width // pipe.vae_scale_factor
    latent = torch.randn(
        (1, channels, height_lat, width_lat),
        generator=generator,
        dtype=torch.float32,
    )
    return latent.to(dtype=dtype)


def _pretrained_kwargs(dtype: torch.dtype) -> dict:
    return {"torch_dtype": dtype, "safety_checker": None, "use_safetensors": True}


def _load_generation_pipeline(model_id: str, dtype: torch.dtype, device: str) -> StableDiffusionPipeline:
    pipe = StableDiffusionPipeline.from_pretrained(model_id, **_pretrained_kwargs(dtype))
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    return pipe.to(device)


def _load_inversion_pipeline(model_id: str, dtype: torch.dtype, device: str) -> StableDiffusionDiffEditPipeline:
    # Loaded only after generation is freed. Two copies at once do not fit a small QOS memory cap.
    inverse = StableDiffusionDiffEditPipeline.from_pretrained(model_id, **_pretrained_kwargs(dtype))
    inverse.scheduler = DDIMScheduler.from_config(inverse.scheduler.config)
    inverse.inverse_scheduler = DDIMInverseScheduler.from_config(inverse.scheduler.config)
    return inverse.to(device)


def _release_pipeline(pipe) -> None:
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _recovered_latent(stacked: torch.Tensor, original_shape: torch.Size) -> torch.Tensor:
    """Keep the fully noised latent from DiffEdit's trajectory and drop the rest."""
    if tuple(stacked.shape) == tuple(original_shape):
        return stacked
    if stacked.ndim == len(original_shape) + 1 and tuple(stacked.shape[:1]) + tuple(stacked.shape[2:]) == tuple(original_shape):
        return stacked[:, 0]
    raise RuntimeError(
        f"inverted latent shape {tuple(stacked.shape)} does not match original {tuple(original_shape)}"
    )


def _verify(output_dir: Path, expected_shape: tuple[int, ...], image_size: int) -> None:
    image_path = output_dir / "generated_image.png"
    original_path = output_dir / "latent_original.pt"
    inverted_path = output_dir / "latent_inverted.pt"
    metadata_path = output_dir / "metadata.json"
    pngs = sorted(output_dir.glob("*.png"))

    problems: list[str] = []
    if pngs != [image_path]:
        problems.append(f"expected exactly one image at {image_path.name}, found {[p.name for p in pngs]}")
    for path in (original_path, inverted_path, metadata_path, image_path):
        if not path.is_file():
            problems.append(f"missing {path.name}")
    if problems:
        raise SystemExit("sanity check failed:\n- " + "\n- ".join(problems))

    original = torch.load(original_path, map_location="cpu", weights_only=True)
    inverted = torch.load(inverted_path, map_location="cpu", weights_only=True)
    if tuple(original.shape) != expected_shape:
        problems.append(f"original shape {tuple(original.shape)} != {expected_shape}")
    if tuple(inverted.shape) != expected_shape:
        problems.append(f"inverted shape {tuple(inverted.shape)} != {expected_shape}")
    if original.ndim == 0 or original.shape[0] != 1:
        problems.append("original latent batch size is not 1")
    if not torch.isfinite(original).all():
        problems.append("original latent has non-finite values")
    if not torch.isfinite(inverted).all():
        problems.append("inverted latent has non-finite values")

    try:
        with Image.open(image_path) as image:
            image.load()
            if image.size != (image_size, image_size):
                problems.append(f"image size {image.size} != {(image_size, image_size)}")
    except OSError as exc:
        problems.append(f"image could not be opened: {exc}")

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        problems.append(f"metadata.json is not valid JSON: {exc}")
        metadata = {}
    for key in (
        "model_id",
        "seed",
        "device",
        "dtype",
        "image_size",
        "run_id",
        "git_commit",
        "latent_original_path",
        "latent_inverted_path",
        "image_path",
        "scheduler",
        "inverse_scheduler",
        "num_inference_steps",
        "guidance_scale",
        "inversion_strength",
        "prompt",
    ):
        if key not in metadata:
            problems.append(f"metadata missing {key}")

    if problems:
        raise SystemExit("sanity check failed:\n- " + "\n- ".join(problems))


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda was requested but torch.cuda.is_available() is false")

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    pipe = _load_generation_pipeline(args.model_id, args.dtype, args.device)
    init_noise_sigma = float(pipe.scheduler.init_noise_sigma)
    if init_noise_sigma != 1.0:
        raise SystemExit(
            f"DDIM init_noise_sigma is {init_noise_sigma}, not 1. "
            "Refusing to run because the pipeline would rescale the supplied Gaussian."
        )

    z0 = _sample_latent(pipe, args.seed, args.image_size, args.image_size, args.dtype)
    result = pipe(
        prompt=args.prompt,
        latents=z0.to(args.device),
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=args.image_size,
        width=args.image_size,
    )
    images = result.images
    if len(images) != 1:
        raise SystemExit(f"pipeline returned {len(images)} images, expected 1")
    image = images[0]
    _release_pipeline(pipe)

    inverse = _load_inversion_pipeline(args.model_id, args.dtype, args.device)
    inverted = inverse.invert(
        prompt=args.prompt,
        image=image,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        inpaint_strength=args.inversion_strength,
        num_reg_steps=0,
    )
    z1 = _recovered_latent(inverted.latents, z0.shape)

    z0_cpu = z0.detach().to(device="cpu", dtype=torch.float32).contiguous()
    z1_cpu = z1.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if not torch.isfinite(z0_cpu).all() or not torch.isfinite(z1_cpu).all():
        raise SystemExit("a latent contains non-finite values; nothing was saved")

    image_path = output_dir / "generated_image.png"
    original_path = output_dir / "latent_original.pt"
    inverted_path = output_dir / "latent_inverted.pt"
    metadata_path = output_dir / "metadata.json"
    image.save(image_path)
    torch.save(z0_cpu, original_path)
    torch.save(z1_cpu, inverted_path)

    metadata = {
        "model_id": args.model_id,
        "seed": args.seed,
        "device": args.device,
        "dtype": _dtype_name(args.dtype),
        "saved_latent_dtype": "float32",
        "image_size": f"{args.image_size}x{args.image_size}",
        "run_id": output_dir.name,
        "git_commit": _git_commit(),
        "latent_original_path": original_path.name,
        "latent_inverted_path": inverted_path.name,
        "image_path": image_path.name,
        "prompt": args.prompt,
        "scheduler": "DDIMScheduler",
        "inverse_scheduler": "DDIMInverseScheduler",
        "inversion_api": "StableDiffusionDiffEditPipeline.invert",
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "inversion_strength": args.inversion_strength,
        "init_noise_sigma": init_noise_sigma,
        "latent_shape": list(z0_cpu.shape),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    _verify(output_dir, tuple(z0_cpu.shape), args.image_size)

    print(f"Model: {args.model_id}")
    print(f"Seed: {args.seed}")
    print(f"Original latent shape: {tuple(z0_cpu.shape)}")
    print(f"Inverted latent shape: {tuple(z1_cpu.shape)}")
    print(f"Image: {image_path}")
    print(f"Original latent: {original_path}")
    print(f"Inverted latent: {inverted_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
