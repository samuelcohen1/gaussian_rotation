def run_generate_and_invert() -> None:
    if DEVICE == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Use a one-GPU HiPerGator Jupyter session with the gaussian-rotation kernel."
        )
    from experiments.generate_and_invert import (
        _dtype,
        _git_commit,
        _load_generation_pipeline,
        _load_inversion_pipeline,
        _recovered_latent,
        _release_pipeline,
        _sample_latent,
    )

    seeds = sample_seeds()
    pending = [i for i in range(NUM_SAMPLES) if not sample_is_complete(i)]
    print(
        f"{NUM_SAMPLES - len(pending)}/{NUM_SAMPLES} samples already complete; "
        f"{len(pending)} still need generation+inversion.",
        flush=True,
    )
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    if not pending:
        print("Nothing to invert.", flush=True)
        return

    dtype = _dtype(DTYPE)
    load_started = time.perf_counter()
    print("Loading generation pipeline...", flush=True)
    pipe = _load_generation_pipeline(MODEL_ID, dtype, DEVICE)
    print(f"Generation ready in {fmt_duration(time.perf_counter() - load_started)}.", flush=True)
    if float(pipe.scheduler.init_noise_sigma) != 1.0:
        raise RuntimeError("DDIM init_noise_sigma is not 1.")
    if pipe.vae_scale_factor != VAE_SCALE_FACTOR:
        raise RuntimeError(f"VAE scale {pipe.vae_scale_factor} != {VAE_SCALE_FACTOR}")

    generated = []
    gen_clock = StepClock(len(pending), "generation")
    try:
        for sample_id in pending:
            gen_clock.begin()
            z = _sample_latent(pipe, seeds[sample_id], IMAGE_SIZE, IMAGE_SIZE, dtype)
            result = pipe(
                prompt=PROMPT,
                latents=z.to(DEVICE),
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                height=IMAGE_SIZE,
                width=IMAGE_SIZE,
            )
            if len(result.images) != 1:
                raise RuntimeError(f"sample {sample_id} returned {len(result.images)} images")
            image = result.images[0].convert("RGB")
            if image.size != (IMAGE_SIZE, IMAGE_SIZE):
                raise RuntimeError(f"bad image size {image.size}")
            generated.append((sample_id, z.detach().to(device="cpu", dtype=torch.float32), image))
            gen_clock.end(f"sample {sample_id}, seed {seeds[sample_id]}")
    finally:
        _release_pipeline(pipe)
        print("Generation pipeline released.", flush=True)

    load_started = time.perf_counter()
    print("Loading inversion pipeline...", flush=True)
    inverse = _load_inversion_pipeline(MODEL_ID, dtype, DEVICE)
    print(f"Inversion ready in {fmt_duration(time.perf_counter() - load_started)}.", flush=True)
    if type(inverse.inverse_scheduler).__name__ != "DDIMInverseScheduler":
        raise RuntimeError(f"unexpected inverse scheduler {type(inverse.inverse_scheduler).__name__}")

    inv_clock = StepClock(len(generated) * len(CONDITIONS), "inversion")
    try:
        for sample_id, z, image in generated:
            blob = {"z": z, "image_seed": torch.tensor(seeds[sample_id], dtype=torch.int64)}
            out_img = sample_image_dir(sample_id)
            image.save(out_img / "original.png")
            sample_started = time.perf_counter()
            for cond_index, cond in enumerate(CONDITIONS):
                inv_clock.begin()
                name = cond["name"]
                params = np.asarray(cond["params"], dtype=float)
                torch.manual_seed(SEED + 1000 * sample_id + cond_index)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(SEED + 1000 * sample_id + cond_index)
                warped = warp_pil(image, params) if name != "baseline" else image.copy()
                warped.save(out_img / f"{name}.png")
                inverted = inverse.invert(
                    prompt=PROMPT,
                    image=warped,
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                    inpaint_strength=1.0,
                    num_reg_steps=0,
                )
                z_hat = _recovered_latent(inverted.latents, z.shape)
                z_hat = z_hat.detach().to(device="cpu", dtype=torch.float32).contiguous()
                if tuple(z_hat.shape) != tuple(z.shape) or not torch.isfinite(z_hat).all():
                    raise RuntimeError(f"bad inverted latent sample {sample_id} {name}")
                blob[latent_key(name)] = z_hat
                inv_clock.end(f"sample {sample_id} {name}")
            torch.save(blob, sample_path(sample_id))
            print(
                f"Saved {sample_path(sample_id)} "
                f"({fmt_duration(time.perf_counter() - sample_started)} for {len(CONDITIONS)} inversions).",
                flush=True,
            )
    finally:
        _release_pipeline(inverse)
        print("Inversion pipeline released.", flush=True)

    meta = {
        "seed": SEED,
        "num_samples": NUM_SAMPLES,
        "sample_seeds": sample_seeds(),
        "samples_complete": [i for i in range(NUM_SAMPLES) if sample_is_complete(i)],
        "model_id": MODEL_ID,
        "image_size": IMAGE_SIZE,
        "vae_scale_factor": VAE_SCALE_FACTOR,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "prompt": PROMPT,
        "dtype": DTYPE,
        "conditions": CONDITIONS,
        "git_commit": _git_commit(),
        "note": "pixel affine then DDIM invert; sign-template registration in later cells",
    }
    (OUTPUT_DIR / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT_DIR / 'metadata.json'}", flush=True)


run_started = time.perf_counter()
run_generate_and_invert()
print(f"generate+invert wall {fmt_duration(time.perf_counter() - run_started)}", flush=True)
