from experiments.generate_and_invert import _git_commit

meta_path = OUTPUT_DIR / "metadata.json"
meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
meta.update(
    {
        "registration": "sign-only bits of baseline z_hat vs full condition z_hat; random 0/1 control",
        "expected_latent_map": "same linear params; translation_px / vae_scale_factor",
        "n_samples_registered": NUM_SAMPLES,
        "trial_rows": int(len(results)),
        "git_commit": meta.get("git_commit") or _git_commit(),
    }
)
meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

print("Interpretation")
base = results.loc[results.condition == "baseline"]
print(
    f"- Baseline matched bit accuracy={base.matched_bit_accuracy.mean():.3f}; "
    f"random={base.random_bit_accuracy.mean():.3f}."
)
print(
    f"- Non-baseline matched bit accuracy={nonbase.matched_bit_accuracy.mean():.3f}; "
    f"random={nonbase.random_bit_accuracy.mean():.3f}; "
    f"random at true={nonbase.random_bit_accuracy_at_true.mean():.3f}."
)
print(
    f"- Non-baseline matched bit correlation={nonbase.matched_bit_correlation.mean():.3f}; "
    f"continuous corr(B, inverse-warp Y)={nonbase.matched_continuous_correlation.mean():.3f}."
)
for name in [c["name"] for c in CONDITIONS if c["name"] != "baseline"]:
    block = results.loc[results.condition == name]
    print(
        f"- {name}: matched ang/tx/ty err="
        f"{block.matched_angle_err.mean():.3f}/{block.matched_tx_err.mean():.3f}/{block.matched_ty_err.mean():.3f}; "
        f"bit_acc={block.matched_bit_accuracy.mean():.3f} (random {block.random_bit_accuracy.mean():.3f})."
    )
gap = nonbase.matched_bit_accuracy.mean() - nonbase.random_bit_accuracy.mean()
print(f"- Matched-minus-random bit-accuracy gap: {gap:.3f}.")
print(f"- Artifacts under {OUTPUT_DIR}")
print(f"- Total wall since generate start ~{fmt_duration(time.perf_counter() - run_started)}")
