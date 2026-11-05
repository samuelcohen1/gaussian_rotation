incomplete = [i for i in range(NUM_SAMPLES) if not sample_is_complete(i)]
if incomplete:
    raise FileNotFoundError(f"incomplete samples {incomplete}; finish generate/invert first")

blobs = [torch.load(sample_path(i), map_location="cpu", weights_only=True) for i in range(NUM_SAMPLES)]
SHAPE = as_chw(blobs[0]["z"]).shape
VALID_MASK = circular_valid_mask(SHAPE[-2:])
rng = np.random.default_rng(SEED + 7)
random_bits_by_sample = {i: (rng.random(SHAPE) < 0.5).astype(np.float64) for i in range(NUM_SAMPLES)}

rows = []
reg_started = time.perf_counter()
jobs = [(sid, cond) for sid in range(NUM_SAMPLES) for cond in CONDITIONS]
print(f"registering {len(jobs)} (sample, condition) pairs x matched+random...", flush=True)

for index, (sample_id, cond) in enumerate(jobs, start=1):
    blob = blobs[sample_id]
    z = as_chw(blob["z"])
    baseline = as_chw(blob[latent_key("baseline")])
    observation = as_chw(blob[latent_key(cond["name"])])
    expected = expected_latent_params(cond["params"])
    matched_bits = bits_from_latent(baseline)
    random_bits = random_bits_by_sample[sample_id]

    if cond["name"] == "baseline":
        matched = {
            "estimated_params": expected.copy(),
            "method": "identity",
            "runtime_s": 0.0,
            **bit_metrics(matched_bits, observation, expected, VALID_MASK),
        }
        random = {
            "estimated_params": expected.copy(),
            "method": "identity",
            "runtime_s": 0.0,
            **bit_metrics(random_bits, observation, expected, VALID_MASK),
        }
    else:
        matched = register_sign_affine(matched_bits, observation, VALID_MASK)
        random = register_sign_affine(random_bits, observation, VALID_MASK)

    random_at_true = bit_metrics(random_bits, observation, expected, VALID_MASK)
    row = {
        "sample_id": sample_id,
        "condition": cond["name"],
        "matched_method": matched["method"],
        "random_method": random["method"],
        "matched_runtime_s": matched["runtime_s"],
        "random_runtime_s": random["runtime_s"],
        "matched_bit_accuracy": matched["bit_accuracy"],
        "matched_bit_correlation": matched["bit_correlation"],
        "matched_continuous_correlation": matched["continuous_correlation"],
        "matched_forward_correlation": matched["forward_correlation"],
        "random_bit_accuracy": random["bit_accuracy"],
        "random_bit_correlation": random["bit_correlation"],
        "random_forward_correlation": random["forward_correlation"],
        "random_bit_accuracy_at_true": random_at_true["bit_accuracy"],
        "matched_param_mae": float(np.mean(np.abs(matched["estimated_params"] - expected))),
        "random_param_mae": float(np.mean(np.abs(random["estimated_params"] - expected))),
        "matched_angle_err": float(abs(matched["estimated_params"][0] - expected[0])),
        "matched_tx_err": float(abs(matched["estimated_params"][4] - expected[4])),
        "matched_ty_err": float(abs(matched["estimated_params"][5] - expected[5])),
        "random_angle_err": float(abs(random["estimated_params"][0] - expected[0])),
        "random_tx_err": float(abs(random["estimated_params"][4] - expected[4])),
        "random_ty_err": float(abs(random["estimated_params"][5] - expected[5])),
        "corr_zhat_vs_z": full_latent_corr(observation, z, VALID_MASK),
    }
    for name, truth, pix, matched_hat, random_hat in zip(
        PARAM_NAMES,
        expected,
        np.asarray(cond["params"], dtype=float),
        matched["estimated_params"],
        random["estimated_params"],
    ):
        row[f"pixel_{name}"] = float(pix)
        row[f"true_latent_{name}"] = float(truth)
        row[f"matched_est_{name}"] = float(matched_hat)
        row[f"random_est_{name}"] = float(random_hat)
    rows.append(row)

    if index == 1 or index == len(jobs) or index % max(1, len(jobs) // 5) == 0:
        elapsed = time.perf_counter() - reg_started
        eta = elapsed / index * (len(jobs) - index)
        print(
            f"[{index}/{len(jobs)}] sample {sample_id} {cond['name']} | "
            f"matched bit={row['matched_bit_accuracy']:.3f} random={row['random_bit_accuracy']:.3f} | "
            f"ang_err={row['matched_angle_err']:.2f} | "
            f"elapsed {fmt_duration(elapsed)} ~{fmt_duration(eta)} left",
            flush=True,
        )

results = pd.DataFrame(rows)
results.to_csv(OUTPUT_DIR / "trial_results.csv", index=False)
print(f"wrote {OUTPUT_DIR / 'trial_results.csv'} ({len(results)} rows)")
print(f"registration wall {fmt_duration(time.perf_counter() - reg_started)}")
