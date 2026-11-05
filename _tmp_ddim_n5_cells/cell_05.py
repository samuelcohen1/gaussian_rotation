def forward_correlation(bits_pm: np.ndarray, observation: np.ndarray, params: np.ndarray, mask: np.ndarray) -> float:
    warped = warp_array(bits_pm, *params_to_matrix(params))
    return full_latent_corr(warped, observation, mask)


def coarse_initialize(bits_pm: np.ndarray, observation: np.ndarray, mask: np.ndarray) -> np.ndarray:
    identity = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=float)
    best_params = identity.copy()
    best_corr = forward_correlation(bits_pm, observation, best_params, mask)

    for angle in np.arange(ANGLE_RANGE[0], ANGLE_RANGE[1] + 0.5 * COARSE_ANGLE_STEP, COARSE_ANGLE_STEP):
        params = np.array([angle, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=float)
        corr = forward_correlation(bits_pm, observation, params, mask)
        if corr > best_corr:
            best_corr, best_params = corr, params

    for tx in np.arange(TRANSLATION_RANGE[0], TRANSLATION_RANGE[1] + 0.5 * COARSE_TRANSLATION_STEP, COARSE_TRANSLATION_STEP):
        for ty in np.arange(TRANSLATION_RANGE[0], TRANSLATION_RANGE[1] + 0.5 * COARSE_TRANSLATION_STEP, COARSE_TRANSLATION_STEP):
            params = np.array([0.0, 1.0, 1.0, 0.0, tx, ty], dtype=float)
            corr = forward_correlation(bits_pm, observation, params, mask)
            if corr > best_corr:
                best_corr, best_params = corr, params

    angle0, _, _, _, tx0, ty0 = best_params
    for angle in np.unique(np.clip([angle0 - COARSE_ANGLE_STEP, angle0, angle0 + COARSE_ANGLE_STEP], *ANGLE_RANGE)):
        for tx in np.unique(np.clip([tx0 - COARSE_TRANSLATION_STEP, tx0, tx0 + COARSE_TRANSLATION_STEP], *TRANSLATION_RANGE)):
            for ty in np.unique(np.clip([ty0 - COARSE_TRANSLATION_STEP, ty0, ty0 + COARSE_TRANSLATION_STEP], *TRANSLATION_RANGE)):
                params = np.array([angle, 1.0, 1.0, 0.0, tx, ty], dtype=float)
                corr = forward_correlation(bits_pm, observation, params, mask)
                if corr > best_corr:
                    best_corr, best_params = corr, params
    return best_params


def register_sign_affine(bits01: np.ndarray, observation: np.ndarray, mask: np.ndarray) -> dict:
    started = time.perf_counter()
    bits_pm = bits_to_pm(bits01)
    init = coarse_initialize(bits_pm, observation, mask)

    def objective(params: np.ndarray) -> float:
        return -forward_correlation(bits_pm, observation, params, mask)

    refined = minimize(
        objective,
        init,
        method="L-BFGS-B",
        bounds=SCIPY_BOUNDS,
        options={"ftol": 1e-9, "gtol": 1e-7, "maxiter": 200},
    )
    estimate = np.asarray(refined.x, dtype=float)
    metrics = bit_metrics(bits01, observation, estimate, mask)
    return {
        "estimated_params": estimate,
        "method": "coarse+L-BFGS-B",
        "runtime_s": time.perf_counter() - started,
        **metrics,
    }


print("registration helpers ready")
