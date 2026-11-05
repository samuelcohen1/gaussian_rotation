def fmt_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, secs = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


class StepClock:
    def __init__(self, total: int, label: str):
        self.total = total
        self.label = label
        self.done = 0
        self.started = time.perf_counter()
        self.durations: list[float] = []
        self._tick = self.started

    def begin(self) -> None:
        self._tick = time.perf_counter()

    def end(self, detail: str) -> None:
        now = time.perf_counter()
        duration = now - self._tick
        self.durations.append(duration)
        self.done += 1
        average = sum(self.durations) / len(self.durations)
        remaining = self.total - self.done
        eta = average * remaining
        print(
            f"[{self.label} {self.done}/{self.total}] {detail} | "
            f"this {duration:.1f}s avg {average:.1f}s | "
            f"elapsed {fmt_duration(now - self.started)} | "
            f"~{fmt_duration(eta)} left",
            flush=True,
        )


def sample_seeds() -> list[int]:
    rng = np.random.default_rng(SEED)
    return [int(s) for s in rng.integers(0, 2**31 - 1, size=REFERENCE_SAMPLE_COUNT, endpoint=False)[:NUM_SAMPLES]]


def latent_key(name: str) -> str:
    return f"z_hat__{name}"


def sample_path(sample_id: int) -> Path:
    return LATENT_DIR / f"sample_{sample_id:03d}.pt"


def sample_image_dir(sample_id: int) -> Path:
    path = IMAGE_DIR / f"sample_{sample_id:03d}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def sample_is_complete(sample_id: int) -> bool:
    path = sample_path(sample_id)
    if not path.is_file():
        return False
    blob = torch.load(path, map_location="cpu", weights_only=True)
    needed = {"z", "image_seed"} | {latent_key(c["name"]) for c in CONDITIONS}
    return needed.issubset(blob.keys())


def params_to_matrix(params) -> tuple[np.ndarray, np.ndarray]:
    angle, sx, sy, shear, tx, ty = np.asarray(params, dtype=float)
    theta = np.deg2rad(angle)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rotation = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=float)
    linear = np.array([[sx, sx * shear], [0.0, sy]], dtype=float)
    return rotation @ linear, np.array([tx, ty], dtype=float)


def image_center(spatial_shape: tuple[int, int]) -> np.ndarray:
    return np.array(spatial_shape, dtype=float) / 2.0 - 0.5


def warp_array(image: np.ndarray, matrix: np.ndarray, translation: np.ndarray) -> np.ndarray:
    # Warp HWC or CHW float array with a shared 2-D affine (inverse map).
    if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3):
        channels = image
        spatial = image.shape[-2:]
        stack_axis = 0
    elif image.ndim == 3:
        channels = np.moveaxis(image, -1, 0)
        spatial = image.shape[:2]
        stack_axis = -1
    elif image.ndim == 2:
        channels = image[None]
        spatial = image.shape
        stack_axis = 0
    else:
        raise ValueError(f"unsupported image shape {image.shape}")

    center = image_center(spatial)
    inverse = np.linalg.inv(matrix)
    affine = np.zeros((2, 3), dtype=float)
    affine[:, :2] = inverse
    affine[:, 2] = -inverse @ (center + translation) + center
    warped = np.stack(
        [
            ndimage.affine_transform(
                channel,
                affine,
                order=WARP_ORDER,
                mode="constant",
                cval=0.0,
                prefilter=True,
            )
            for channel in channels
        ],
        axis=0,
    )
    if stack_axis == -1:
        return np.moveaxis(warped, 0, -1)
    if image.ndim == 2:
        return warped[0]
    return warped


def warp_pil(image: Image.Image, params) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    warped = np.clip(warp_array(rgb, *params_to_matrix(params)), 0.0, 1.0)
    return Image.fromarray((warped * 255.0).round().astype(np.uint8), mode="RGB")


def expected_latent_params(pixel_params) -> np.ndarray:
    angle, sx, sy, shear, tx, ty = np.asarray(pixel_params, dtype=float)
    return np.array(
        [angle, sx, sy, shear, tx / VAE_SCALE_FACTOR, ty / VAE_SCALE_FACTOR],
        dtype=float,
    )


def as_chw(tensor) -> np.ndarray:
    array = np.asarray(tensor, dtype=np.float64)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"expected (C,H,W), got {array.shape}")
    return array


def invert_params(params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix, translation = params_to_matrix(params)
    inverse = np.linalg.inv(matrix)
    return inverse, -inverse @ translation


def circular_valid_mask(spatial_shape: tuple[int, int], radius_frac: float = VALID_RADIUS_FRAC) -> np.ndarray:
    height, width = spatial_shape
    center_y, center_x = image_center(spatial_shape)
    rows, cols = np.mgrid[0:height, 0:width]
    radius = radius_frac * float(min(height, width))
    return (rows - center_y) ** 2 + (cols - center_x) ** 2 <= radius ** 2


def bits_from_latent(latent: np.ndarray) -> np.ndarray:
    return (np.asarray(latent) >= 0).astype(np.float64)


def bits_to_pm(bits01: np.ndarray) -> np.ndarray:
    return 2.0 * bits01 - 1.0


def sign_pm(latent: np.ndarray) -> np.ndarray:
    return np.where(np.asarray(latent) >= 0, 1.0, -1.0)


def masked_vectors(a: np.ndarray, b: np.ndarray, mask: np.ndarray):
    return a[:, mask].ravel(), b[:, mask].ravel()


def full_latent_corr(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        left, right = a.ravel(), b.ravel()
    else:
        left, right = masked_vectors(a, b, mask)
    if left.size < 2 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return 0.0
    return float(pearsonr(left, right).statistic)


def bit_metrics(bits01: np.ndarray, observation: np.ndarray, params: np.ndarray, mask: np.ndarray) -> dict:
    bits_pm = bits_to_pm(bits01)
    forward = warp_array(bits_pm, *params_to_matrix(params))
    back = warp_array(observation, *invert_params(params))
    signed_back = sign_pm(back)
    bits_m, signed_m = masked_vectors(bits_pm, signed_back, mask)
    _, back_m = masked_vectors(bits_pm, back, mask)
    return {
        "forward_correlation": full_latent_corr(forward, observation, mask),
        "bit_accuracy": float(np.mean(bits_m == signed_m)),
        "bit_mse": float(np.mean((bits_m - signed_m) ** 2)),
        "bit_correlation": full_latent_corr(bits_pm, signed_back, mask),
        "continuous_correlation": full_latent_corr(bits_pm, back, mask),
        "continuous_mse": float(np.mean((bits_m - back_m) ** 2)),
    }


print("conditions:")
for cond in CONDITIONS:
    exp = expected_latent_params(cond["params"])
    print(f"  {cond['name']}: pixel={tuple(cond['params'])} -> latent tx/ty=({exp[4]:.3f},{exp[5]:.3f})")
