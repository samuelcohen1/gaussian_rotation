example_dir = sample_image_dir(0)
missing = [c["name"] for c in CONDITIONS if not (example_dir / f"{c['name']}.png").is_file()]
if missing:
    raise FileNotFoundError(f"missing sample 0 images {missing}; run the generate/invert cell first")

fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(3.2 * len(CONDITIONS), 3.4))
for ax, cond in zip(axes, CONDITIONS):
    img = Image.open(example_dir / f"{cond['name']}.png")
    ax.imshow(img)
    ax.set_title(cond["name"], fontsize=9)
    ax.axis("off")
fig.suptitle("Sample 0: generation after each pixel affine (baseline = unwarped)")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "example_warped_images_sample0.png", bbox_inches="tight")
plt.show()
