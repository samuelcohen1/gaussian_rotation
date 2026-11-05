summary = (
    results.groupby("condition", observed=True)
    .agg(
        n=("sample_id", "count"),
        matched_bit_accuracy=("matched_bit_accuracy", "mean"),
        random_bit_accuracy=("random_bit_accuracy", "mean"),
        random_bit_accuracy_at_true=("random_bit_accuracy_at_true", "mean"),
        matched_bit_correlation=("matched_bit_correlation", "mean"),
        matched_continuous_correlation=("matched_continuous_correlation", "mean"),
        matched_param_mae=("matched_param_mae", "mean"),
        random_param_mae=("random_param_mae", "mean"),
        matched_angle_err=("matched_angle_err", "mean"),
        matched_tx_err=("matched_tx_err", "mean"),
        matched_ty_err=("matched_ty_err", "mean"),
        random_angle_err=("random_angle_err", "mean"),
        mean_true_angle=("true_latent_angle_deg", "mean"),
        mean_matched_angle=("matched_est_angle_deg", "mean"),
        mean_true_tx=("true_latent_tx", "mean"),
        mean_matched_tx=("matched_est_tx", "mean"),
        mean_true_ty=("true_latent_ty", "mean"),
        mean_matched_ty=("matched_est_ty", "mean"),
        corr_zhat_vs_z=("corr_zhat_vs_z", "mean"),
    )
    .reset_index()
)
order = {c["name"]: i for i, c in enumerate(CONDITIONS)}
summary["ord"] = summary.condition.map(order)
summary = summary.sort_values("ord").drop(columns=["ord"]).reset_index(drop=True)
summary.to_csv(OUTPUT_DIR / "summary_by_condition.csv", index=False)
print(summary.round(4).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
x = np.arange(len(summary))
axes[0].bar(x - 0.17, summary.matched_bit_accuracy, width=0.34, label="matched")
axes[0].bar(x + 0.17, summary.random_bit_accuracy, width=0.34, label="random")
axes[0].axhline(0.5, color="0.4", ls=":", label="chance")
axes[0].set_xticks(x)
axes[0].set_xticklabels(summary.condition, rotation=25, ha="right")
axes[0].set_ylim(0.45, 1.02)
axes[0].set_ylabel("bit accuracy")
axes[0].set_title("Bit accuracy after inverse-warp")
axes[0].legend(fontsize=8)

axes[1].bar(x - 0.17, summary.matched_param_mae, width=0.34, label="matched")
axes[1].bar(x + 0.17, summary.random_param_mae, width=0.34, label="random")
axes[1].set_xticks(x)
axes[1].set_xticklabels(summary.condition, rotation=25, ha="right")
axes[1].set_ylabel("mean |param| error")
axes[1].set_title("Latent parameter MAE vs ideal map")
axes[1].legend(fontsize=8)
fig.suptitle(f"DDIM affine sign-template ({NUM_SAMPLES} samples)")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "bits_and_param_error_by_condition.png", bbox_inches="tight")
plt.show()

nonbase = results.loc[results.condition != "baseline"]
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
axes[0].scatter(nonbase.true_latent_angle_deg, nonbase.matched_est_angle_deg, s=40, label="matched")
axes[0].scatter(nonbase.true_latent_angle_deg, nonbase.random_est_angle_deg, s=28, marker="x", label="random")
lim = (ANGLE_RANGE[0] - 1, ANGLE_RANGE[1] + 1)
axes[0].plot(lim, lim, ls="--", color="0.3")
axes[0].set_xlabel("expected latent angle")
axes[0].set_ylabel("estimate")
axes[0].set_title("Angle")
axes[0].legend(fontsize=8)

axes[1].scatter(nonbase.true_latent_tx, nonbase.matched_est_tx, s=40, label="matched")
axes[1].scatter(nonbase.true_latent_tx, nonbase.random_est_tx, s=28, marker="x", label="random")
axes[1].plot([-0.3, 1.6], [-0.3, 1.6], ls="--", color="0.3")
axes[1].set_xlabel("expected latent tx")
axes[1].set_title("tx")
axes[1].legend(fontsize=8)

axes[2].scatter(nonbase.true_latent_ty, nonbase.matched_est_ty, s=40, label="matched")
axes[2].scatter(nonbase.true_latent_ty, nonbase.random_est_ty, s=28, marker="x", label="random")
axes[2].plot([-1.0, 1.6], [-1.0, 1.6], ls="--", color="0.3")
axes[2].set_xlabel("expected latent ty")
axes[2].set_title("ty")
axes[2].legend(fontsize=8)
fig.suptitle("Estimated vs expected latent affine parameters")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "estimated_vs_expected_params.png", bbox_inches="tight")
plt.show()
