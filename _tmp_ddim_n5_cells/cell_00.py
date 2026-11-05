# DDIM images to pixel affine to invert to sign-template latent affine

End-to-end on **real** SD1.5 DDIM generations (not the saved `latent_undo_n5` files):

1. Draw 5 Gaussian latents `z`, generate images.
2. Apply a small set of **pixel-space affine** warps (including identity).
3. DDIM-invert each warped image to `z_hat`.
4. Estimate the affine in latent space with the **sign-only baseline template** vs full signed observation, plus a **random 0/1 control**.
5. Score bit accuracy, bit/continuous correlation, and parameter error vs the ideal latent-scale map of the pixel affine (`t / 8`).
6. Save images, latents, tables, and figures under `results/ddim_affine_sign_registration_n5/`.

### HiPerGator (OOD Jupyter)

- Kernel: `gaussian-rotation`
- **One GPU**, ~8 GB, account/QoS `eel5840`, partition `hpg-turin`
- Working directory: **repo root**
- Budget: 5 samples x 5 conditions (1 gen + 5 inv each). Target **under 10 minutes** total.
- Re-running skips samples whose `.pt` files are already complete.
