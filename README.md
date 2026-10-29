# Gaussian rotation

The latent-space decoder comparison is `gaussian_shading_vs_rotation.ipynb`. It is separate from the diffusion job below.

## Notebooks

Dates are the day each notebook was added. Local notebooks only read saved files. HiPerGator notebooks need the `gaussian-rotation` kernel and one GPU.

| Notebook | Added | Where it runs | What it does |
| --- | --- | --- | --- |
| `gaussian_shading_vs_rotation.ipynb` | 2026-09-25 | Local | Compares Gaussian Shading, percentile-rotation, and threshold/skip decoding under synthetic latent noise. A later section repeats those decoders on one real inversion pair. |
| `geometric_registration_analysis.ipynb` | 2026-10-08 | Local | Reads the 100-image Slurm run and plots `pearsonr(\|z\|, \|z_hat\|)` against shift and rotation size. It does not generate or invert. |
| `geometric_registration_interactive.ipynb` | 2026-10-08 | HiPerGator | Runs that same 100-image, 13-condition experiment inside Jupyter, then plots it. Writes `results/geometric_registration/interactive`. |
| `geometric_latent_trajectory.ipynb` | 2026-10-15 | HiPerGator | One image, a 16 px shift, and a 15° rotation. Saves every inversion step and plots latent channels, differences, decodes, a shared PCA, and equivariance error. |
| `geometric_latent_undo.ipynb` | 2026-10-15 | HiPerGator | Regenerates the first 5 images from the 100-image run, saves `z` and every inverted `z_hat`, and compares correlation before and after undoing the shift or rotation in latent space. |
| `geometric_latent_undo_sweep.ipynb` | 2026-10-15 | Local or HiPerGator | Uses those five saved latent pairs. For each true perturbation, tries many hypothesized undo amounts and plots magnitude correlation and sign-bit accuracy, with a star at the true amount. |

## Diffusion generation and inversion on HiPerGator

The job generates one image from an explicit Gaussian latent and inverts that image with Diffusers' DDIM inversion. It does not decode or compare the latents. Edit `MODEL_ID`, `SEED`, and `OUTPUT_DIR` at the top of `scripts/run_diffusion_inversion.slurm` before submitting.

`runwayml/stable-diffusion-v1-5` is the default model. It is gated. On the model page, accept the license, then once on HiPerGator:

```bash
huggingface-cli login
```

### 1. Sync the latest code to HPG

```bash
git pull
```

### 2. Environment (first time, or if Diffusers is not installed yet)

Do this on a compute node by submitting a job. Do not install packages on the login node. The experiment uses `.conda-env`.

```bash
sbatch scripts/setup_env.slurm
```

### 3. Submit the experiment

```bash
sbatch scripts/run_diffusion_inversion.slurm
```

### 4. Check the job

```bash
squeue -u $USER
```

Logs are `results/slurm-diffusion-<JOBID>.out` and `results/slurm-diffusion-<JOBID>.err`.

### Geometric perturbation experiment

This job reuses the same model, `.conda-env`, and one-GPU Slurm settings. For each sample it generates one image, then inverts that image once with no transform and once after each shift or rotation. The saved score is the Pearson correlation of the unsigned latent magnitudes. Edit `NUM_SAMPLES`, `SEED`, and `OUTPUT_DIR` at the top of `scripts/run_geometric_registration.slurm`.

```bash
sbatch scripts/run_geometric_registration.slurm
```

Results land in `results/geometric_registration/run_001/correlations.csv`, with `metadata.json` and a few example images beside it. After copying that directory back, open `geometric_registration_analysis.ipynb`.

To run the same experiment inside an interactive HiPerGator Jupyter session, open `geometric_registration_interactive.ipynb` from this repo on a one-GPU notebook session. That notebook writes to `results/geometric_registration/interactive`, so it does not share files with the Slurm job.

`geometric_latent_trajectory.ipynb` is a separate one-image notebook. It inverts an image, a 16 px translation, and a 15° rotation, saves every inverse step, and plots the latent trajectory. Run it with the `gaussian-rotation` kernel. Outputs go to `results/latent_trajectory/example_001` and are gitignored.

`geometric_latent_undo.ipynb` repeats the registration conditions for the first 5 of those 100 images, saves both latents, and undoes the shift or rotation on the inverted latent. Run it with the `gaussian-rotation` kernel on a one-GPU notebook session. Outputs go to `results/geometric_registration/latent_undo_n5` and are gitignored.

### 5. After completion, inspect the output directory

For the default `OUTPUT_DIR`, the four files are:

```text
results/diffusion_inversion/run_001/latent_original.pt
results/diffusion_inversion/run_001/latent_inverted.pt
results/diffusion_inversion/run_001/generated_image.png
results/diffusion_inversion/run_001/metadata.json
```
