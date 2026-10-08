# Gaussian rotation

The latent-space decoder comparison is `gaussian_shading_vs_rotation.ipynb`. It is separate from the diffusion job below.

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

### 5. After completion, inspect the output directory

For the default `OUTPUT_DIR`, the four files are:

```text
results/diffusion_inversion/run_001/latent_original.pt
results/diffusion_inversion/run_001/latent_inverted.pt
results/diffusion_inversion/run_001/generated_image.png
results/diffusion_inversion/run_001/metadata.json
```
