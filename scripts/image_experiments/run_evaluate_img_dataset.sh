#!/usr/bin/env bash

# Run evaluate image dataset with different gen dir paths.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"


RUNS=(
  "experiment.denoiser=sd3_medium    experiment.sampler=ding experiment.steps=27  data.gen_dir=/path/to/ding_sd3_gen_dir"
  "experiment.denoiser=sd3.5_medium  experiment.sampler=ding experiment.steps=27  data.gen_dir=/path/to/ding_sd35_gen_dir"
  "experiment.denoiser=flux          experiment.sampler=ding experiment.steps=27  data.gen_dir=/path/to/ding_flux_gen_dir"

  "experiment.denoiser=sd3_medium    experiment.sampler=flair experiment.steps=50  data.gen_dir=/path/to/flair_sd3_gen_dir"
  "experiment.denoiser=sd3.5_medium  experiment.sampler=flair experiment.steps=50  data.gen_dir=/path/to/flair_sd35_gen_dir"
  "experiment.denoiser=flux          experiment.sampler=flair experiment.steps=50  data.gen_dir=/path/to/flair_flux_gen_dir"

  "experiment.denoiser=sd3_medium    experiment.sampler=ddnm experiment.steps=50  data.gen_dir=/path/to/ddnm_sd3_gen_dir"
  "experiment.denoiser=sd3.5_medium  experiment.sampler=ddnm experiment.steps=50  data.gen_dir=/path/to/ddnm_sd35_gen_dir"
  "experiment.denoiser=flux          experiment.sampler=ddnm experiment.steps=50  data.gen_dir=/path/to/ddnm_flux_gen_dir"

  "experiment.denoiser=sd3_medium    experiment.sampler=diffpir experiment.steps=50  data.gen_dir=/path/to/diffpir_sd3_gen_dir"
  "experiment.denoiser=sd3.5_medium  experiment.sampler=diffpir experiment.steps=50  data.gen_dir=/path/to/diffpir_sd35_gen_dir"
  "experiment.denoiser=flux          experiment.sampler=diffpir experiment.steps=50  data.gen_dir=/path/to/diffpir_flux_gen_dir"

  "experiment.denoiser=sd3_medium    experiment.sampler=flow_chef experiment.steps=51  data.gen_dir=/path/to/flow_chef_sd3_gen_dir"
  "experiment.denoiser=sd3.5_medium  experiment.sampler=flow_chef experiment.steps=51  data.gen_dir=/path/to/flow_chef_sd35_gen_dir"
  "experiment.denoiser=flux          experiment.sampler=flow_chef experiment.steps=51  data.gen_dir=/path/to/flow_chef_flux_gen_dir"

  "experiment.denoiser=sd3_medium    experiment.sampler=blended_diffusion experiment.steps=51  data.gen_dir=/path/to/blended_diffusion_sd3_gen_dir"
  "experiment.denoiser=sd3.5_medium  experiment.sampler=blended_diffusion experiment.steps=51  data.gen_dir=/path/to/blended_diffusion_sd35_gen_dir"
  "experiment.denoiser=flux          experiment.sampler=blended_diffusion experiment.steps=51  data.gen_dir=/path/to/blended_diffusion_flux_gen_dir"
)

for entry in "${RUNS[@]}"; do
  python -m ding.runner.evaluate_img device=cuda:0 ${entry} 
done















