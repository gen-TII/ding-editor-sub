#!/usr/bin/env bash

# Run inpaint_batch with different denoiser presets. Adjust the RUNS array below to
# describe the configuration overrides you need for each run.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"


RUNS=(  
  "flow_chef_flux sampler.name=flow_chef denoiser.name=flux steps=51"
  "flow_chef_sd3 sampler.name=flow_chef denoiser.name=sd3_medium steps=51"
  "flow_chef_sd35 sampler.name=flow_chef denoiser.name=sd3.5_medium steps=51"
  
  "diffpir_flux sampler.name=diffpir denoiser.name=flux steps=50"
  "diffpir_sd3 sampler.name=diffpir denoiser.name=sd3_medium steps=50"
  "diffpir_sd35 sampler.name=diffpir denoiser.name=sd3.5_medium steps=50"
  
  "ddnm_flux sampler.name=ddnm denoiser.name=flux steps=50"
  "ddnm_sd3 sampler.name=ddnm denoiser.name=sd3_medium steps=50"
  "ddnm_sd35 sampler.name=ddnm denoiser.name=sd3.5_medium steps=50"
  
  "blended_diffusion_flux sampler.name=blended_diffusion denoiser.name=flux steps=51"
  "blended_diffusion_sd3 sampler.name=blended_diffusion denoiser.name=sd3_medium steps=51"
  "blended_diffusion_sd35 sampler.name=blended_diffusion denoiser.name=sd3.5_medium steps=51"
  
  "ding_flux sampler.name=ding denoiser.name=flux steps=26"
  "ding_sd3 sampler.name=ding denoiser.name=sd3_medium steps=26"
  "ding_sd35 sampler.name=ding denoiser.name=sd3.5_medium steps=26"
  
  "flair_flux sampler.name=flair denoiser.name=flux steps=50"
  "flair_sd3 sampler.name=flair denoiser.name=sd3_medium steps=50"
  "flair_sd35 sampler.name=flair denoiser.name=sd3.5_medium steps=50"
)

for entry in "${RUNS[@]}"; do
  run_name=${entry%% *}
  overrides=${entry#* }
  echo "Running inpaint_img_dataset for ${run_name}"
  python -m ding.runner.inpaint_img_dataset device=cuda:0 ${overrides} 
done
