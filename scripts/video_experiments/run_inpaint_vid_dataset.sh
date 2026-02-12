#!/usr/bin/env bash

# Run inpaint_batch with different denoiser presets. Adjust the RUNS array below to
# describe the configuration overrides you need for each run.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"


RUNS=(
  "flow_chef_wan denoiser.name=wan sampler.name=flow_chef conditioning.guidance_scale=5.0 steps=46"
  "diffpir_wan denoiser.name=wan sampler.name=diffpir conditioning.guidance_scale=5.0 steps=45"
  "ddnm_wan denoiser.name=wan sampler.name=ddnm conditioning.guidance_scale=5.0 steps=45"
  "blended_diffusion_wan denoiser.name=wan sampler.name=blended_diffusion conditioning.guidance_scale=5.0 steps=46"
  "ding_wan denoiser.name=wan sampler.name=ding conditioning.guidance_scale=5.0 steps=24"
  "flair_wan denoiser.name=wan sampler.name=flair conditioning.guidance_scale=5.0 steps=45"

  "flow_chef_ltx denoiser.name=ltx sampler.name=flow_chef conditioning.guidance_scale=3.5 steps=46"
  "diffpir_ltx denoiser.name=ltx sampler.name=diffpir conditioning.guidance_scale=3.5 steps=45"
  "ddnm_ltx denoiser.name=ltx sampler.name=ddnm conditioning.guidance_scale=3.5 steps=45"
  "blended_diffusion_ltx denoiser.name=ltx sampler.name=blended_diffusion conditioning.guidance_scale=3.5 steps=46"
  "ding_ltx denoiser.name=ltx sampler.name=ding conditioning.guidance_scale=3.5 steps=24"
  "flair_ltx denoiser.name=ltx sampler.name=flair conditioning.guidance_scale=3.5 steps=45"
)

for entry in "${RUNS[@]}"; do
  run_name=${entry%% *}
  overrides=${entry#* }
  echo "Running inpaint_vid_dataset for ${run_name}"
  python -m ding.runner.inpaint_vid_dataset device=cuda:0 ${overrides} 
done
