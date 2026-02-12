#!/usr/bin/env bash

# Run evaluate video dataset with different gen dir paths.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"


RUNS=(
  "experiment.sampler=ding experiment.denoiser=ltx experiment.guidance_scale=3.5 experiment.steps=24 data.gen_dir=/path/to/ding_ltx_gen_dir"
  "experiment.sampler=ding experiment.denoiser=wan experiment.guidance_scale=5.0 experiment.steps=24 data.gen_dir=/path/to/ding_wan_gen_dir"

  "experiment.sampler=flair experiment.denoiser=ltx experiment.guidance_scale=3.5 experiment.steps=45 data.gen_dir=/path/to/flair_ltx_gen_dir"
  "experiment.sampler=flair experiment.denoiser=wan experiment.guidance_scale=5.0 experiment.steps=45 data.gen_dir=/path/to/flair_wan_gen_dir"

  "experiment.sampler=ddnm experiment.denoiser=ltx experiment.guidance_scale=3.5 experiment.steps=45 data.gen_dir=/path/to/ddnm_ltx_gen_dir"
  "experiment.sampler=ddnm experiment.denoiser=wan experiment.guidance_scale=5.0 experiment.steps=45 data.gen_dir=/path/to/ddnm_wan_gen_dir"

  "experiment.sampler=blended_diffusion experiment.denoiser=ltx experiment.guidance_scale=3.5 experiment.steps=46 data.gen_dir=/path/to/blended_diffusion_ltx_gen_dir"
  "experiment.sampler=blended_diffusion experiment.denoiser=wan experiment.guidance_scale=5.0 experiment.steps=46 data.gen_dir=/path/to/blended_diffusion_wan_gen_dir"

  "experiment.sampler=diffpir experiment.denoiser=ltx experiment.guidance_scale=3.5 experiment.steps=45 data.gen_dir=/path/to/diffpir_ltx_gen_dir"
  "experiment.sampler=diffpir experiment.denoiser=wan experiment.guidance_scale=5.0 experiment.steps=45 data.gen_dir=/path/to/diffpir_wan_gen_dir"

  "experiment.sampler=flow_chef experiment.denoiser=ltx experiment.guidance_scale=3.5 experiment.steps=46 data.gen_dir=/path/to/flow_chef_ltx_gen_dir"
  "experiment.sampler=flow_chef experiment.denoiser=wan experiment.guidance_scale=5.0 experiment.steps=46 data.gen_dir=/path/to/flow_chef_wan_gen_dir"

  "experiment.sampler=vace experiment.denoiser=wan experiment.guidance_scale=5.0 experiment.steps=30 data.gen_dir=/path/to/vace_wan_gen_dir"
)

for entry in "${RUNS[@]}"; do
  python -m ding.runner.evaluate_vid device=cuda:0 ${entry} 
done