# vpipe-ming-image

[Ming-Image-0.1-Design](https://huggingface.co/inclusionAI/Ming-Image-0.1-Design)
(inclusionAI, 6B, MIT-licensed weights) as a
[vpipe](https://github.com/tgo-app-dev/vpipe) plugin: text-to-image for
UI, infographics, posters and other text-rich designs, with RGBA output
(transparent backgrounds) -- on Apple Silicon, on vpipe's own Metal
backend.

STATUS: bring-up, working end to end. The full graph (conditioner ->
generate-image -> vae-decode -> save-image) produces correct pictures,
and every component is verified against the reference implementation;
see "What is verified". Measured on an M4 Pro (64 GB, checkpoint on an
external SSD): a 1024x1024 picture in ~2 m 40 s (~97 s of it the
11-forward denoise, ~45 s loading the 32 GB MLLM), a 2048x2048 one in
~10 m 30 s.

## Build

```sh
# vpipe, built and installed (the plugin must be built against the
# libvpipe it loads into -- re-install after every vpipe rebuild):
cmake --build ../vpipe-build && cmake --install ../vpipe-build --prefix ../vpipe-install

cmake -S . -B build -DCMAKE_PREFIX_PATH=$PWD/../vpipe-install
cmake --build build -j
ctest --test-dir build        # model-free tests; see below for the rest
```

Run a graph with it:

```sh
vpipe --plugin build/vpipe-ming-image.so \
      --launch docs/pipelines/ming-image-text-to-image.vpipeline
```

The host must decode a 4-channel QwenImage VAE with a scalar
`scaling_factor` -- vpipe's `MetalKrea2Vae` reads `input_channels` and
`scaling_factor` from the config (the fix landed with this plugin's
bring-up). An older host decodes this checkpoint's RGBA output at the
wrong stride.

## What it registers

* image family **`ming-image`** for `generate-image`: claims a
  checkpoint root holding `transformer/` (DiffusionTransformer with the
  Ming padding contract), `mllm/` (BailingMM2), `connector/`, `mlp/`,
  `vae/` and `scheduler/`. A layer-decomposition checkpoint
  (Ming-Image-0.1-Design-Layer) is claimed and refused by name.
* stage **`ming-image-conditioner`**: prompt in, the DiT's two-stream
  conditioning out.
* stage **`ming-image-model-config`**: the family's own knobs for
  generate-image's `model_config` port -- `shift` (override the
  checkpoint's static 6.0), and the reference pipeline's
  `cfg_truncation` / `cfg_normalization` (both off by default), and
  runtime LoRA: `lora` / `lora_scale` and seven more slots, `lora2` /
  `lora2_scale` .. `lora8` / `lora8_scale`, on a host whose DiT holds
  eight (the adapter is bound at load; strengths change live,
  and 0 reproduces the base exactly). Adapters are read in the
  diffusers, pipeline (`transformer.`) and kohya spellings over the 7
  block projections of all 34 blocks (238 modules). Set them HERE, not
  on generate-image: that stage's own `lora` keys are for its built-in
  families.
* a catalogue entry, so `model-fetch` can fetch the checkpoint (49 GB;
  the MLLM is 34 GB of it).
* a **quantize family** for `model-quantize`: w8 / w4 (group 64) packs
  of the DiT, quantizing the same tensors the in-tree Z-Image row does
  (every block projection and the block modulation; embedders, norms
  and the final layer stay dense). Point generate-image's `dit_dir` at a
  pack (or its `transformer/`) to run it.
* a **detect profile** (labels and modalities for `model-register`). The
  directory is RECOGNISED by the family's own structural `claims()` --
  the DiT's class name, `DiffusionTransformer`, is too generic to key
  on safely.

## Configuration

* `generate-image` `steps` counts DiT forwards: the reference's
  recommended **12 steps is 11 here** (its last step is a no-op at
  sigma 0). Guidance 1 (the default and the recommendation) runs no CFG;
  above 1 the negative is zeros (the reference's own) unless a graph
  wires one.
* A `scheduler-select` beat contributes only its `steps`. Its shift is
  NOT taken: that stage emits 1.15 when nothing was set, which would run
  this checkpoint on a curve it was not trained on. Set `shift` on
  `ming-image-model-config` instead. Other scheduler / sampler choices
  (karras, dynamic shift, heun...) are not implemented and are said so
  once per run.
* Recommended size is **2048x2048** (the reference's default); 1024x1024
  is ~5.6x faster but renders text less well and often ignores a
  transparent-background request (the reference does the same). Sizes
  are multiples of 16. MEASURED on an M4 Pro: 11 forwards take ~97 s at
  1024 and ~546 s at 2048.
* Transparent background: start the prompt with one of the reference's
  fixed phrases, e.g. `RGBA, 4-channel, transparent background` or
  `带透明通道，4通道RGBA图像`; the VAE always emits 4 channels.
* `ming-image-conditioner`: `unload_when_idle` (`auto` / `destroy` /
  `keep`) and `stream` (`auto` / `on` / `off`) for the ~32 GB MLLM.

## What is verified

Tests are plain executables (`build/*-test`). Those that need the
checkpoint or reference goldens read:

* `VPIPE_MING_TEST_MODEL_PATH` -- the checkpoint root;
* `VPIPE_MING_GOLDEN` -- goldens written by
  `tools/dump_ming_golden.py` (run with a venv pinned to the reference's
  versions: torch, transformers 4.57.1, diffusers 0.36.0);

* `VPIPE_MING_W8_PACK` / `VPIPE_MING_W4_PACK` -- quantized DiT packs
  written by `model-quantize` (for `ming-quant`);

and print SKIPPED when they are unset.

Every bar is the reference's OWN bf16 error against its fp32 run, from
the goldens (the reference's code, driven with fp32 compute).

* `ming-config` -- the 11-forward schedule against the reference's
  sigma table, the padded positions, patch packing, torch-identical
  noise, the claims, and the identity test against the checkpoint and
  lookalikes.
* `ming-dit` -- the DiT, vendored from vpipe's Z-Image forward: a
  small random-weight model (3 geometries, incl. unaligned lengths)
  rel-L2 0.0148-0.0151 against fp32 (the reference's bf16: 0.0158-
  0.0168); the real 6B at 512x512 0.0526 (reference bf16: 0.0622).
  Streamed and the no-slot fallback are bit-identical to resident.
* `ming-encoder` -- the tokenizer id-exact on 10 golden prompts (incl.
  the 1407-token JSON one) and 224 more strings; every MLLM tap, the
  two streams and the connector below the reference's bf16 error
  (e.g. stream A cap_feats 0.098 vs 0.134, stream B 0.025 vs 0.0375 on
  the short prompt); MoE routing agrees with fp32 on 95.6% of expert
  picks (the reference's bf16: 93.6%), every disagreement a near-tie.
  Streamed (4 GB held) is bit-identical to resident (34.6 GB).
* `ming-generator` -- the whole 11-forward denoise through the
  production generator against the reference's 12-step trajectory at
  512x512: final latent rel-L2 0.0130 (the reference's own bf16 run:
  0.0117).
* `ming-lora` / the LoRA cases of `ming-dit` -- a synthetic adapter
  (rank 8, alpha 16) in three spellings binds 28/28 modules of the
  small model and 238/238 of the real one; base + runtime adapter is
  0.0154 from the reference's FUSED fp32 model (its own bf16: 0.0167)
  while moving the output 0.34 (22x that floor); strength 0 is
  bit-identical to the base; streamed, promoted, Sol, quantized and
  ANE paths all carry it (on the ANE the adapter is fused into the
  module, checked against a negative control that drops it).
* `ming-quant` -- w8 / w4 packs: 270 block matrices quantized, the
  qmm route engaged on every one, streamed == resident; 512^2 forward
  0.0583 (w8) and 0.298 (w4) from the fp32 reference.
* The VAE is the host's: decoding this checkpoint's RGBA VAE, rel-L2
  5.7e-4 against the reference's fp32 decode (host test
  `qwen_image_21_vae.decode_matches_golden` pointed at a Ming golden).

Transparency is verified against the reference too, and it is a
property of the RESOLUTION as much as the prompt: with the same prompt,
seed and conditioning, the reference itself gives clean alpha cutouts at
2048x2048 and opaque pictures at 1024x1024 for longer prompts (a short
one, "RGBA, 4-channel, transparent background, a red apple", is
transparent at both). This port matches it at both sizes -- 71.0% of a
2048x2048 cat cutout's pixels transparent against the reference's 71%,
and the 1024 apple 55% against 60% -- picture for picture.

The MoE router runs in f32, as the reference's code is written (and as
vLLM-Omni's port runs its sigmoid and bias). Under CUDA autocast the
reference's own router rounds `sigmoid + expert_bias` to bf16, which
changes its expert picks; the goldens show that this does not decide
transparency, so this port is held to the f32 model.

## Acceleration tiers (generate-image accel keys)

MEASURED on an M4 Pro (fan-cooled, 64 GB, checkpoint on an external
SSD), one process, arms interleaved; the denoise is 11 forwards.

| arm | one forward 1024^2 | one forward 2048^2 | denoise 2048^2 | what it costs |
|---|---|---|---|---|
| dense | 8.81 s | 49.4 s | 548 s | -- |
| `ane_ffn` | 7.20 s (1.22x) | 40.7 s (1.21x) | 453 s (1.21x) | fp16 on the ANE: 0.112 from the reference at 512^2 vs the GPU's 0.053 (bar 0.124) |
| `sol_attn` tau 0 | 8.19 s (1.08x) | 39.9 s (1.24x) | 439 s (1.25x) | approximate: 0.26 from dense at 2048^2; every prompted string intact |
| `sol_attn` tau 0 + `ane_ffn` | 6.68 s (1.32x) | 31.0 s (1.59x) | 342 s (1.60x) | both of the above |

* **Sol** pays where the attention is: 16% of a 1024^2 forward, 31% at
  1536^2, 43% at 2048^2 (the largest bucket there; it already runs at
  the GEMMs' 6-7 TFLOP/s, so doing less of it is the lever). It runs on
  the M4's ALU arm as well as on matrix cores; with every key block kept
  it is bit-identical to dense. Higher tau keeps fewer blocks: tau 1 is
  1.50x at 2048^2 and BREAKS the layout (a button rendered six times) --
  judge it by a picture.
* **ANE feed-forward** is for the M4 series (no matrix cores): the
  balancer gives the ANE ~half the rows; lost ANE rows are recomputed on
  the GPU. q/k/v on the ANE is not offered (the installed `ane::Tier`
  cannot express three fused sources).
* **Quantized DiT** (w8: 6.55 GB, 0.0583 from the reference at 512^2;
  w4: 3.48 GB, 0.298) is a footprint option, not a speed one: on the M4
  a w8 forward runs at 0.87x of bf16. Streaming, not quantization, is
  how the 12 GB DiT fits a small box.

Not yet exercised: the M5 matrix-core paths (matmul2d, NAX attention,
i8_gemm, Sage, and Sol's NAX arm) are wired in the DiT and asserted to
engage on a matrix-core box, but have only run on an M4.

## Localization

The stages render their English `StageSpec` text in every locale: vpipe
has no way yet for a plugin to supply UI strings, and the plugin's
strings do not belong in vpipe's own catalogue. The zh-cn / zh-tw
translations are kept in `i18n/strings.json` (keyed as vpipe's web UI
keys them) for when it does.

## Licence

Apache-2.0 for this plugin (see LICENSE and NOTICE). The Ming-Image
weights are inclusionAI's, under the MIT License, and are not part of
this repository.
