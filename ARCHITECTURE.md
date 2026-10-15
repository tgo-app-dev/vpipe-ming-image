# vpipe-ming-image: architecture

## The model

inclusionAI's Ming-Image-0.1-Design is a 6B text-to-image DiT for UI,
posters and other text-rich designs, with RGBA output. Its parts, and
what this plugin does with each (exact math, with citations into the
reference, is in [docs/MING-IMAGE.md](docs/MING-IMAGE.md)):

| part | what it is | here |
|---|---|---|
| `mllm/` | BailingMM2: a BailingMoeV2 LLM (20 layers, 256 experts top-8, two routers) + a Qwen2.5 ViT | the LLM only, as a one-pass prefill encoder; the ViT never runs for text-to-image |
| `mlp/` | the conditioning heads + 256 learned query tokens | `MingEncoder` |
| `connector/` | Qwen2-1.5B, run **bidirectionally** over the 256 query rows | `MingEncoder` |
| `transformer/` | diffusers' Z-Image NextDiT at Z-Image 6B's dimensions | `MingDit` (vendored from vpipe's Z-Image DiT) |
| `vae/` | AutoencoderKLQwenImage at 4 channels (RGBA), latent / 8.0064 | the host's `vae-decode` |
| `scheduler/` | FlowMatchEuler, static shift 6.0 | `MingImageGenerator` |

## The seam

```
text-prompt ─► ming-image-conditioner ─► generate-image ─► vae-decode ─► save-image
                (this plugin's stage)     (host stage;      (host stage;
                                           this plugin's     reads RGBA
                                           image family)     QwenImage VAE)
```

* **`generate-image` + an image family** (`MingImageFamily`, tag
  `ming-image`). The host stage owns the ports and beats; the family owns
  the schedule, the guidance rule and the denoise loop, and returns the
  DiT's latent.
* **`ming-image-conditioner`**, this plugin's own stage, because the
  host's `diffusion-conditioner` has no seam for a third-party encoder and
  this one is not a text encoder in the host's sense.
* **No VAE family.** The host's QwenImage VAE decodes this checkpoint
  once it reads `input_channels` and a scalar `scaling_factor` (a host
  fix that ships with this plugin's bring-up; see "Host requirements").
* **The host's FLOP/s meter** (`common/perf-flops.h`). The DiT counts
  its GEMMs and attention as it encodes them with the toolkit's
  `GpuFlopsTally` and flushes at each block's start and before every
  commit, so the GPU is credited as each block retires; Sol's exact
  half is counted at the keep ratio it realizes (`SolFlops`). A cached
  caption encodes nothing for its path and counts nothing. The ANE
  feed-forward runs through the host's `ane::Tier`, which credits the
  ANE itself. Optional: on a host without the meter the plugin loads
  and counts nothing. `DitEngagement::gpu_flops` reports a forward's
  count; `ming-dit-test` holds it to the shapes exactly.

### The conditioning contract

One tensor, because generate-image hands a family one conditioning
tensor: bf16 `[256 + N, 3840]`.

* rows `[0, 256)`: stream A, the connector's caption features
  (`[256, 2560]`) in columns `[0, 2560)`, zero beyond. The DiT's own
  `cap_embedder` maps them to its width, so that weight stays with the
  DiT.
* rows `[256, 256 + N)`: stream B, the MLLM's N template rows, already
  projected to 3840 (they bypass `cap_embedder`).
* sideband `ming_query_rows = 256`.

The negative for CFG is **zeros of the same shape** unless a graph wires
one: that is the reference's own negative. CFG is off at the
recommended scale of 1.

### The DiT's three departures from Z-Image

1. **zero_masked padding.** No learned pad tokens; pad rows are zero
   and masked as keys. This port drops them and runs the real rows only,
   which is exact as long as the positions still count them: the image's
   axis-0 position is `roundup32(cap_len) + 1` (`ming-layout.h`).
2. **Two caption streams** (above).
3. **No caption cap** (the axis-0 rotary table is 20480 long; the
   reference's JSON prompts run ~1.4k tokens).

### The sampler

`steps` counts DiT forwards, as everywhere in vpipe. The reference's
"12 steps" is **11** forwards (its last runs at sigma 0 with dt 0), so
the example graph sets `steps: 11`. Static shift 6.0 at every size (the
reference's dynamic-shifting override is a silent no-op). The DiT is fed
`1 - sigma` and its output negated; noise is torch's CPU `randn(seed)`
bit for bit.

## Reading order for the tests

* `tests/ming-config-test.cc` -- checkpoint-free facts (the schedule
  table, the padded positions, the packing, the noise) and, with the
  checkpoint, the identity test against it and against lookalikes.
* `tests/ming-dit-test.cc` -- the DiT against reference goldens.
* `tests/ming-encoder-test.cc` -- the tokenizer, the MLLM taps, routing
  agreement and the connector against reference goldens.

Goldens come from `tools/dump_ming_golden.py`, run in a venv pinned to
the reference's own versions.
