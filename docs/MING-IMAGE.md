# Ming-Image-0.1-Design: reference architecture spec (text-to-image)

Scope: the **text-to-image** path of `inclusionAI/Ming-Image-0.1-Design` as
implemented by the companion repo, traced end to end for a vpipe port.
Layer decomposition and image edit are mentioned only where they share
code and a flag decides between them.

Sources and conventions

- `[R] file:line` -- the reference repo, `~/dock/dump/vpipe-test/ming-image/Ming-Image`
  at commit `3515cd7b` ("Inline natural-language demo prompt").
- `[V] file:line` -- vpipe, at `8ab6c6cb`.
- `[D] file:line` -- diffusers **0.36.0** / `[T]` transformers **4.57.1**,
  the exact pins in `[R] requirements.txt:4-6` (wheels unpacked to a
  scratch dir and read, not the newer venv copies).
- Checkpoint configs were read from the HF repo (`mlp/config.json`,
  `transformer/config.json`, `vae/config.json`,
  `scheduler/scheduler_config.json`, `mllm/config.json`,
  `mllm/tokenizer*.json`, `connector/config.json`, the safetensors index
  files, and the safetensors **headers** of `mlp/`, `vae/` and one `mllm/`
  shard via HTTP range requests). No weights were downloaded.
- **VERIFIED** = executed against the pinned library code.
  **INFERRED** = reasoned, not read or run.

---

## 0. One-screen summary

```
prompt P
 └─ processor template (no truncation)                         [R] processing_bailingmm2.py:380-415
    "<role>SYSTEM</role>你是一个友好的AI助手。\n\ndetailed thinking off<|role_end|>"
    "<role>HUMAN</role>{P}<|role_end|><role>ASSISTANT</role>"           -> N text ids
 └─ append [157158 <image>] + 256 x [157157 <imagePatch>] + [157159 </image>]   L = N + 258
    the 256 patch rows' embeddings are REPLACED by 256 learned query tokens
 └─ BailingMoeV2 LLM, 20 layers, CAUSAL, 3D "video_rope", MoE (text gate / image gate)
    ├─ stream B ("directvlm"): hs[5] ‖ hs[12] ‖ hs[20] at the N text rows -> [N, 6144]
    │     -> RMSNorm(6144,1e-5) -> Linear(6144->3840)          = cap_feats_2  [N, 3840]
    └─ stream A ("learnable tokens"): hs[20] at the 256 query rows -> [256, 2048]
          -> Linear(2048->1536) -> Qwen2-1.5B connector, 28 layers, BIDIRECTIONAL,
             positions 0..255, final norm -> Linear(1536->2560) = cap_feats    [256, 2560]
 └─ DiT = diffusers ZImageTransformer2DModel (Z-Image 6B dims) + 3 deltas:
       caption = [cap_embedder(cap_feats) (256) ; cap_feats_2 (N) ; zero pad to x32]
       zero_masked padding (pads = 0, excluded as attention KEYS), axes_lens[0] 20480
 └─ FlowMatchEuler, STATIC shift 6.0, 12 "steps" = 11 effective Euler updates
       DiT gets t = 1 - sigma (x1000 inside), output NEGATED, CFG off at cfg 1.0
       noise = torch CPU randn(seed), f32, [1,16,H/8,W/8]
 └─ latent / 8.0064 -> AutoencoderKLQwenImage (z16, /8, RGBA in/out), 1 frame
       clamp[-1,1] -> (x+1)/2 -> round(x*255) -> uint8 RGBA (straight alpha)
       -> PIL LANCZOS resize if the requested size != the bucket
```

Component sizes (from the safetensors indexes/headers):

| component | class | on disk | needed for T2I |
|---|---|---|---|
| `mllm/` | BailingMM2Native (BailingMoeV2 LLM + Qwen2.5-ViT) | 17.00 B params, 34.0 GB bf16, 7 shards | LLM body only: ~15.95 B params, ~31.9 GB bf16 (no ViT, no `linear_proj`, no `lm_head`) |
| `connector/` | Qwen2ForCausalLM 1.5B | 1.54 B params **F32**, 6.17 GB | ~1.31 B (no `embed_tokens`/tied head) -> ~2.6 GB as bf16 |
| `mlp/model.safetensors` | 8 loose tensors, **F32** | ~125 MB | all |
| `transformer/` | DiffusionTransformer | 12.31 GB bf16 (6.15 B params) | all |
| `vae/` | AutoencoderKLQwenImage | bf16, 194 tensors | decoder (+`post_quant_conv`) |

Per-MoE-layer weight: 256 experts x 3 x (2048x512) + shared (3 x 2048x512)
+ 3 gates (256x2048) ~ 810 M params ~ **1.62 GB bf16**, x19 layers.

---

## 1. Conditioning

### 1.1 Driver path (what actually runs for `--task text-to-image`)

1. `infer.py` builds `[{"role":"HUMAN","content":[{"type":"text","text":P}]}]`
   `[R] infer.py:186-191`, renders it with `processor.apply_chat_template`
   `[R] infer.py:420` and tokenizes it with `processor(...)` `[R] infer.py:428-437`.
2. `model.generate(..., image_gen=True, image_gen_task="text-to-image",
   image_gen_seed=seed, image_gen_steps=12, image_gen_cfg=1.0)`
   `[R] infer.py:444-452`. Defaults: steps 12 / cfg 1.0
   `[R] inference_profile.py:55-58`, seed 42 `[R] infer.py:67`, dtype bf16
   `[R] infer.py:73`, attention `eager` `[R] infer.py:74-80`.
3. `generate` -> `get_condition_embeds_for_image_gen` `[R] modeling_bailingmm2.py:160-184`
   -> `diffusion_loss.sample(...)` `[R] modeling_bailingmm2.py:259-274`
   -> `ImageGenerator.sample` `[R] diffusion/generator.py:254-296`
   -> `ImageGenerationPipeline.__call__` `[R] diffusion/pipeline.py:309-716`.

### 1.2 Prompt wrapping and token ids

The processor's own `apply_chat_template` is used, **not** the jinja
`chat_template` stored in `mllm/tokenizer_config.json`:

```
text = "<role>SYSTEM</role>你是一个友好的AI助手。\n\ndetailed thinking off"   # SYSTEM_PROMPT_LINGV2_FLASH_NOTHINK
     + "<|role_end|>"                                                     # tokenizer.eos_token
     + "<role>HUMAN</role>" + P + "<|role_end|>"
     + "<role>ASSISTANT</role>"
```

`[R] processing_bailingmm2.py:57-61` (constants), `:351-359` (system),
`:380-415` (assembly; `use_cot_system_prompt=False` -> "thinking off").

Tokenization: `BailingTokenizer(PreTrainedTokenizerFast)`
`[R] tokenization_bailing.py:46-114`, `add_bos_token=False`,
`add_eos_token=False`; called with only `padding=True`
`[R] processing_bailingmm2.py:188, 346-347` -> **no truncation, no BOS/EOS
added**. `tokenizer.json`: BPE, `normalizer` NFC, pre-tokenizer
`Split(regex) + ByteLevel(add_prefix_space=false)`, regex
`'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+`
(possessive quantifiers), post-processor ByteLevel (adds nothing).

Special ids (from `mllm/tokenizer_config.json` / `mllm/config.json`):
`<|role_end|>` 156895 (eos, also pad), `<role>` 157151, `</role>` 157152,
`<imagePatch>` 157157 (`image_patch_token`), `<image>` 157158
(`image_start_token`), `</image>` 157159 (`image_end_token`).

**Sample (VERIFIED with `tokenizers` on the checkpoint's tokenizer.json)**,
P = `A small red cube on a white table`, N = 34:

```
[157151, 90827, 157152, 651, 4929, 82317, 10618, 37862, 311, 198, 198,
 14136, 5381, 6350, 928, 156895, 157151, 39, 116171, 157152, 32, 2258,
 3862, 33821, 366, 259, 4999, 3742, 156895, 157151, 8469, 7342, 5468, 157152]
 <role> SYSTEM </role> 你 是一个 友好的 AI 助手 。 \n \n det ailed ␠thinking ␠off <|role_end|>
 <role> H UMAN </role> A ␠small ␠red ␠cube ␠on ␠a ␠white ␠table <|role_end|> <role> ASS IST ANT </role>
```

The README's JSON "cabin" prompt is 1407 template tokens.
`get_text_from_prompt` (quoted-text extraction, `[R] processing_bailingmm2.py:83-115, 298`)
fills `image_gen_text`, which `generate` accepts and never reads
`[R] modeling_bailingmm2.py:103` (a vestige of a byt5 glyph path that is
refused outright, `:325-329`).

### 1.3 The appended generation tokens ("learnable token condition")

`append_input_ids_with_multiscale_learnable_tokens` `[R] modeling_bailingmm2.py:512-570`,
called with `start = image_patch_token+1 = 157158`, `end = +2 = 157159`,
`patch = 157157` `[R] :661-668`. For each scale in `img_gen_scales = [16]`
(`mlp/config.json`): `[157158] + [157157]*256 + [157159]` is inserted at
the first padding position (= the end, batch 1). Masks:
`attention_mask` 1 for all 258, `gen_mask` 1 only on the 256 patch rows,
`labels` = -100 on the text, 1 on all 258 appended rows `[R] :552-560`.

`appand_learnable_tokens` `[R] :572-649`: the 256 patch rows are a
"generated image" (the first patch run's `gen_mask` is 1), so
`image_grid_thw = [[1, 2, 16*16*2]] = [[1, 2, 512]]` `[R] :604-609` and
`image_embeds = query_tokens_dict["16x16"]` `[256, 2048]`.
`query_tokens_dict` is loaded from `mlp/model.safetensors` key
`query_tokens_dict.16x16` (F32, [256, 2048]) into a bf16 parameter
`[R] :362-384`. The `normalize(randn)` init is overwritten by the load;
the learned values are used raw.

`prompt_wrap_navit` `[R] modeling_bailing_moe_v2.py:1540-1563`: word
embeddings for all L ids, then `masked_scatter` the 256 query rows into
the `<imagePatch>` positions (in order) `[R] :1510-1529`; returns
`vision_mask = (ids == 157157)` [B, L, 1] which becomes the MoE
`image_mask` `[R] modeling_bailingmm2.py:685-689, 708`.

So "img_gen_scales [16]" = **one** scale of 16x16 = **256 learned query
embeddings**, appended after the prompt, framed by `<image>`/`</image>`.
Only the last scale is consumed (`[...][-1]`, `[R] :746-748`).

### 1.4 Vision tower: does NOT run

`pixel_values` is None for T2I, so `extract_image_feature` is skipped
`[R] modeling_bailingmm2.py:161-165`; the "image" in the sequence is the
query tokens. The Qwen2.5-ViT (32 blocks, hidden 1280) and `linear_proj`
(`mlp_depth` 2, `[R] :57-61`) are loaded but unused. A port can skip
`vision.*` and `linear_proj.*` (and `model.lm_head.weight`).

### 1.5 BailingMoeV2 LLM (config `mllm/config.json` -> `llm_config`)

hidden 2048, 20 layers, 16 q heads / 4 kv heads (GQA 4), head_dim 128,
vocab 157184, `rms_norm_eps` 1e-6, `rope_theta` 600000,
`partial_rotary_factor` 0.5, `rope_scaling {"type":"video_rope"}`,
`first_k_dense_replace` 1, dense `intermediate_size` 5120, 256 experts,
top-8, `moe_intermediate_size` 512, 1 shared expert, `n_group` 8,
`topk_group` 4, `routed_scaling_factor` 2.5, `router_type` "MultiRouter",
`use_qkv_bias` false, `use_bias` false, `tie_word_embeddings` false.
All checkpoint tensors are **BF16** (incl. gates and `expert_bias`).

Weight names (per layer `i`, prefix `model.model.`):
`word_embeddings.weight` [157184,2048]; `layers.i.input_layernorm.weight`;
`layers.i.attention.query_key_value.weight` [3072,2048];
`layers.i.attention.{q_norm,k_norm}.weight` [128];
`layers.i.attention.dense.weight` [2048,2048];
`layers.i.post_attention_layernorm.weight`;
layer 0: `layers.0.mlp.{gate,up,down}_proj.weight` ([5120,2048]x2, [2048,5120]);
layers 1-19: `mlp.{gate,image_gate,audio_gate}.{weight [256,2048], expert_bias [256]}`,
`mlp.experts.{0..255}.{gate,up}_proj.weight` [512,2048],
`mlp.experts.{e}.down_proj.weight` [2048,512] (**one tensor per expert**,
768 per layer -- no stacked/switch layout),
`mlp.shared_experts.{gate,up,down}_proj.weight` (512-wide);
`norm.weight` [2048].

**Layer** `[R] modeling_bailing_moe_v2.py:1261-1353`: pre-norm
`h = x + Attn(RMSNorm(x)); y = h + FFN(RMSNorm(h))`; FFN = dense SwiGLU at
layer 0, MoE at 1..19 `[R] :1268-1272`.

**RMSNorm** `[R] :119-133`: `w * (x_f32 * rsqrt(mean(x_f32^2) + eps)).to(bf16)`;
plain weight (not zero-centred), eps 1e-6.

**Attention** `[R] :874-1018` (eager, the CLI default; flash-attn-2 is
the same math):
- `qkv = x @ Wqkv^T`, viewed as `[L, 24, 128]` and split by **head**
  `[16 q | 4 k | 4 v]` -> row blocks q 0..2047, k 2048..2559, v 2560..3071
  `[R] :935-940`. No bias.
- **q/k RMSNorm per head always applied** (`q_norm`, `k_norm`, eps 1e-6)
  `[R] :951-952`. (`use_qk_norm` in the config class is never stored or
  read, `[R] configuration_bailing_moe_v2.py:16`.)
- RoPE (below), GQA repeat, `softmax(q k^T / sqrt(128) + causal_mask)` in
  f32 then cast `[R] :980-996`, `dense` output proj, no bias.
- Mask: `_prepare_4d_causal_attention_mask` (ordinary **causal**, no
  sliding window) `[R] :1706-1710`. Batch 1, no padding.

**RoPE: "video_rope", partial 64/128, NeoX rotate-half**
- `inv_freq[k] = 600000^(-2k/64)`, k = 0..31 (transformers default init
  with `partial_rotary_factor`, `[T] modeling_rope_utils.py:123-131`);
  `BailingMoeV2RotaryEmbedding3D` `[R] :182-219` builds
  `cos/sin[axis, B, L, 64] = cos/sin(pos[axis] * inv_freq[k mod 32])`
  (emb = cat(freqs, freqs)), computed f32 then **cast to the activation
  dtype (bf16)** `[R] :219`.
- `apply_3d_rotary_pos_emb(..., mrope_section=[8,12,12], rope_type="video_rope")`
  `[R] :279-300` (section set at `[R] :910-912`): sections become
  `[24, 8, 24, 8]`; resulting axis per rotary pair (k, k+32):
  **k < 24: axis H if k even, W if k odd; k >= 24: axis T.**
- Rotation on `q[..., :64]` / `k[..., :64]` with
  `x*cos + rotate_half(x)*sin` (pairs k and k+32); dims 64..127 pass
  through `[R] :305-316`.

**Position ids** -- `get_t_scale_rope_index(..., scale_factor=2.0)` is the
one used for `video_rope` `[R] :1658-1667`, defined `[R] :322-465`
(`get_rope_index` at `:467` is the "3D" variant and is NOT used).
`spatial_merge_size` is forced to 2 `[R] :1498`. For the T2I sequence
`[N text][<image>][256 patch][</image>]` (VERIFIED by running it):

| rows | (t, h, w) |
|---|---|
| text i = 0..N-1 and `<image>` (i = N) | (i, i, i) |
| patch j = 0..255 | (N+1, N+1, N+1 + j - 127) |
| `</image>` | (N+2, N+2, N+2) |

The grid is t=1, h=2//2=1, w=512//2=256; w is centred
(`- (grid_w-1)//2` = -127, `[R] :416-419`), then t is added to h and w
(`[R] :426-429`). **w goes negative when N < 126** (sample N=34:
w = -92..163). The text after the image resumes from `max(t)+1`, not from
`max(w)+1` (`[R] :436`). For text rows all three axes are equal, so text
gets plain 1D partial RoPE.

**MoE block** `[R] :740-857`:

```
# per token x (bf16), for each of gate / image_gate (audio_gate unused):
logits = x_f32 @ W_gate_f32^T                       # [256], f32        :725
s      = sigmoid(logits)                                                 :727
r      = s + expert_bias                            # routing score      :729
gs[g]  = sum(top2(r[32g : 32g+32])), g = 0..7       # group score        :705
G      = top4(gs)                                   # 4 of 8 groups      :706
idx    = top8(r masked to groups in G, others -inf)                      :717-718
w      = s[idx] / (sum(s[idx]) + 1e-20) * 2.5       # bias NOT in weight :732-735
```

- **MultiRouter** `[R] :754-758, 801-815`: all three gates are evaluated;
  tokens with `image_mask` (the 256 `<imagePatch>` rows) take
  `image_gate`'s `idx/w`, every other row (text, `<image>`, `</image>`)
  takes `gate`'s. `audio_mask` is None.
- Expert k: `down(silu(gate(x)) * up(x))` in bf16; combine
  `sum_k out_k.f32 * w_k` in **f32**, then cast to bf16 `[R] :827-857`;
  `y = routed + shared_experts(x)` (un-gated, intermediate 512) `[R] :822-823`.
- Config keys that do **nothing** in this code: `norm_topk_prob`
  (normalisation is unconditional for top_k > 1), `use_expert_bias`
  (bias always added), `moe_router_topk_scaling_factor`, `norm_softmax`,
  `norm_head`, `use_sliding_window`/`sliding_window`, `use_qk_norm`
  (grep: none are read in `modeling_bailing_moe_v2.py`).
- `bias_update_coeff`/`update_bias` are training-only.

Precision: weights bf16, under `torch.cuda.amp.autocast(bf16)`
`[R] modeling_bailingmm2.py:679`; router logits/sigmoid/top-k in f32;
expert combine f32; norms f32 internally; softmax f32; RoPE tables rounded
to bf16. The CausalLM wrapper also computes `lm_head` logits for every row
(`num_logits_to_keep=0`, `[R] modeling_bailing_moe_v2.py:1908`) -- wasted,
skip it.

### 1.6 Which hidden states, and the index convention

`outputs.hidden_states` is built `[R] modeling_bailing_moe_v2.py:1724-1770`:
appended **before** each layer, then once more after `self.norm`.
So it has 21 entries:

- `hs[0]` = embedding output (input to layer 0),
- `hs[k]`, k = 1..19 = residual stream **after layer k-1, un-normed**,
- `hs[20]` = `norm(output of layer 19)` -- **post-final-RMSNorm** (= `hs[-1]`).

`selected_hidden_states_layers = [5, 12, 20]` therefore =
{after layer 4 (raw), after layer 11 (raw), final-normed}. In vpipe
`MetalQwenModel` tap numbering (tap L = HF `hs[L+1]`, un-normed,
`[V] generative-models/qwen3/metal-qwen-model.h:367-374`, entry `:403-407`; per its header
comment) that is taps **{4, 11, 19}** plus a host-side final RMSNorm
(`model.model.norm.weight`, eps 1e-6) on tap 19.

### 1.7 Stream B: "directvlm" (`use_vlm_directvlm_condition: true`)

`[R] modeling_bailingmm2.py:716-734`:

```
use = (labels < 0) * attention_mask          # 1 on the N text rows only
n   = argmax(cumsum(use)) + 1                # = N (first index of the max)
B   = cat([hs[5][:, :n], hs[12][:, :n], hs[20][:, :n]], -1) * use[..., None]   # [1, N, 6144]
cap_feats_2 = Linear_{6144->3840}( RMSNorm_{6144, eps 1e-5}(B) )                # [1, N, 3840]
```

- The three layers are concatenated **then** normalised by **one** RMSNorm
  over all 6144 channels (diffusers `RMSNorm`: statistics f32, normalised
  value cast to the bf16 weight dtype, then `* weight`). Two inputs are
  raw residual streams, one is final-normed -- the joint RMS mixes scales.
- Weights `proj_directvlm.0.weight` [6144], `proj_directvlm.1.{weight
  [3840,6144], bias [3840]}`, F32 on disk, cast to bf16
  `[R] :410-424`.
- Covers the N template tokens only (the 258 appended rows have label 1).
- Output dim 3840 = DiT `dim`: it **bypasses** the DiT's `cap_embedder`
  (section 2).
- Because the LLM is causal, stream B does not depend on the query tokens.

### 1.8 Stream A: learnable tokens -> connector (`use_learnable_token_condition: true`)

`[R] modeling_bailingmm2.py:736-763`:

```
Q   = hs[20] rows where gen_mask == 1           # [1, 256, 2048], in sequence order (masked_select)
E   = proj_in(Q)                                # Linear 2048->1536 (+bias)
C   = Qwen2Model_connector(inputs_embeds=E,
        attention_mask=ones[1,1,256,256] (float32)).hidden_states[-1]   # [1, 256, 1536]
cap_feats = proj_out(C)                         # Linear 1536->2560 (+bias)  -> [1, 256, 2560]
# connector_norm: mlp/config.json says false -> NO F.normalize           :762-763
```

Connector (`connector/config.json`): Qwen2, 28 layers, hidden 1536,
12 q / 2 kv heads (GQA 6), head_dim 128, **q/k/v bias**, no o bias, no q/k
norm, SwiGLU 8960, RMSNorm eps 1e-6, `rope_theta` 1e6, full-head
rotate-half RoPE, positions 0..255 (no `position_ids` passed ->
`cache_position` arange, `[T] models/qwen2/modeling_qwen2.py:352-357`).
F32 on disk, loaded with `torch_dtype=bf16` `[R] :388`.

**It runs bidirectionally:** every layer gets `self_attn.is_causal = False`
`[R] :389-390`, and the explicit 4D mask is returned as-is by transformers
4.57.1 (`[T] masking_utils.py:710-711`), so SDPA gets `is_causal=False`
(`[T] integrations/sdpa_attention.py:81`) and an additive float mask of
all ones -- a uniform +1 on every logit, which softmax cancels. Net: **full
attention, no mask**. `hidden_states[-1]` = `model.norm(last layer)`
(`[T] utils/generic.py:1089-1091` replaces the last recorded state with
`last_hidden_state`). The connector's `embed_tokens`/tied `lm_head` are
unused.

### 1.9 What the DiT receives

`ConditionedTransformer.forward` `[R] diffusion/generator.py:55-88`:
`text_encoder_norm` false (`mlp/config.json`) so no `normalize*1000`
`:69-70`; `mlp` is `nn.Identity()` (`use_identity_mlp` true, `:39`;
`mlp/model.safetensors` has no `mlp.*` keys so the strict empty load
passes, `[R] modeling_bailingmm2.py:427-430`, `generator.py:210-211`).
Then `transformer(x, cap_feats=[256x2560], t, cap_feats_2=[Nx3840])`.

Inside the DiT (section 2.3) the caption sequence is
`[cap_embedder(cap_feats) : 256 rows][cap_feats_2 : N rows][zeros : pad]`,
`pad = (-(256+N)) mod 32`. Dtype bf16. There is no separate caption mask
input -- padding is the only masking (section 2.3).

Sample (N = 34): caption 256 + 34 = 290 real rows, 30 pad, cap_total 320.

### 1.10 CFG / negative

- `do_classifier_free_guidance = guidance_scale > 1` `[R] pipeline.py:291-293`.
  The recommended **cfg 1.0 disables CFG entirely**: the negative is never
  run (`[R] pipeline.py:578, 590-597`).
- When cfg > 1, the negative is **zeros shaped like the positive
  conditioning**, not an empty prompt: `negative_prompt_embeds = [e*0]`
  for both streams `[R] generator.py:266-276` (and `[R] modeling_bailingmm2.py:180-184`).
  In the DiT that gives 256 rows of `cap_embedder(0) = cap_embedder.1.bias`
  (RMSNorm(0) = 0) followed by N zero rows. Same lengths, same positions.
- Formula `pred = pos + g * (pos - neg)` `[R] pipeline.py:651` (so g=2 means
  3*pos - 2*neg), optional norm clamp `cfg_normalization` (default False),
  truncation `cfg_truncation` (default 1.0: guidance off when
  `1 - sigma > 1.0`, i.e. never) `[R] pipeline.py:567-578, 653-659`.
  `ImageGenerator.sample` passes neither, so defaults apply.

---

## 2. DiT

### 2.1 Relation to diffusers `ZImageTransformer2DModel` (0.36.0)

A textual diff of `[R] diffusion/transformer.py` against
`[D] models/transformers/transformer_z_image.py` (class names normalised)
shows **identical math** except:

1. `axes_lens` default `[20480, 512, 512]` (config agrees) vs
   `[1024,512,512]` default / `[1536,512,512]` in Z-Image checkpoints
   `[R] :331`.
2. `alignment_padding_mode` (`"learned"` | `"zero_masked"`) and
   `multi_frame_output` constructor args `[R] :332-356`.
   `zero_masked`: no `x_pad_token`/`cap_pad_token` parameters
   `[R] :406-411`; pad rows set to 0 `[R] :632-633, 678-679`; pad rows
   excluded from every attention mask `[R] :645-646, 694-695, 721-726`
   via `[R] diffusion/padding.py:8-45`.
3. A **second caption stream** `cap_feats_2` (already `dim`-wide) that is
   appended after `cap_embedder(cap_feats)` and carries the caption padding
   `[R] :491-527, 656-673`.
4. `ref_x`: reference latents concatenated on the frame axis (image edit /
   layers only) `[R] :530-531`.
5. `multi_frame_output=False` keeps only frame 0 at unpatchify `[R] :441-442`.
6. `TimestepEmbedder` drops diffusers' `compute_dtype` branch `[R] :66-72`
   (no numeric effect).

Checkpoint tensor names are exactly Z-Image's minus the pad tokens:
`all_x_embedder.2-1`, `all_final_layer.2-1.{adaLN_modulation.1,linear}`,
`cap_embedder.{0,1}`, `t_embedder.mlp.{0,2}`,
`{noise_refiner,context_refiner,layers}.N.{attention.{to_q,to_k,to_v,to_out.0,norm_q,norm_k},
attention_norm1/2, ffn_norm1/2, feed_forward.w1/w2/w3}`, `adaLN_modulation.0`
on `noise_refiner` and `layers` only. Dimensions are **identical to
Z-Image 6B** (dim 3840, 30 layers, 2+2 refiners, 30 heads, head_dim 128,
`cap_feat_dim` 2560 = Qwen3-4B's width, axes [32,48,48], theta 256,
t_scale 1000, patch 2). INFERRED: the DiT was fine-tuned from Z-Image; the
connector's `proj_out` to 2560 exists to feed Z-Image's `cap_embedder`.

### 2.2 Forward, exactly

Constants `ADALN_EMBED_DIM = 256`, `SEQ_MULTI_OF = 32` `[R] :36-37`.

- **Timestep** `[R] :40-72, 606-607`: `t_in = t * 1000` where the pipeline
  passes `t = 1 - sigma` (section 3). `f = exp(-ln(10000) * i/128)`,
  i = 0..127; `emb = [cos(t_in f), sin(t_in f)]` (cos first), f32, cast
  to bf16; `temb = Linear(1024->256)(SiLU(Linear(256->1024)(emb)))`.
  `adaln_input = temb` (**no SiLU** before the block adaLN).
- **x embed**: patchify `[16, 1, H, W]` -> rows of `(pF=1, pH, pW, C)`,
  C fastest, row-major over (h, w) tokens `[R] :537-539`;
  `Linear(64 -> 3840)` + bias.
- **Block** `[R] :169-241` (sandwich norms, all RMSNorm eps `norm_eps`
  1e-5; per-head q/k RMSNorm eps 1e-5 via `Attention(qk_norm="rms_norm")`
  `[R] :187-197`):
  ```
  modulated:   [s_msa, g_msa, s_mlp, g_mlp] = Linear_{256->4*3840}(temb).chunk(4)
               x += tanh(g_msa) * norm2_attn( Attn( norm1_attn(x) * (1+s_msa) ) )
               x += tanh(g_mlp) * norm2_ffn ( FFN ( norm1_ffn(x)  * (1+s_mlp) ) )
  unmodulated: x += norm2_attn(Attn(norm1_attn(x)));  x += norm2_ffn(FFN(norm1_ffn(x)))
  Attn: q,k,v = x W (no bias), 30 heads x 128, q/k per-head RMSNorm,
        RoPE, SDPA (scale 1/sqrt(128), bool KEY mask), to_out.0 (no bias)
  FFN:  w2( silu(w1 x) * w3 x ), hidden int(3840/3*8) = 10240
  ```
- **RoPE** `[R] :113-122, 262-305`: complex multiply on **interleaved
  pairs** `(x[2k], x[2k+1])` in f32; per axis a `d`-dim table,
  `freq_j = 256^(-2j/d)` (computed f64, outer with position in f64, then
  f32 `polar`); concatenated `[axis0: 16 pairs | axis1: 24 | axis2: 24]`.
- **Order** `[R] :590-737`: x_embedder -> **noise_refiner** (2, modulated,
  image rows only) -> cap path: `cap_embedder` = `RMSNorm(2560, 1e-5)` +
  `Linear(2560->3840)` applied to the 256 stream-A rows only, then append
  stream B -> **context_refiner** (2, unmodulated, caption rows only) ->
  unified `[image rows ; caption rows]` -> 30 `layers` -> final layer.
- **Final layer** `[R] :244-259`: `LayerNorm(no affine, eps 1e-6)(x) *
  (1 + Linear_{256->3840}(SiLU(temb)))` -> `Linear(3840 -> 64)` + bias.
- **Unpatchify** `[R] :426-443`: first `img_tokens` rows,
  `view(1, H/2, W/2, 1, 2, 2, 16).permute(6,0,3,1,4,2,5)` -> `[16,1,H,W]`.
- Pipeline then keeps frame 0, casts to f32 `[R] pipeline.py:636-639`.

### 2.3 Sequence layout, positions, zero_masked padding

`patchify_and_embed` `[R] :454-588` (per sample, batch 1):

| part | rows | RoPE position (axis0, axis1, axis2) |
|---|---|---|
| image token (h, w) | `img = (H/16)(W/16)` | `(cap_total + 1, h, w)` `[R] :544-548` |
| image pad | `(-img) mod 32` | `(0, 0, 0)` `[R] :549-557` |
| caption row i (stream A then B, **and its pads**) | `cap_total = roundup32(256 + N)` | `(1 + i, 0, 0)` `[R] :500-506` |

Unified order is `[image | image pad | caption | caption pad]`
`[R] :704-713`. The only masks are boolean **key** masks
`[B, 1, 1, L]` (`[R] :128-130`): False on every pad row. Pad query rows
still compute but are never read (masked as keys everywhere; unpatchify
takes the first `img` rows).

**Port consequence (INFERRED, follows from the above):** dropping all pad
rows and running the real rows only is mathematically identical, **as
long as positions keep the padded offsets** -- the image's axis-0 position
is `roundup32(256+N) + 1`, not `256+N+1`. That removes any need for a
masked attention kernel.

Sizes: 1024^2 -> 4096 image tokens; 2048^2 -> 16384; the sample prompt's
caption 320 -> joint 16704 rows at 2048^2. Max caption: axis-0 table has
20480 rows, so `cap_total + 1 <= 20479` (INFERRED ceiling, ~20k text tokens).

### 2.4 Against vpipe's in-tree Z-Image (`MetalZImageTransformer`)

Same: block math, sandwich norms, tanh gates, `(1+scale)`, unmodulated
context refiner, final layer, interleaved-pair 3-axis RoPE with one theta,
patch packing `(pH, pW, C)`, position layout including caption pads
continuing the axis-0 run and image pads at the origin
(`[V] generative-models/z-image/z-image-layout.cc:55-76`), timestep
`t_scale * (1 - sigma)` (`[V] metal-z-image-transformer.cc:901`), output
negation (`[V] stages/generate-image-stage.cc:4165`), CFG formula
`p + g(p - n)`. Dimensions are the Z-Image defaults.

Must change:

| # | Ming | vpipe Z-Image today | change |
|---|---|---|---|
| 1 | `_class_name` `DiffusionTransformer` | refuses anything but `ZImageTransformer2DModel` `[V] metal-z-image-transformer.cc:272-275` | accept (gate on `alignment_padding_mode` present) |
| 2 | no `x_pad_token`/`cap_pad_token`; pads = 0 and masked | loads both pad tokens `[V] :439-440`, fills pads with them, **no mask anywhere** | drop pad rows (keep padded positions), or zero + key-mask |
| 3 | caption = `cap_embedder(256x2560) ; 3840-wide N rows` | `cap_embedder` applied to every caption row | second, pre-embedded caption input |
| 4 | no caption cap (256 + N, N unbounded; 1407 for the README JSON) | `kMaxCaptionTokens = 512` `[V] z-image-layout.h:51` | lift for this family |
| 5 | `axes_lens` [20480,512,512] | read from config, used only as a bounds check | none (verify the check accepts it) |
| 6 | noise = torch CPU `randn(seed)` NCHW | `std::mt19937_64` in packed space `[V] generate-image-stage.cc:4044-4049` | use `TorchCpuRandn` `[V] generative-models/shared/torch-randn.h:32` for golden parity |
| 7 | "12 steps" with diffusers' `sigma_min = 0` | `simple` grid `linspace(1, 1/S, S)` + terminal 0 `[V] generative-models/krea2/flow-sampler.cc:256-268` | Ming 12 == vpipe `simple` **S = 11**, exactly (section 3) |

Precision differences that are fine: vpipe folds `(1+scale)` into the
norm weight in bf16, computes RoPE angles in f64 (reference: f32 `polar`
of an f32 angle), keeps `temb` bf16 -- all as the reference does or
within bf16 noise.

---

## 3. Sampler

### 3.1 Schedule (VERIFIED with diffusers 0.36.0)

`scheduler/scheduler_config.json`: `FlowMatchEulerDiscreteScheduler`,
`num_train_timesteps` 1000, `shift` 6.0, `use_dynamic_shifting` **false**.

`ImageGenerator.__init__` does `noise_scheduler.config['use_dynamic_shifting'] = True`
`[R] diffusion/generator.py:214`, and the pipeline computes a `mu`
`[R] pipeline.py:525-537`. **Both are dead.** `config` is a diffusers
`FrozenDict`. The item assignment succeeds because its
`hasattr(self, "__frozen")` guard never matches the name-mangled attribute
(`[D] configuration_utils.py:55-84`). But `set_timesteps` reads the
*attribute* `self.config.use_dynamic_shifting` (`[D] schedulers/scheduling_flow_match_euler_discrete.py:275, 312`),
which was frozen at construction as False. So the **static shift 6.0
applies at every resolution**; `mu`, `max_shift`, `max_image_seq_len` do
nothing. The pipeline sets `scheduler.sigma_min = 0.0` `[R] pipeline.py:538`.

```
u_k     = linspace(1, 0, 12)[k] = 1 - k/11               (sigma_max 1 -> sigma_min 0)
sigma_k = 6 u / (1 + 5 u)                                 [D] :315
timesteps_k = 1000 * sigma_k ; sigmas += [0]              [D] :331-344
```

| k | sigma_k | timestep | DiT `t = 1-sigma` (x1000 inside) |
|---|---|---|---|
| 0 | 1.0 | 1000.0 | 0.0 |
| 1 | 0.98360658 (60/61) | 983.6066 | 0.01639343 |
| 2 | 0.96428573 (54/56) | 964.2857 | 0.03571429 |
| 3 | 0.94117647 (48/51) | 941.1765 | 0.05882355 |
| 4 | 0.91304350 (42/46) | 913.0435 | 0.08695648 |
| 5 | 0.87804878 (36/41) | 878.0488 | 0.12195123 |
| 6 | 0.83333331 (30/36) | 833.3333 | 0.16666669 |
| 7 | 0.77419353 (24/31) | 774.1935 | 0.22580646 |
| 8 | 0.69230771 (18/26) | 692.3077 | 0.30769226 |
| 9 | 0.57142860 (12/21) | 571.4286 | 0.42857140 |
| 10 | 0.375 (6/16) | 375.0 | 0.625 |
| 11 | 0.0 | 0.0 | 1.0 |
| (terminal) | 0.0 | -- | -- |

Closed form: `sigma = 6m / (11 + 5m)`, m = 11-k.

**The 12th forward is a no-op**: at k = 11, `dt = sigma_12 - sigma_11 = 0 - 0`,
so `x += 0 * v`. A port runs **11 DiT forwards** with
`sigmas = shift6(linspace(1, 1/11, 11)) + [0]`. That is exactly vpipe's
`simple` grid at S = 11, and bit-identical to the reference barring
inf/NaN in the discarded output. The same quirk exists in diffusers'
Z-Image pipeline (`[D] pipelines/z_image/pipeline_z_image.py:477`).

### 3.2 Per-step update

`[R] pipeline.py:557-672`:

```
t_dit = (1000 - timestep_k) / 1000            # = 1 - sigma_k, f32           :563
v     = DiT(x_k.bf16, t_dit, caps).float()    # [16, H/8, W/8]               :591, 628-639
v     = -v                                    # NEGATED                      :669
x_{k+1} = x_k + (sigma_{k+1} - sigma_k) * v   # Euler, f32                   [D] :447-458
```

Equivalently `x_{k+1} = x_k + (sigma_k - sigma_{k+1}) * DiT(...)`: the
model predicts data-minus-noise in its own `t` = 1 - sigma convention
(INFERRED). Latents stay f32 between steps `[R] pipeline.py:476, 672-673`.

### 3.3 Initial noise

`prepare_latents` `[R] pipeline.py:263-285`:
shape `(1, 16, 2*(H//16), 2*(W//16))` = `[1, 16, H/8, W/8]`, **f32**,
`randn_tensor(shape, generator=torch.manual_seed(seed))`
(`[R] generator.py:285`). `torch.manual_seed` returns the default CPU
generator, so `randn_tensor` draws on the **CPU** and moves the result
(`[D] utils/torch_utils.py:160-186`). The result is
`torch.Generator("cpu").manual_seed(seed)` -> `torch.randn(16*h*w)` in
NCHW order, reproducible bit-for-bit with
`[V] generative-models/shared/torch-randn.h:32` (`TorchCpuRandn`).
A seed that is None or negative becomes `datetime.now().microsecond % 1000`
`[R] modeling_bailingmm2.py:255-257`.

### 3.4 Resolution buckets

H and W must be multiples of 16 (`vae_scale_factor*2`, `[R] pipeline.py:427-437`).

**Step 1: pick the bucket.** `[R] infer.py:26-35, 125-160`:
- `--resolution r`, or for `--width W --height H`
  `r = round(sqrt(W*H))` (W/H in [1/4, 4]); with no flag the default is 2048.
- `bucket = min({1024, 2048}, key=(|b - r|, b))`. Ties go to 1024, so
  anything up to 1536 maps to 1024.

**Step 2: pick the aspect within the bucket.**
- `ar = W/H` if `W != H`, else None `[R] infer.py:434`.
- `get_default_image_gen_hw(bucket, ar) = process_ratio(512, int(512*ar), bucket)`
  `[R] bailingmm_utils.py:498-504`.
- `process_ratio` `[R] bailingmm_utils.py:325-419`: take the table whose key
  is nearest the bucket (512/672/1024/2048). The `closest_size` is the value
  whose key (a string of **h/w**) minimises `|key - h/w|`
  (`[R] :320-323`). Ties go to the first in dict insertion order; the 2048
  table has extra keys appended unsorted. Values are `[h, w]`.

**Step 3: generate.** `generate` re-runs `process_ratio` on that `(h, w)`,
which returns the same size with `resize_size == closest_size`, so its
LANCZOS resize is a no-op (`[R] modeling_bailingmm2.py:238-250, 281-289`).

**Step 4: final resize.** If `--width/--height` were given and differ from
the bucket size, the result is PIL-`LANCZOS` resized to `(W, H)`
(`[R] infer.py:454-459`). INFERRED from Pillow's implementation: Pillow
resamples RGBA premultiplied ("RGBa").

Examples (VERIFIED by executing the reference functions):

| request | bucket | diffusion H x W (latent, tokens) | output |
|---|---|---|---|
| default / `--resolution 2048` / 2048x2048 | 2048 | 2048x2048 (256x256, 16384) | same |
| `--resolution 1024..1536` / 1024x1024 | 1024 | 1024x1024 (128x128, 4096) | same |
| 2560x1440 | 2048 | 1440x2560 (180x320, 14400) | same |
| 4096x1024 | 2048 | 1024x4096 (128x512, 16384) | same |
| 1280x720 | 1024 | 720x1280 (90x160, 3600; image pad 16 rows) | same |
| **1920x1080** | 1024 | 720x1280 | **LANCZOS up to 1920x1080** |
| 1024x768 | 1024 | 864x1152 (108x144, 3888) | LANCZOS down |
| 1200x1200 | 1024 | 1024x1024 | LANCZOS up |

The tables, verbatim (`[R] bailingmm_utils.py:340-358, 373-397`; key = h/w, value = [h, w]):

```python
ASPECT_RATIO_1024 = {
 "0.25": [512, 2048], "0.26": [512, 1984], "0.27": [512, 1920], "0.28": [512, 1856],
 "0.32": [576, 1792], "0.33": [576, 1728], "0.35": [576, 1664], "0.4": [640, 1600],
 "0.42": [640, 1536], "0.48": [704, 1472], "0.5": [704, 1408], "0.52": [704, 1344],
 "0.5581": [768, 1376], "0.5625": [720, 1280], "0.5647": [768, 1360], "0.57": [768, 1344],
 "0.6": [768, 1280], "0.622": [816, 1312], "0.625": [800, 1280], "0.65": [832, 1280],
 "0.6582": [832, 1264], "0.6667": [832, 1248], "0.6709": [848, 1264], "0.68": [832, 1216],
 "0.7013": [864, 1232], "0.72": [832, 1152], "0.7467": [896, 1200], "0.75": [864, 1152],
 "0.7568": [896, 1184], "0.78": [896, 1152], "0.8": [896, 1120], "0.8056": [928, 1152],
 "0.82": [896, 1088], "0.88": [960, 1088], "0.94": [960, 1024], "0.9846": [1024, 1040],
 "1.0": [1024, 1024], "1.07": [1024, 960], "1.13": [1088, 960], "1.21": [1088, 896],
 "1.2414": [1152, 928], "1.25": [1120, 896], "1.2807": [1168, 912], "1.29": [1152, 896],
 "1.3333": [1152, 864], "1.3393": [1200, 896], "1.38": [1152, 832], "1.46": [1216, 832],
 "1.4906": [1264, 848], "1.5": [1248, 832], "1.6": [1280, 800], "1.67": [1280, 768],
 "1.75": [1344, 768], "1.7708": [1360, 768], "1.7778": [1280, 720], "2.0": [1408, 704],
 "2.09": [1472, 704], "2.4": [1536, 640], "2.5": [1600, 640], "2.89": [1664, 576],
 "3.0": [1728, 576], "3.11": [1792, 576], "3.62": [1856, 512], "3.75": [1920, 512],
 "3.88": [1984, 512], "4.0": [2048, 512],
}
ASPECT_RATIO_2048 = {
 "0.25": [1024, 4096], "0.26": [1024, 3968], "0.27": [1024, 3840], "0.28": [1024, 3712],
 "0.32": [1152, 3584], "0.33": [1152, 3456], "0.35": [1152, 3328], "0.4": [1280, 3200],
 "0.42": [1280, 3072], "0.48": [1408, 2944], "0.5": [1408, 2816], "0.52": [1408, 2688],
 "0.5625": [1440, 2560], "0.57": [1536, 2688], "0.6": [1536, 2560], "0.6667": [1664, 2496],
 "0.68": [1664, 2432], "0.72": [1664, 2304], "0.75": [1824, 2432], "0.78": [1792, 2304],
 "0.7917": [1824, 2304], "0.8": [1792, 2240], "0.82": [1792, 2176], "0.88": [1920, 2176],
 "0.94": [1920, 2048],
 "1.0": [2048, 2048], "1.07": [2048, 1920], "1.13": [2176, 1920], "1.21": [2176, 1792],
 "1.25": [2240, 1792], "1.2632": [2304, 1824], "1.29": [2304, 1792],
 "1.3333": [2432, 1824], "1.38": [2304, 1664],
 "1.46": [2432, 1664], "1.5": [2496, 1664], "1.67": [2560, 1536], "1.75": [2688, 1536],
 "1.7778": [2560, 1440], "2.0": [2816, 1408], "2.09": [2944, 1408], "2.4": [3072, 1280],
 "2.5": [3200, 1280], "2.89": [3328, 1152], "3.0": [3456, 1152], "3.11": [3584, 1152],
 "3.62": [3712, 1024], "3.75": [3840, 1024], "3.88": [3968, 1024], "4.0": [4096, 1024],
 "0.2941": [1120, 3808], "0.3043": [1120, 3680], "0.3679": [1248, 3392],
 "0.3846": [1280, 3328], "0.433": [1344, 3104], "0.4574": [1376, 3008],
 "0.4681": [1408, 3008], "0.5465": [1504, 2752], "0.6296": [1632, 2592],
 "0.6582": [1664, 2528], "0.7432": [1760, 2368], "0.8551": [1888, 2208],
 "0.9692": [2016, 2080], "1.0317": [2080, 2016], "1.1695": [2208, 1888],
 "1.3455": [2368, 1760], "1.5192": [2528, 1664], "1.5882": [2592, 1632],
 "1.8298": [2752, 1504], "1.913": [2816, 1472], "2.1364": [3008, 1408],
 "2.186": [3008, 1376], "2.3095": [3104, 1344], "2.6": [3328, 1280],
 "2.7179": [3392, 1248], "3.2857": [3680, 1120], "3.4": [3808, 1120],
}
```

(The 512 and 672 tables, `[R] :326-339, 360-371`, are reachable only via
the layer/edit tasks.) Note that not every 1024 bucket is ~1 MP:
`"0.5625": [720, 1280]` is 0.92 MP and `"1.7778": [1280, 720]` likewise.

---

## 4. VAE

### 4.1 Config and class

`vae/config.json`: `AutoencoderKLQwenImage`, `base_dim` 96, `dim_mult`
[1,2,4,4] (/8), `num_res_blocks` 2, `attn_scales` [] (mid-block attention
only), `temperal_downsample` [false,true,true], `z_dim` 16,
**`input_channels` 4**, `latents_mean` null, `latents_std` null,
`scaling_factor` 8.0064, `shift_factor` 0.0. Weights bf16; encoder
`conv_in` [96,4,3,3,3], decoder `conv_out` [4,96,3,3,3], `post_quant_conv`
[16,16,1,1,1].

The vendored class `[R] diffusion/autoencoder_kl_qwenimage.py` is diffusers
0.36's QwenImage VAE plus two changes (diff VERIFIED):
1. `input_channels` drives encoder `conv_in` and decoder `conv_out` channels
   `[R] :403, 420, 580, 632, 695-713`.
2. `QwenImageRMS_norm` normalises in **f32** for bf16/f16 input
   `[R] :108-116`: `F.normalize(x.float(), dim=C).to(x.dtype) * sqrt(C) * gamma`.

`scaling_factor`/`shift_factor` are not constructor args. diffusers warns
"not expected and will be ignored", but keeps them as hidden config entries,
so `vae.config.scaling_factor == 8.0064` (VERIFIED). The profile requires
this class with 4 channels and argmax encoding
(`[R] inference_profile.py:123-139, 271-302`).

### 4.2 Decode path

`[R] pipeline.py:698-708`:

```
z   = latents.bf16.unsqueeze(2)                    # [1,16,1,H/8,W/8]   (temperal_downsample present)
z   = z / 8.0064 + 0.0                             # scaling/shift; latents_mean/std UNUSED
img = vae.decode(z)                                # post_quant_conv, decoder per frame, clamp[-1,1]  [R] vae:842-866
img = img.squeeze(2)                               # [1,4,H,W]
pil = VaeImageProcessor.postprocess(img)           # (x*0.5+0.5).clamp(0,1) -> round(x*255) uint8 -> Image.fromarray -> "RGBA"
```

- **One frame of the causal 3D VAE**: the loop runs once
  (`[R] autoencoder_kl_qwenimage.py:853-858`) with an empty `feat_cache`.
  Each `CausalConv3d` zero-pads 2 frames in front, so only the last temporal
  kernel slice (kt = 2) contributes. The decoder's temporal upsample
  `time_conv` is skipped on the first chunk (the `"Rep"` path, `[R] :181-192`).
  Net: a 2-D decoder.
- No tiling (`use_tiling` False and never enabled, `[R] :725`).
- Latent normalisation is a **single scalar**: `z_vae = z_model / 8.0064`.
  The DiT's latent space is `z_model = 8.0064 * z_vae` (encode side,
  `[R] pipeline.py:499`). INFERRED: this is a retrained RGBA VAE with
  Qwen-Image's architecture, not Qwen-Image's weights; its raw latent std
  is ~1/8.
- **RGBA**: channel 3 is alpha in the same [-1,1] -> [0,1] mapping as colour.
  It is **straight (not premultiplied)**. Evidence: the encoder is fed
  `PIL.convert("RGBA")` straight pixels as `(to_tensor - 0.5) * 2`
  (`[R] processing_bailingmm2.py:165-180`), and the output goes straight to
  `Image.fromarray` / PNG. Opaque content decodes to alpha 255.
  `postprocess` is VERIFIED to return a PIL "RGBA" image for 4 channels.
- Output saved as PNG `image_00.png` (`[R] infer.py:246-259`).

### 4.3 Against vpipe's QwenImage VAE (`MetalKrea2Vae`)

`[V] generative-models/krea2/metal-krea2-vae.{h,cc}` already implements
this architecture for the z16 /8 family: the 2-D collapse of causal 3x3x3
convs to the kt = 2 slice, the skipped `time_conv`, `post_quant_conv`,
clamp, the `[C,H,W]` RGBA beat with `(x+1)/2` and straight alpha, and
RGBA PNG save. It does **not** decode this checkpoint as-is:

1. **Null stats make both VAE stages inert.** `config_from_json`
   (`[V] metal-krea2-vae.cc:141-200`) reads `latents_mean/std` and ignores
   `scaling_factor`/`shift_factor`. vae-decode then refuses
   (`[V] stages/vae-decode-stage.cc:1049-1056`), and vae-encode likewise
   (`[V] stages/vae-encode-stage.cc:822-829`).
   Fix: when the stats are null, use `mean = shift_factor`,
   `std = 1/scaling_factor` per channel. The existing un-whiten
   `latent*std + mean` then equals `latent/8.0064`.
2. **`input_channels` is never read**, so `out_channels` stays 3. The
   weights still load (shapes come from the tensors), but the decode
   transpose then strides 3 over 4-channel data: scrambled RGB, alpha lost
   (per the in-tree survey: clamp `cc:1898`, transpose `cc:1940-1948`).
   Fix: map `input_channels` to both `in_channels` and `out_channels`, and
   refuse a load where `conv_out.cout != out_channels`.
3. The class has no explicit `vae_family_()` row. It reaches "krea2" only via
   the fall-through (`[V] stages/vae-decode-stage.cc:310`). That works, but
   is the stale-fall-through hazard: a per-family chain whose last arm is
   a default, which goes stale the next time a family is added.

Not the 2.1 path: vpipe's "Qwen-Image-2.1" VAE is
`AutoencoderKLQwenImage21` (z64, /16). This checkpoint is the z16 /8
Qwen-Image / Krea-2 geometry. An out-of-tree alternative to an in-tree
fix: `VaeModelRegistry::claim_for` is asked first
(`[V] stages/vae-decode-stage.cc:755`; `[V] generative-models/vae-model-registry.h:536`).

---

## 5. Everything else a port needs

### 5.1 Precision map (reference, `--dtype bfloat16`)

| part | weights | compute |
|---|---|---|
| LLM | bf16 | bf16 GEMMs, f32 norm stats / softmax / router / expert combine, RoPE cos/sin rounded to bf16 |
| `query_tokens`, `proj_in/out`, `proj_directvlm` | F32 on disk -> **bf16** at load (`[R] modeling_bailingmm2.py:376, 407-408, 424`) | bf16 |
| connector | F32 on disk -> **bf16** (`[R] :388`) | bf16 (SDPA) |
| DiT | bf16 | bf16; timestep sinusoid f32 then bf16; RoPE math f32; RMSNorm stats f32 |
| scheduler / latents / CFG | -- | **f32** |
| VAE | bf16 | bf16, RMS normalise in f32 |

There is no fp32 island beyond these. `_skip_layerwise_casting_patterns`
(`[R] transformer.py:312`) only matters for layerwise casting, which is not
used.

### 5.2 Prompt length

There is no truncation anywhere on the T2I path. The processor does not
truncate (`[R] processing_bailingmm2.py:188, 347`). `max_sequence_length=512`
(`[R] generator.py:289`) only feeds `_encode_prompt`, which is bypassed
because embeddings are passed in (`[R] pipeline.py:451-452`). Limits:
the LLM's `max_position_embeddings` 32768, and the DiT axis-0 RoPE table of
20480 (section 2.3). Long JSON prompts (~1.4k tokens) are the advertised use.

### 5.3 Transparent background

The transparent background is **just prompt text**. The README lists ten
fixed prefixes, e.g. `RGBA, 4-channel, transparent background` and
`带透明通道，4通道RGBA图像` (`[R] README.md:180-194`). No code inspects
them. The VAE always emits 4 channels.

### 5.4 Surprises (each one a likely port bug)

1. `use_dynamic_shifting = True` is a silent no-op. The schedule is static
   shift 6 at every size (section 3.1).
2. "12 steps" is 11 effective forwards. The last forward runs at sigma 0
   with dt = 0.
3. `hs[5]` and `hs[12]` are **pre-norm** residual streams, while `hs[20]` is
   **post-final-norm**. One RMSNorm covers the 6144-wide concatenation.
4. The Qwen2 connector runs **bidirectionally**: `is_causal=False` plus an
   all-ones float 4D mask.
5. `connector_norm` is **false** in `mlp/config.json`, although the code
   default is True. Do not L2-normalise the connector output.
6. The 256 query tokens route through **`image_gate`**, and text through
   `gate`. Three router weight sets live in every MoE layer.
7. The query tokens' RoPE `w` positions are centred, so they overlap the
   text positions and **go negative** for prompts under 126 tokens.
8. Several config knobs are ignored by the code: `norm_topk_prob`,
   `use_expert_bias`, `moe_router_topk_scaling_factor`, `use_qk_norm`,
   `sliding_window`. Behaviour is hard-coded: always renormalise, always
   bias the selection, always q/k-norm, scale by 2.5.
9. Caption order is [256 connector rows][N text rows][pads]. Only the first
   block passes through `cap_embedder`.
10. `zero_masked` pads are removable, but the image's axis-0 position still
    counts them.
11. The CFG negative is zeros, not an empty prompt. CFG only engages for
    g > 1, and g means `pos + g(pos - neg)`.
12. The VAE latent scale is 8.0064 with null mean/std: a new VAE.
13. 1920x1080 diffuses at 1280x720 and is LANCZOS-upscaled, because
    sqrt(W*H) = 1440 is nearer 1024 than 2048.
14. The LLM computes full-vocab logits for every row, and the ViT is
    loaded. Both are pure waste for T2I.

### 5.5 Reuse assessment for vpipe

- **DiT: about 95% reuse of `MetalZImageTransformer`**, with the changes in
  section 2.4: class name, no pad tokens with pads dropped or masked, a
  second pre-embedded caption input, no 512 caption cap, and a step-count
  mapping. Its tiers carry over unchanged: streaming, Sol, i8, ANE, NAX.
  The 6.15 B bf16 DiT is the same size class Z-Image streams on 16 GB.
- **VAE: `MetalKrea2Vae` architecture fits.** It needs the two
  config-reading fixes in section 4.3 and a guard.
- **Connector: `MetalQwenModel`'s math fits.** Use the
  `encoder_config_qie_` recipe with Qwen2 dims (qkv bias, no qk norm, plain
  norms; `[V] stages/diffusion-conditioner-stage.cc:715-728`), feed bf16
  embeddings, and tap layer 27 plus the host final norm. It needs a
  **non-causal** attention mode, which the model lacks (causal or prefix
  key-mask only). Alternatively, use the existing non-causal `sdpa_full*`
  kernels in a small standalone encoder: 256 rows, 28 layers.
- **LLM: the attention half maps onto `MetalQwenModel`.** That half covers
  per-head q/k norm, GQA, partial rotary, causal attention, the
  `forward_embeddings_taps_mrope` entry with custom 3-axis positions
  (`[V] metal-qwen-model.h:432-438`), and layer streaming. Genuinely new:
  - the **video_rope axis map** (freq k < 24 alternates H/W, k >= 24 is T;
    not Qwen3-VL's interleave);
  - **sigmoid + expert-bias + group-limited (8 groups, top 4, top-2 sum)
    top-8 routing, renormalised x 2.5**, with a **per-token choice between
    two routers**;
  - an **un-gated** shared expert (vpipe's finalize hard-codes a sigmoid
    shared gate);
  - **per-expert** HF tensors, to stack at load;
  - a dense first layer;
  - the fused `query_key_value` naming and `word_embeddings`/`dense` names.

  The tree's MoE routing is softmax-only (`moe_route_f16`). The gather,
  combine and sort machinery is reusable.
- **Sampler:** FlowSampler `simple`, static shift 6.0, S = 11. No new code.
- **`mlp/` heads:** three GEMMs, one RMSNorm and a 256-row embedding
  splice.

### 5.6 Top risks

1. **The 31.9 GB bf16 MoE LLM.** Each prompt is one ~N+258-row causal
   prefill, which touches nearly all 256 experts per layer (~300 tokens x 8
   picks). On 16 GB it must stream all 19 MoE layers at 1.62 GB each,
   ~32 GB read per prompt, or ~7 s at internal-SSD pread rates. Otherwise
   quantise the experts: w8 ~16 GB, w4 ~8 GB, with a quality check. Cache
   the conditioning per prompt; it is independent of seed and size.
2. **Routing fidelity.** Sigmoid scores plus bf16 expert bias plus
   group-limited top-k gives near-ties, so bf16 drift can flip experts.
   Verify by per-layer hidden-state rel-L2 and the routing-agreement rate
   against an fp32 reference, not by token-exactness. There is no decoding
   here.
3. **Linkage.** `MetalQwenModel`, `MetalZImageTransformer` and
   `MetalKrea2Vae` are compiled into libvpipe, not the plugin toolkit
   (`[V] generative-models/CMakeLists.txt:67, 87, 100-101`; toolkit target
   `[V] CMakeLists.txt:358`). Libvpipe is hidden by default, so an
   out-of-tree plugin cannot call them unless they are exported or
   vendored. Decide this before writing code. The plugin entry points are
   `ImageModelRegistry` (`[V] generative-models/image-model-registry.h:337`;
   claimed at `[V] stages/generate-image-stage.cc:1829` and by the
   conditioner at `[V] stages/diffusion-conditioner-stage.cc:1632`) and
   `VaeModelRegistry`.
4. **Tokenizer exactness.** vpipe routes any `\p{L}`+`\p{N}` Split regex to
   its hand-coded Llama-3/Qwen scanner (`[V] generative-models/tokenizer.cc:1198-1205`).
   Ling's regex differs (possessive quantifiers, single-newline `\s*[\r\n]`,
   single-digit `\p{N}`), and an NFC normaliser is declared. Gate on id-exact
   tests over Chinese, JSON, digits and newlines.
5. **Bidirectional connector and 3-axis LLM positions with negative
   values.** Both are easy to get "almost right" with no visible failure.
6. **Golden plan.** The reference wants CUDA autocast; on CPU the autocast
   blocks are inert and the bf16 weights still define compute. Dump these
   per stage with a fixed prompt:
   - ids;
   - `hs[5]`, `hs[12]` and `hs[20]` (text rows);
   - `hs[20]` (query rows);
   - `cap_feats_2`;
   - the connector output and `cap_feats`;
   - one DiT forward at a mid sigma on a small geometry. Keep it at or above
     the vpipe fast-path floors: >= 64 rows for matmul2d, `K >= 1024` for
     i8 -- a golden below either floor verifies the arm nobody takes;
   - the VAE decode of a fixed latent;
   - the 11-step latent with the torch CPU noise.

   An fp32 reference forward for the LLM gives the bf16-vs-fp32 bar.
