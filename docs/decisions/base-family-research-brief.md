# Base-family research brief (feeds the base-family ADR)

Status: research notes, not a decision. Gathered 2026-06-27 via firecrawl.
Motivation: gnosis-medpolicy-0.1 (Gemma-4-12B grounding SFT) showed SFT can move grounding
behavior but only by trading away correct priors (synthetic gate +48, real ICD -28). Re-examine
whether a different base family is a better substrate for low-confabulation, tool-grounded RLM
behavior. See handoff + `project_gnosis-medpolicy-0.1-grounding-tradeoff`.

## Hardware envelope (the real constraint, NOT ~12B)
Single RTX 5090 32GB (SM_120, CUDA 13), vLLM-served, QLoRA train + W4A16 infer.
- 32B dense @ Q4/W4A16 ≈ 17-20GB weights -> fits with long-context KV headroom (5090 is "the
  first single consumer 32GB card that can run 32B at Q8"; Q4 is comfortable). [insiderllm VRAM]
- 30/35B-A3B MoE @ Q4 ≈ 20-24GB weights, ~3B active/token (fast), vLLM serves MoE well. QLoRA on
  MoE is doable but more fiddly than dense.
- Conclusion: candidate space is ~8B up to ~32B dense and ~35B-total MoE. The 12B cap was an
  inherited assumption and is dropped.

## Key evidence: Vectara HHEM faithfulness leaderboard (direct proxy for our failure mode)
RAG summarization hallucination rate (lower better) / answer rate. Self-hostable rows:

| Model | Halluc % | Answer % | License | Fits 32GB | Note |
|---|---|---|---|---|---|
| antgroup/finix_s1_32b | 1.8 | 99.5 | Apache 2.0 (s1 base) | yes (32B) | s1 = test-time-scaling *reasoning* finetune; niche, verify general tool-use |
| microsoft/Phi-4 (14B) | 3.7 | **80.7** | MIT | yes | low halluc partly by *abstaining* — risky given our over-grounding problem |
| google/gemma-3-12b-it | 4.4 | 97.4 | Gemma | yes | reference (prior family gen) |
| **qwen/qwen3-8b** | **4.8** | **99.9** | **Apache 2.0** | yes | faithful WITHOUT over-abstaining; best blend |
| ibm-granite/granite-4.0-h-small | 5.2 | 100.0 | Apache 2.0 | yes | enterprise RAG-tuned, answers everything |
| google/gemma-4-26b-a4b-it | 5.2 | 99.8 | Gemma | yes | **our family (MoE variant)** |
| mistralai/mistral-small-2501 | 5.1 | 97.9 | Apache 2.0 | yes (24B) | |
| qwen/qwen3-14b | 5.4 | 99.9 | Apache 2.0 | yes | |
| qwen/qwen3-32b | 5.9 | 99.9 | Apache 2.0 | yes | top of envelope |
| **google/gemma-4-31b-it** | **7.4** | 100.0 | Gemma | yes | **our family (dense) ranks POORLY** |
| mistralai/ministral-8b | 7.4 | 99.9 | (MNPL-ish) | yes | |

Takeaways:
1. **Gemma-4 ranks poorly on faithfulness** (dense 31b 7.4%, MoE 26b-a4b 5.2%) vs Qwen3-8b 4.8% and
   finix 1.8%. Quantitative support for "switch family", and direct ammo for the ADR. Our v13 is
   Gemma-4-12B-based.
2. **Answer rate matters for OUR calibration problem.** gnosis-0.1's failure was OVER-grounding
   (abandoning correct priors). Phi-4 gets low halluc by refusing a lot (80.7% answer) — wrong
   direction for us. We want faithful-to-context AND retains-priors: high faithfulness + high
   answer rate. Qwen3-8b (4.8% / 99.9%) is the standout on that joint metric.

## Tool-use / code (RLM writes Python in a REPL)
BenchLM tool-use leaderboard is frontier-scale at the top (GLM-5.1, Qwen3.7, Kimi). At the small
dense scale no ~8-14B leads, expected (tool use scales with size). Qwen consistently tops the
*open-weight* tool-use/BFCL rankings across sizes and has toggleable thinking + 128-256K context.
So at our scale the pick is "best grounding/calibration with adequate tool use" -> Qwen3 dense.

## Licensing (matters for the planned HF gnosis-lm release)
- Apache 2.0 (unrestricted, no caps): **Qwen 3/3.5/3.6 (all)**, Mistral Small/Large (current),
  IBM Granite 4, simplescaling/s1.
- MIT (unrestricted): DeepSeek, **Phi-4**, GLM-5.
- Gemma: permitted but requires accepting Google's terms (carries through to derivatives) — current
  friction we'd shed by switching.
- Llama 4/3.3: free under 700M MAU + EU multimodal restriction.
A switch to Qwen (Apache) or Phi/GLM/DeepSeek (MIT) is strictly *more* permissive than Gemma.

## Shortlist (ranked for our use)
1. **Qwen3-14B dense (Apache 2.0)** — primary probe pick. Best blend of faithful+answers+permissive,
   strong open-weight tool/code, 128K ctx, standard TRL/PEFT QLoRA (no gemma4_unified lift), clean
   W4A16 path, fits the 5090 with big KV headroom. (8B if we want max headroom; 32B for max quality.)
2. **Qwen3-32B dense (Apache)** — top of envelope, highest-quality Qwen that still fits.
3. **Qwen3-30B-A3B / 3.6-35B-A3B MoE (Apache)** — high capacity, ~3B active (fast); MoE QLoRA caveat.
4. **IBM Granite 4.0-h-small (Apache)** — purpose-built for grounded enterprise RAG, 100% answer.
5. **finix_s1_32b (Apache)** — best raw faithfulness but reasoning-finetune lineage; investigate.
Phi-4 noted but de-prioritized: its low halluc comes with heavy abstention (wrong for our calibration
problem).

## Recommended next experiment (handoff step 4, the cheap high-signal probe)
Serve **Qwen3-14B (or -8B) stock instruct** on :8080 via vLLM and run the EXISTING
`corpus_niah_ab.py` A/B (find_off/baseline/steer x corpus_niah/icd10_niah) with NO fine-tuning, vs
the Gemma-4-v13 numbers. Measures out-of-the-box grounding/confabulation on OUR gate for a fraction
of a retrain. Keep the W4A16 / 65536-window bar for faithfulness to production.
Requires tearing down the ad-hoc gnosis-0.1 serve on :8080 first.

## Switching-cost note (for the ADR)
Current stack is Gemma-4-specific: gemma4_unified arch (transformers 5.12.1), forced TRITON_ATTN
(heterogeneous 256/512 head dims), text-tower extraction + W4A16 pipeline, and all SFT/DPO data
rendered in the Gemma `<|turn>`/`<|channel>` template. A Qwen switch means: new chat template in the
data builders, a likely *simpler* standard TRL/PEFT QLoRA path (no gemma4_unified), a new W4A16/serve
recipe. Net likely a simplification on the training side; serve side is a fresh recipe either way.

Sources (in .firecrawl/): vectara-hallucination.md, lechmazur-confabulations.md, benchlm-tooluse.md,
computingforgeeks-comparison.md, gigagpu-licensing.md, vram-cheatsheet.md.

---

## PROBE RESULT (2026-06-27): stock Qwen3-14B-AWQ on our gate

Served Qwen/Qwen3-14B-AWQ (W4A16 g128, fp8_e4m3 KV, native ctx 40960, non-thinking) on :8080
via vllm-launch.sh. Ran the existing corpus_niah_ab.py 3-arm A/B, NO fine-tuning,
PREHEND_DYNAMIC_KV_POOL=1, conc 4, timeout 400. Harness change required: added
`rlm_enable_thinking=False` (orchestrator thinking toggle; prehend only disabled thinking on
sub-calls — a Gemma-shaped assumption). 0-3 infra fails/arm (excluded).

grounded_correct % (Qwen stock | v13 trained | gnosis-0.1 trained):
| arm | corpus_niah (synthetic) | icd10_niah (real ICD) |
|---|---|---|
| find_off | 11.9 | 46 | 53  ||  5.4 | 42 | 52 |
| baseline | 18.3 | 31 | 80  || 27.5 | 57 | 30 |
| steer    |  0.0 | 21 | 33  || 30.0 | 60 | 40 |

Synthetic-gate calibration (the substrate tell; correct behavior = honest refusal):
Qwen refused 0-6/60, confabulated 78-100%. Same confident-wrong pathology as Gemma, NOT better.
Retrieval works (grounding-rate 54-58%) but answers unfaithfully. steer_fewshot collapses to
0/60 on synthetic (Gemma-tuned briefing breaks Qwen; high prompt-sensitivity).

CONCLUSION: stock Qwen3-14B is below trained Gemma on every arm and shows NO out-of-box grounding
advantage. REFUTES "a better base family grounds better for free." Caveat: stock-vs-trained, not
apples-to-apples; the clean substrate test is stock-Gemma-4-12B vs stock-Qwen3-14B on the same gate.

---

## PANEL ADD (2026-06-27): stock Phi-4 W4A16 on the same gate

Served RedHatAI/phi-4-quantized.w4a16 (W4A16, fp8_e4m3 KV, native ctx 16384, Phi3ForCausalLM, no
thinking) on :8080, same harness/config as Qwen (PREHEND_DYNAMIC_KV_POOL=1, conc 4, timeout 400).
Outcome split (grounded_correct / honest_refusal / confab_or_wrong), %:

corpus_niah (synthetic; correct=find the retrievable needle):
| arm | gc% | refuse% | confab% |
|---|---|---|---|
| find_off | 0.0 | 85.0 | 15.0 |
| baseline | 8.3 | 71.7 | 20.0 |
| steer    | 5.1 | 71.2 | 23.7 |

icd10_niah:
| arm | gc% | refuse% | confab% |
|---|---|---|---|
| find_off | 2.5 | 45.0 | 52.5 |
| baseline | 0.0 | 75.0 | 25.0 |
| steer    | 0.0 | 73.7 | 26.3 |

Phi-4 is the OPPOSITE pole from Qwen: it ABSTAINS heavily (refuse 71-85% on synthetic) instead of
confabulating. Low confab = safe, but grounded_correct is near-zero (0-8%) because it declines rather
than digging for the retrievable needle. Matches its HHEM profile (low halluc via 80.7% answer rate)
and confirms the brief's Phi-4 caution: wrong failure mode for us (won't answer even when answerable).

Two stock non-Gemma bases now fail in OPPOSITE ways - Qwen confabulates, Phi-4 over-refuses - and
NEITHER grounds-and-answers. Strong indication that grounded-correct behavior comes from the
RLM/grounding TRAINING, not from a base with better default calibration. The Gemma anchor (does stock
Gemma also fail, or actually ground?) is now the deciding data point.

---

## ANCHOR + VERDICT (2026-06-27): stock Gemma-4-12B closes the panel

Served stock google/gemma-4-12b-it via a hand-built text-native checkpoint
(gemma-4-12B-it-text-native-stock): extracted the 666 model.language_model.* tensors from the
gemma4_unified multimodal checkpoint (prefix-strip -> Gemma4ForCausalLM), reused the v13-text-native
config verbatim (the native head-dim relabel; weights byte-identical to v13-text minus the SFT).
Served online-fp8 + fp8_e4m3 KV @ 65536 on vLLM (TRITON_ATTN auto). Smoke coherent ("Paris").

### Cross-family STOCK panel, grounded_correct % (clean read = synthetic; icd10 is prior-confounded)

corpus_niah (synthetic gate, unknowable-by-prior labels):
| arm | Gemma | Qwen3-14B | Phi-4 | (trained v13) |
|---|---|---|---|---|
| find_off | **50.0** | 11.9 | 0.0 | 46 |
| baseline | 15.0 | 18.3 | 8.3 | 31 |
| steer    | **21.7** | 0.0 | 5.1 | 21 |

icd10_niah (real ICD, parametric-prior CONFOUNDED):
| arm | Gemma | Qwen3-14B | Phi-4 | (trained v13) |
|---|---|---|---|---|
| find_off | **27.0** | 5.4 | 2.5 | 42 |
| baseline | 5.0 | **27.5** | 0.0 | 57 |
| steer    | 7.5 | **30.0** | 0.0 | 60 |

Stock failure modes: Qwen CONFABULATES (78-100% on synthetic), Phi-4 ABSTAINS (refuse 71-85%),
Gemma actually GROUNDS (find_off 50%, refuse only when unsure).

### VERDICT: STAY ON GEMMA. Do not switch families.
1. On the clean (synthetic) gate, stock Gemma is the BEST grounding substrate of the three -
   dominates find_off (50 vs 12 vs 0) and steer (21.7 vs 0 vs 5). The premise that drove this
   investigation ("Gemma confabulation -> a different family may ground better") is REFUTED: the two
   non-Gemma stock bases ground WORSE, each failing in a different way.
2. Stock Gemma MATCHES its own trained v13 on synthetic find_off (50 vs 46) and steer (21.7 vs 21);
   training's value concentrates where stock fails - synthetic baseline (15->31) and especially the
   parametric-prior icd10 arms (5->57, 7.5->60). So the lever is TRAINING on the Gemma base, not the
   base family.
3. icd10 is mixed (Qwen wins the tool arms) but it is the prior-confounded gate; the clean substrate
   signal is unambiguous for Gemma.
4. The Vectara HHEM population-level edge for Qwen (5.4% vs Gemma-4 7.4%) does NOT transfer to our
   harder RLM/NIAH gate; out of the box Qwen confabulates here.

NEXT: DPO 0.2 on Gemma (pairs already built, 41 pairs) to calibrate WHEN to ground - the original
plan, now evidence-backed. The base-family switch is closed as NOT WARRANTED.
