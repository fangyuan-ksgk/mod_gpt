# 04 — Probing for Arithmetic Errors in Language Models

**Authors:** Yucheng Sun, Alessandro Stolfo, Mrinmaya Sachan  
**Venue:** EMNLP 2025, Suzhou, China  
**URL:** https://aclanthology.org/2025.emnlp-main.411/  
**PDF:** [04_sun_probing_arithmetic_errors.pdf](04_sun_probing_arithmetic_errors.pdf)

## Core Claim

Internal activations in language models encode enough information to detect arithmetic errors before they surface in output — lightweight linear probes trained on 3-digit addition generalize to chain-of-thought arithmetic in GSM8K and can guide selective re-prompting.

## Key Findings

- **Probes decode both predicted and correct answers** — hidden states contain representations of what the model will say AND what the correct answer is, simultaneously
- **>90% error detection accuracy** — lightweight probes predict model correctness with high reliability
- **Generalization to GSM8K CoT** — probes trained on controlled addition transfer to multi-step reasoning problems without retraining
- **Selective re-prompting** — using probe predictions to trigger re-prompting only on likely-wrong steps improves overall accuracy while minimizing disruption to correct outputs
- **Self-correction via internal signals** — errors are anticipatable from activations alone, without external ground truth

## Relevance to This Study

Sun et al. do two things we want to match and extend with SoRL:

- **H: SoRL token forcing ≈ selective re-prompting.** Their re-prompting targets steps where probes predict errors. SoRL's surgical token swap (finding 6) is an analogue: forcing a corrected abstract token at inference time fixes wrong answers. We should demonstrate this more systematically — framed as "token forcing" rather than just swap analysis.
- **H: SoRL representations admit high-quality linear probes.** Their probes decode both the model's predicted answer and the correct answer from hidden states. We should run equivalent probes on SoRL abstract token representations and on baseline hidden states, and compare probe accuracy. The prediction: SoRL tokens are *more probeable* than baseline activations at the same position, because they are trained to encode subtask-relevant structure explicitly.
- **H: Probe quality correlates with token specialization.** Models with higher token subtask purity (exp 03 heatmap) should yield better probes — this ties the two lines of evidence together.
