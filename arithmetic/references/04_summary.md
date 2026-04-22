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

Complementary to our approach: this paper reads arithmetic correctness from hidden activations; we externalize intermediate computation as explicit tokens. Both confirm that arithmetic structure is represented internally in transformers. The probing methodology (linear probes on hidden states) is a useful baseline for comparing what activation-level vs token-level interpretability can recover.
