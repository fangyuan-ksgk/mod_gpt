# 05 — Automatically Interpreting Millions of Features in Large Language Models

**Authors:** Gonçalo Paulo, Alex Mallen, Caden Juang, Nora Belrose  
**Venue:** arXiv preprint (2024)  
**arXiv:** https://arxiv.org/abs/2410.13928  
**PDF:** [05_paulo_autointerp_features.pdf](05_paulo_autointerp_features.pdf)

## Core Claim

An automated pipeline using LLMs can generate and evaluate natural language explanations for sparse autoencoder (SAE) features at scale — millions of features across large models. SAE latents are substantially more interpretable than individual neurons, even sparsified ones.

## Key Findings

- **SAE latents >> neurons** — sparse autoencoder features are more interpretable than neurons even when neurons are individually sparsified
- **Five new scoring techniques** — including "intervention scoring" (evaluates causal effect of feature activation, not just correlation), outperforming prior recall-based methods
- **Intervention scoring captures distinct features** — finds interpretable features missed entirely by recall-based evaluation
- **Cross-SAE semantic similarity** — independently trained SAEs on nearby layers learn highly similar features; suggests features are stable and not arbitrary
- **Scale** — pipeline applied to millions of features; code and explanations released publicly

## Relevance to This Study

Directly relevant to the SAE component of this study (`arithmetic/interp_utils/sae_trainer.py`). The automated explanation pipeline is a template for how we could assign natural language descriptions to our abstract tokens. Intervention scoring (causal, not correlational) aligns with our own causal verification approach — both reject pure correlation in favor of intervention evidence. The finding that SAE latents are more interpretable than neurons supports using SAEs as a comparison baseline for SoRL token interpretability.
