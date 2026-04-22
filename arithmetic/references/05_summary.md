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

Paulo et al. provide the methodological template for making SoRL feature interpretation systematic rather than ad hoc:

- **H: Apply automated explanation pipeline to SoRL abstract tokens.** Our current token analysis (subtask correlation heatmaps, vignettes) is manual and example-driven. Paulo's pipeline — LLM generates candidate explanations, scores them via activation/intervention correlation — could be adapted to abstract tokens: present the model with token activation patterns across examples, ask an LLM to name the feature, then score via intervention (does forcing the token produce the predicted behavior?).
- **H: Intervention scoring is directly applicable.** Their intervention scoring evaluates whether activating a feature causally produces the predicted effect — exactly what our surgical swap (finding 6) does. Framing our experiments in these terms makes the methodology more rigorous and comparable.
- **H: SoRL tokens should score higher on interpretability metrics than SAE latents on the same model.** SAEs extract features post-hoc; SoRL trains them in. If SoRL tokens yield higher explanation quality scores under Paulo's metrics, it is evidence that structured training produces more interpretable representations than post-hoc decomposition.
