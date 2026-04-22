# 03 — Progress Measures for Grokking via Mechanistic Interpretability

**Authors:** Neel Nanda, Lawrence Chan, Tom Lieberum, Jess Smith, Jacob Steinhardt  
**Venue:** arXiv preprint (2023)  
**arXiv:** https://arxiv.org/abs/2301.05217  
**PDF:** [03_nanda_grokking_progress_measures.pdf](03_nanda_grokking_progress_measures.pdf)

## Core Claim

Grokking — the delayed generalization that occurs long after training loss plateaus — is not a sudden phase transition. It unfolds through three continuous phases (memorization → circuit formation → cleanup), trackable via mechanistic progress measures derived from reverse-engineering the learned algorithm.

## Key Findings

- **Three training phases** — memorization (fits train set via lookup), circuit formation (structured Fourier/trig algorithm emerges in weights), cleanup (memorization components pruned, generalization jumps)
- **Fourier/trig algorithm** — transformers trained on modular addition learn to represent numbers as rotations on a circle, using discrete Fourier transforms and trigonometric identities
- **Progress measures** — continuous weight-space metrics that predict generalization long before test accuracy rises; enables early detection of whether a model will grok
- **Ablation in Fourier space** — confirms the algorithm by zeroing non-Fourier components and showing near-zero accuracy drop

## Relevance to This Study

Nanda's Fourier/trig basis findings have **not been verified through SoRL** — only Quirke's carry/borrow circuits have. This is a key secondary hypothesis:

- **H: Do SoRL abstract tokens recover Fourier-basis structure?** Nanda shows transformers represent numbers as rotations on a circle via discrete Fourier transforms. If SoRL abstract tokens encode the same Fourier components (detectable via DFT over token embeddings across examples), it suggests SoRL is externalizing the same underlying algorithm. If not — if tokens are purely positional/carry-based — it tells us SoRL converges to Quirke's circuits rather than Nanda's, which is itself a meaningful finding about which representation the training procedure favors.
