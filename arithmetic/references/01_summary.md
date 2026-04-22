# 01 — Understanding Addition in Transformers

**Authors:** Philip Quirke, Fazl Barez  
**Venue:** ICLR 2024  
**arXiv:** https://arxiv.org/abs/2310.13121  
**PDF:** [01_quirke_understanding_addition.pdf](01_quirke_understanding_addition.pdf)

## Core Claim

A one-layer transformer trained on n-digit integer addition decomposes the task into **parallel digit-position streams**, each running a distinct algorithm. The model does not treat addition as a single monolithic computation.

## Key Findings

- **Parallel streams per digit position** — each output digit is computed largely independently using position-specific circuits
- **Position-specific algorithms** — units digit, carry digit, and higher positions use different strategies; carry propagation is handled separately
- **Subtask taxonomy** — introduces labels (SA, SC, SS, UC, US for addition; MD, MB, ME, UB, UD for subtraction) that classify each digit computation by its carry/borrow context
- **Rare high-loss cases explained** — edge cases (e.g. sum-to-9 digits) require a specialized "sum-9" circuit; its failure explains most error modes
- **Methodology generalizes** — the activation-patching + subtask decomposition approach extends to subtraction and multi-layer models

## Relevance to This Study

This paper is the **primary reference** for the arithmetic interpretability study. We:
- Adopt the Quirke subtask labels (SA/SC/SS/UC/US, MD/MB/ME/UB/UD) directly in `arithmetic/data/addition.py`
- Use the same eval split structure (splits by subtask complexity)
- Extend the findings to SoRL models: asking whether abstract tokens externalize the carry/borrow circuits that Quirke found inside activations
- Our finding 4 (cross-operation unification) and finding 6 (guided computation) are direct extensions of Quirke's §4 activation patching
