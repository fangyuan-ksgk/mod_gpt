# 02 — Understanding Addition and Subtraction in Transformers

**Authors:** Philip Quirke, Clement Neo, Fazl Barez  
**Venue:** arXiv preprint (2024)  
**arXiv:** https://arxiv.org/abs/2402.02619  
**PDF:** [02_quirke_understanding_addition_subtraction.pdf](02_quirke_understanding_addition_subtraction.pdf)

## Core Claim

Extends the single-operation analysis of paper 01 to **both addition and subtraction**, providing a unified mechanistic account via cascading carry and borrow circuits. Small transformers trained from scratch reach 99.999% accuracy; most large public LLMs fail basic arithmetic.

## Key Findings

- **Unified carry/borrow account** — addition and subtraction share the same positional-stream architecture; subtraction runs borrow cascades in place of carry cascades
- **Near-perfect accuracy possible** — tiny transformers (trained from scratch) solve n-digit addition and subtraction at 99.999%
- **49-model validation** — mechanisms confirmed via ablations and constraints across a sweep of trained models
- **LLM arithmetic gap** — surveying 180 public LLMs, only 7% reliably perform addition, showing specialized circuits don't emerge reliably in general training
- **Interpretability toolkit released** — subtask labels, ablation utilities, and circuit-tracing tools made public

## Relevance to This Study

Direct extension of paper 01 to subtraction — the subtask labels for subtraction (MD, MB, ME, UB, UD) that we use in `arithmetic/data/addition.py` come from this work. The borrow-cascade enrichment added to our data generation (forcing 40% equal digit positions to populate MB/UB splits) was motivated by the sparsity of these subtasks noted here. The "7% of LLMs pass" finding motivates why we train from scratch rather than fine-tuning.
