## Goal
We want to mirror the study on addition in transformers. https://arxiv.org/abs/2310.13121

**Hypothesis:**  
The internal mechanisms and steps of the transformer can be deduced from internal activations. We show that SoRL externalizes these reasoning capabilities, allowing for more targeted insight into model behavior.

---

## Experiments

### 1. Train Models
- 3-layer transformer on 6-digit addition / subtraction  
- Baseline: standard SFT  
- SoRL: same model + abstraction tokens  
- Sweep abstraction vocab: {5, 10, 15}  
- Track:
  - Loss  
  - Accuracy  
  - Vocabulary utilization  

---

### 2. Label Arithmetic Structure
Tag each example with:
- Addition / Subtraction  
- Has_Carry_Add  
- Has_Borrow  
- Has_Cascade_Carry  
- Has_Cascade_Borrow  

---

### 3. Identify Reasoning Units
- **SoRL:** correlate abstraction tokens with labeled data cases  
- **Baseline:** extract SAE / activation features correlated with the same labels  

---

### 4. External vs Internal Mechanisms
Show correspondence:
- Baseline reasoning step → latent activation / SAE feature  
- SoRL reasoning step → explicit abstraction token  

---

### 5. Paired Interventions
For each reasoning step:
- **Baseline:** activation patching / feature ablation  
- **SoRL:** token swap / deletion / corruption  

Measure:
- Error increase on the corresponding class  

---

### 6. Feature–Token Mapping
- Match SoRL tokens to baseline SAE features  
  - e.g., Hungarian matching or correlation over labeled subsets  

---

### 7. Redundancy / Polysemanticity
Check whether:
- Multiple tokens map to the same step (redundancy)  
- One token maps to multiple steps (polysemanticity)  

---

### 8. Auto-Interpretability
- Collect top-k strongest SoRL-token usages (logit margin)  
- Run auto-interpretability on these examples  
- Compare with SAE feature interpretations  

*(Auto-interpretability: using a model like GPT-5 to explain what role a feature plays across examples.)*

---

### 9. Probing
- **Baseline:** probe residual stream for correctness / sub-step features  
- **SoRL:** probe representation of the final SoRL token before answer generation  

Compare:
- Whether correctness and sub-step information are more linearly accessible in SoRL representations  

---

## Final Goal
Show that arithmetic sub-mechanisms that appear as latent activation features in the baseline instead appear as explicit tokens in SoRL, making them directly observable and intervenable without activation-level tooling.