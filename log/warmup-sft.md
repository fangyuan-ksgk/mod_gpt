# Warmup-SFT Evaluation (Qwen3-0.6B, GSM8K)


All the warmup training lead to horrible accuracy, but good dependency. (Acc[K=4] > Acc[K is None])
This indicates the SoRL search procedure is potentially unstable, as it doesn't build up reliable dependency. 
Moreover, SoRL is prune to vocabulary / embedding collapse, when explicitly regularized against vocabulary / embedding collapse, accuracy degrades. A few ideas can be explored to address this: 

1. 2 phase training, first phase builds up diversity, second phase build up dependency 
   (a). second phase can adopt pure distillation without re-initialization. 
   Rationale: we know pure SFT (without abstraction search) induces dependency, we know regularized SoRL gives diversity, the only question is how is the accuracy looking with the combo, and whether dependency can be built up.

2. Differentiable abstraction search, here we'd use STE trick to make choice of abstraction differentiable, so that we can directly optimize for the search process. (Not sure if this is possible)

3. Group relative preference optimization. If we ortho-init on abstract embedding, then we might be able to exploit their difference in predictability to utilize GRPO (without the explicit KL regularization). 




Setup: 2000 warmup SFT steps, bs=2, K=4, abs_vocab=128, eval=800 samples
Losses: abs=0.5, traj=1.0, jacobi=0.5 (m_traj varies by row)
No SoRL search — abstractions pinned via K-means clustering throughout.


tag                          K=4  K=None    gap    notes
-----------------------------------------------------------
baseline                    13.2%    9.4%  +3.9%   centroid init, m_traj=1.0
no_centroid_init            11.0%    6.6%  +4.4%   random init, m_traj=1.0
m_traj_0.0                   5.4%    5.4%  +0.0%   no masked-traj loss
m_traj_0.25                  4.6%    6.0%  -1.4%
m_traj_0.5                   7.9%    6.0%  +1.9%
m_traj_1.0                   9.1%    8.6%  +0.5%   = baseline (different run)
m_traj_2.0                  12.1%   10.4%  +1.8%

Key observations:
1. Accuracy is terrible across the board (best 13.2% vs pretrained ~40-50%).
   Warmup degrades base NL ability — the model shifts its representations
   to accommodate abstract tokens, losing pretrained NL prediction capability.
2. Dependency IS established: baseline gap +3.9%, no_centroid +4.4%.
   With stable (pinned) abstractions, model learns to benefit from them.
   This rules out "model routes around abstractions" as the root cause of
   SoRL's failure to build dependency — the real culprit is search instability.
3. Masked-traj loss (m_traj) matters: m_traj=0 gives no dependency (gap=0),
   higher m_traj strengthens both accuracy and dependency. m_traj forces the
   model to predict NL tokens from abstract context, directly building dependency.
4. Centroid init helps absolute accuracy (13.2% vs 11.0%) but gap is similar,
   suggesting init quality affects convergence speed, not dependency mechanism.

                                                                                                                                                                                                                                    