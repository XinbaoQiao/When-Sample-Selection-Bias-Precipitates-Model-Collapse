# Reviewer h5Ke

## 1. Non-Gaussian Empirical Evidence

[Non-Gaussian empirical evidence](subexperiments/NonGaussian_Modeling/rebuttal_non_gaussian_results.md)

## 2. LLM-related Experiments

**Caption.** Figure 1. An illustrative topic-generalization trend on held-out non-tech topics after recursive training with a tech-local verifier. The `Random` curve decreases mildly with small fluctuations, while the `ROUGE` curve drops early and then stabilizes at a lower level.

<p align="center">
  <img src="rouge_random_vs_iteration.png" alt="Figure 1: topic-generalization illustration" width="640">
</p>

**Caption.** Table 1. Illustrative language-generalization result. Selection based on an English-local verifier can improve in-domain alignment while degrading broader language coverage.

The table reports ROUGE-1 on Welsh after recursive training with an English-local verifier.

| Setting | Welsh ROUGE-1 |
| --- | ---: |
| Random | 0.257 |
| ROUGE | 0.223 |
