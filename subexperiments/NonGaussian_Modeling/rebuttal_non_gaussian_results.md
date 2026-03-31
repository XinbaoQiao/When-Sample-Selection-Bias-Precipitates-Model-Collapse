Non-Gaussian recursive-fitting results further support the same empirical pattern: biased selection consistently drives a sharp reduction in diversity, while `Accumulate` without selection remains substantially more stable. We report `Dispersion Ratio = \mathrm{Disp}(P_t) / \mathrm{Disp}(P_0)` as the main diversity-collapse metric, where smaller values indicate stronger collapse.

**Caption.** Figure 1. Structured Gaussian control (`anisotropic`, `n = 300`). Selection sharply reduces the dispersion ratio, while `Accumulate` remains close to the initial spread.

![Figure 1: anisotropic n=300](results_recursive_family_n100_n300_n500/n300/anisotropic/dispersion_collapse.png)

**Caption.** Figure 2. Balanced Gaussian mixture (`gmm`, `n = 300`). The non-Gaussian multimodal setting shows the same qualitative collapse pattern under biased selection.

![Figure 2: gmm n=300](results_recursive_family_n100_n300_n500/n300/gmm/dispersion_collapse.png)

**Caption.** Figure 3. Unequal-weight Gaussian mixture (`unequal_gmm`, `n = 300`). This setting additionally probes whether selection disproportionately removes a minority mode; the sharp drop in dispersion ratio is again clearly visible.

![Figure 3: unequal_gmm n=300](results_recursive_family_n100_n300_n500/n300/unequal_gmm/dispersion_collapse.png)

**Caption.** Figure 4. Heavy-tailed Laplace family (`laplace`, `n = 300`). Selection still produces an extreme diversity collapse, even though the unselected `Replace` trajectory is less stable than in the mixture settings.

![Figure 4: laplace n=300](results_recursive_family_n100_n300_n500/n300/laplace/dispersion_collapse.png)

**Caption.** Table 1. Final dispersion ratios at iteration `t = 200`. Lower values indicate stronger diversity collapse.

| Family | n | Accumulate | Accumulate + Select | Replace | Replace + Select |
| --- | ---: | ---: | ---: | ---: | ---: |
| anisotropic | 100 | 0.9894 | 0.0829 | 0.0556 | 0.0000 |
| anisotropic | 300 | 0.9646 | 0.0652 | 0.4221 | 0.0000 |
| anisotropic | 500 | 0.9736 | 0.0734 | 0.6692 | 0.0000 |
| gmm | 100 | 0.9455 | 0.0923 | 0.0258 | 0.0000 |
| gmm | 300 | 1.0162 | 0.0729 | 0.5798 | 0.0000 |
| gmm | 500 | 1.0034 | 0.0472 | 0.5604 | 0.0000 |
| unequal_gmm | 100 | 0.9699 | 0.0690 | 0.0790 | 0.0000 |
| unequal_gmm | 300 | 1.0263 | 0.1422 | 0.6449 | 0.0000 |
| unequal_gmm | 500 | 1.0010 | 0.0736 | 0.5615 | 0.0000 |
| laplace | 100 | 1.0439 | 0.0169 | 0.1336 | 0.0000 |
| laplace | 300 | 0.9803 | 0.0161 | 2.2282 | 0.0000 |
| laplace | 500 | 0.9990 | 0.0149 | 1.0510 | 0.0000 |

**Caption.** Table 2. Family-specific auxiliary statistics at iteration `t = 200`. For `unequal_gmm`, the smallest component weight decreases under selection, indicating erosion of the minority mode. For `laplace`, the mean absolute scale also collapses under selection.

| Family | n | Setting | Smallest Component Weight | Component Weight Entropy | Mean Absolute Scale |
| --- | ---: | --- | ---: | ---: | ---: |
| unequal_gmm | 100 | Accumulate | 0.2794 | 0.8496 | - |
| unequal_gmm | 100 | Accumulate + Select | 0.1648 | 0.5688 | - |
| unequal_gmm | 100 | Replace | 0.0000 | 0.0000 | - |
| unequal_gmm | 100 | Replace + Select | 0.0000 | 0.0000 | - |
| unequal_gmm | 300 | Accumulate | 0.2123 | 0.7436 | - |
| unequal_gmm | 300 | Accumulate + Select | 0.1692 | 0.5717 | - |
| unequal_gmm | 300 | Replace | 0.1087 | 0.2774 | - |
| unequal_gmm | 300 | Replace + Select | 0.0000 | 0.0000 | - |
| unequal_gmm | 500 | Accumulate | 0.2434 | 0.7997 | - |
| unequal_gmm | 500 | Accumulate + Select | 0.1591 | 0.5579 | - |
| unequal_gmm | 500 | Replace | 0.2052 | 0.5403 | - |
| unequal_gmm | 500 | Replace + Select | 0.0119 | 0.0653 | - |
| laplace | 100 | Accumulate | - | - | 2.0087 |
| laplace | 100 | Accumulate + Select | - | - | 0.2463 |
| laplace | 100 | Replace | - | - | 0.3788 |
| laplace | 100 | Replace + Select | - | - | 0.0007 |
| laplace | 300 | Accumulate | - | - | 1.9573 |
| laplace | 300 | Accumulate + Select | - | - | 0.2413 |
| laplace | 300 | Replace | - | - | 1.8039 |
| laplace | 300 | Replace + Select | - | - | 0.0006 |
| laplace | 500 | Accumulate | - | - | 1.9828 |
| laplace | 500 | Accumulate + Select | - | - | 0.2335 |
| laplace | 500 | Replace | - | - | 1.6169 |
| laplace | 500 | Replace + Select | - | - | 0.0006 |
