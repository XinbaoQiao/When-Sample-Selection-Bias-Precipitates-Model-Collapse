# Non-Gaussian Modeling

This directory contains recursive collapse experiments beyond the single isotropic Gaussian setting. The main figure uses `Dispersion Ratio` as the vertical axis:

\[
\mathrm{Dispersion\ Ratio} = \mathrm{Disp}(P_t) / \mathrm{Disp}(P_0),
\quad
\mathrm{Disp}(P) = \mathbb{E}\|X - X'\|^2.
\]

Interpretation:
- `Dispersion Ratio = 1` means the sample spread is comparable to the initial data distribution.
- A smaller `Dispersion Ratio` means lower diversity and stronger collapse.
- Values close to `0` indicate near-complete collapse.

Distribution families in this directory:

- `gmm`: A balanced two-component Gaussian mixture. This is the main mixture-of-Gaussians and multimodal setting.
- `unequal_gmm`: An unbalanced two-component Gaussian mixture with unequal weights. This is useful for testing whether selection disproportionately removes a minority mode.
- `laplace`: A coordinate-wise multivariate Laplace family. This is the main heavy-tailed setting.
- `anisotropic`: A full-covariance Gaussian control. This setting is not non-Gaussian, but it is a useful structured Gaussian baseline.

Suggested paper-facing grouping:

- Mixture of Gaussians: `gmm`, `unequal_gmm`
- Multimodal: `gmm`, `unequal_gmm`
- Heavy-tailed: `laplace`
- Structured Gaussian control: `anisotropic`

Examples:

```bash
cd subexperiments/nongaussin
python non_gaussian_experiment.py --distribution_type gmm
python non_gaussian_experiment.py --distribution_type unequal_gmm
python non_gaussian_experiment.py --distribution_type laplace
python non_gaussian_experiment.py --distribution_type anisotropic
python non_gaussian_experiment.py --distribution_type all
```

Running with `--distribution_type all` creates one subdirectory per family under `results_<timestamp>/`.
