import argparse
import os
from dataclasses import dataclass
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.mixture import GaussianMixture
from tqdm import tqdm


def _use_plot_style():
    for style in ("seaborn-v0_8", "seaborn", "seaborn-whitegrid", "default"):
        try:
            plt.style.use(style)
            break
        except OSError:
            continue


@dataclass
class GaussianModel:
    mu: np.ndarray
    sigma: np.ndarray


@dataclass
class LaplaceModel:
    loc: np.ndarray
    scale: np.ndarray


class NonGaussianExperiment:
    def __init__(
        self,
        dim: int = 10,
        num_iterations: int = 200,
        samples_per_iteration: int = 300,
        selection_ratio: float = 0.05,
        distribution_type: str = "gmm",
        gmm_components: int = 2,
        gmm_n_init: int = 1,
        gmm_max_iter: int = 100,
        eval_sample_size: int = 4096,
        swd_num_projections: int = 64,
        random_seed: int = 42,
    ):
        self.dim = dim
        self.num_iterations = num_iterations
        self.n_train = samples_per_iteration
        self.alpha = selection_ratio
        self.n_generate = int(samples_per_iteration / selection_ratio)
        self.distribution_type = distribution_type
        self.gmm_components = gmm_components
        self.gmm_n_init = gmm_n_init
        self.gmm_max_iter = gmm_max_iter
        self.eval_sample_size = eval_sample_size
        self.swd_num_projections = swd_num_projections
        self.rng = np.random.default_rng(random_seed)

        self.mu_star = self.rng.normal(size=dim)
        A = self.rng.normal(size=(dim, dim))
        self.sigma_star = A @ A.T + np.eye(dim) * 0.1

        self.u_bias = self.mu_star.copy()
        bias_dims = self.rng.choice(dim, size=max(1, dim // 2), replace=False)
        self.u_bias[bias_dims] += self.rng.normal(size=len(bias_dims)) * 3.0
        self.sigma_bias = self.sigma_star * 0.3

        self.true_gmm = self._build_true_gmm(equal_weights=True)
        self.true_unequal_gmm = self._build_true_gmm(equal_weights=False)
        self.true_gaussian = GaussianModel(
            mu=self.mu_star.copy(),
            sigma=self.sigma_star.copy(),
        )
        self.true_laplace = self._build_true_laplace()

        self.truth_reference = self.sample_from_model(self.get_true_model(), self.eval_sample_size)
        self.initial_dispersion = self.compute_dispersion(self.truth_reference)

    def _build_true_gmm(self, equal_weights: bool) -> GaussianMixture:
        gmm = GaussianMixture(
            n_components=self.gmm_components,
            covariance_type="full",
            reg_covar=1e-6,
            random_state=0,
        )
        direction = self.rng.normal(size=self.dim)
        direction = direction / (np.linalg.norm(direction) + 1e-12)
        shift_magnitude = 0.9
        shifts = np.array([direction * shift_magnitude, -direction * shift_magnitude])
        cov_scales = np.array([0.7, 0.9])
        gmm.weights_ = np.array([0.5, 0.5]) if equal_weights else np.array([0.8, 0.2])
        gmm.means_ = np.vstack([self.mu_star + shifts[0], self.mu_star + shifts[1]])
        gmm.covariances_ = np.stack(
            [
                self.sigma_star * cov_scales[0] + np.eye(self.dim) * 1e-6,
                self.sigma_star * cov_scales[1] + np.eye(self.dim) * 1e-6,
            ]
        )
        gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(gmm.covariances_))
        return gmm

    def _build_true_laplace(self) -> LaplaceModel:
        sigma_diag = np.diag(self.sigma_star).copy()
        scale = np.sqrt(np.maximum(sigma_diag, 1e-8) / 2.0)
        return LaplaceModel(loc=self.mu_star.copy(), scale=scale)

    def get_true_model(self):
        if self.distribution_type == "gmm":
            return self.true_gmm
        if self.distribution_type == "unequal_gmm":
            return self.true_unequal_gmm
        if self.distribution_type == "anisotropic":
            return self.true_gaussian
        if self.distribution_type == "laplace":
            return self.true_laplace
        raise ValueError(f"Unsupported distribution_type: {self.distribution_type}")

    def sample_from_model(self, model, n: int) -> np.ndarray:
        if isinstance(model, GaussianMixture):
            component_ids = self.rng.choice(model.n_components, size=n, p=model.weights_)
            samples = np.empty((n, self.dim))
            for k in range(model.n_components):
                mask = component_ids == k
                count = int(np.sum(mask))
                if count == 0:
                    continue
                samples[mask] = self.rng.multivariate_normal(
                    model.means_[k],
                    model.covariances_[k],
                    size=count,
                )
            return samples
        if isinstance(model, GaussianModel):
            return self.rng.multivariate_normal(model.mu, model.sigma, size=n)
        if isinstance(model, LaplaceModel):
            return self.rng.laplace(loc=model.loc, scale=model.scale, size=(n, self.dim))
        raise TypeError(f"Unsupported model type: {type(model)}")

    def fit_model(self, samples: np.ndarray, previous_model=None):
        if self.distribution_type in {"gmm", "unequal_gmm"}:
            gmm = GaussianMixture(
                n_components=self.gmm_components,
                covariance_type="full",
                reg_covar=1e-6,
                n_init=self.gmm_n_init,
                max_iter=self.gmm_max_iter,
                random_state=0,
            )
            if isinstance(previous_model, GaussianMixture):
                gmm.weights_init = previous_model.weights_
                gmm.means_init = previous_model.means_
                gmm.precisions_init = np.linalg.inv(previous_model.covariances_)
            gmm.fit(samples)
            return gmm
        if self.distribution_type == "anisotropic":
            mu = np.mean(samples, axis=0)
            sigma = np.cov(samples.T, bias=True) + np.eye(self.dim) * 1e-6
            return GaussianModel(mu=mu, sigma=sigma)
        if self.distribution_type == "laplace":
            loc = np.median(samples, axis=0)
            scale = np.mean(np.abs(samples - loc), axis=0) + 1e-6
            return LaplaceModel(loc=loc, scale=scale)
        raise ValueError(f"Unsupported distribution_type: {self.distribution_type}")

    def select_samples(self, samples: np.ndarray) -> np.ndarray:
        diff = samples - self.u_bias
        sigma_inv = np.linalg.inv(self.sigma_bias)
        mahalanobis = np.sum(diff @ sigma_inv * diff, axis=1)
        scores = -mahalanobis
        top_indices = np.argsort(scores)[-self.n_train :]
        return samples[top_indices]

    def _model_mean_and_cov(self, model) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(model, GaussianMixture):
            mean = np.sum(model.weights_[:, None] * model.means_, axis=0)
            centered = model.means_ - mean
            cov = np.sum(
                model.weights_[:, None, None]
                * (model.covariances_ + centered[:, :, None] * centered[:, None, :]),
                axis=0,
            )
            return mean, cov
        if isinstance(model, GaussianModel):
            return model.mu, model.sigma
        if isinstance(model, LaplaceModel):
            return model.loc, np.diag(2.0 * np.square(model.scale))
        raise TypeError(f"Unsupported model type: {type(model)}")

    def compute_dispersion(self, samples: np.ndarray) -> float:
        cov = np.cov(samples.T, bias=True)
        return float(2.0 * np.trace(cov))

    def _sliced_wasserstein_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        distances = []
        for _ in range(self.swd_num_projections):
            direction = self.rng.normal(size=self.dim)
            direction = direction / (np.linalg.norm(direction) + 1e-12)
            x_proj = x @ direction
            y_proj = y @ direction
            distances.append(wasserstein_distance(x_proj, y_proj))
        return float(np.mean(distances))

    def evaluate_model(self, model, iteration: int, use_selection: bool, setting: str) -> dict:
        eval_samples = self.sample_from_model(model, self.eval_sample_size)
        mean, cov = self._model_mean_and_cov(model)
        dispersion = self.compute_dispersion(eval_samples)
        result = {
            "iteration": iteration,
            "use_selection": use_selection,
            "setting": setting,
            "mean_error": float(np.linalg.norm(mean - self.mu_star) ** 2),
            "variance": float(np.trace(cov)),
            "dispersion": dispersion,
            "dispersion_ratio": dispersion / self.initial_dispersion,
            "swd_to_truth": self._sliced_wasserstein_distance(eval_samples, self.truth_reference),
        }

        if isinstance(model, GaussianMixture):
            weights = np.clip(model.weights_, 1e-12, 1.0)
            entropy = -np.sum(weights * np.log(weights)) / np.log(len(weights))
            result["component_weight_entropy"] = float(entropy)
            result["smallest_component_weight"] = float(np.min(model.weights_))
            if len(model.means_) >= 2:
                result["component_mean_distance"] = float(np.linalg.norm(model.means_[0] - model.means_[1]))
        if isinstance(model, LaplaceModel):
            result["mean_abs_scale"] = float(np.mean(model.scale))

        return result

    def run_replace(self, use_selection: bool) -> pd.DataFrame:
        results = []
        model_t = self.get_true_model()
        results.append(self.evaluate_model(model_t, iteration=0, use_selection=use_selection, setting="Replace"))

        for t in tqdm(range(1, self.num_iterations + 1), desc=f"Replace-{self.distribution_type}-{use_selection}"):
            samples = self.sample_from_model(model_t, self.n_generate if use_selection else self.n_train)
            if use_selection:
                samples = self.select_samples(samples)
            model_t = self.fit_model(samples, previous_model=model_t)
            results.append(self.evaluate_model(model_t, iteration=t, use_selection=use_selection, setting="Replace"))

        return pd.DataFrame(results)

    def run_accumulate(self, use_selection: bool) -> pd.DataFrame:
        results = []
        model_t = self.get_true_model()
        accumulated = []
        results.append(self.evaluate_model(model_t, iteration=0, use_selection=use_selection, setting="Accumulate"))

        for t in tqdm(range(1, self.num_iterations + 1), desc=f"Accumulate-{self.distribution_type}-{use_selection}"):
            samples = self.sample_from_model(model_t, self.n_generate if use_selection else self.n_train)
            if use_selection:
                samples = self.select_samples(samples)
            accumulated.append(samples)
            all_samples = np.vstack(accumulated)
            model_t = self.fit_model(all_samples, previous_model=model_t)
            row = self.evaluate_model(model_t, iteration=t, use_selection=use_selection, setting="Accumulate")
            row["num_samples"] = len(all_samples)
            results.append(row)

        return pd.DataFrame(results)

    def run_all(self) -> pd.DataFrame:
        df1 = self.run_replace(use_selection=False)
        df1["setting"] = "Replace"
        df2 = self.run_replace(use_selection=True)
        df2["setting"] = "Replace + Select"
        df3 = self.run_accumulate(use_selection=False)
        df3["setting"] = "Accumulate"
        df4 = self.run_accumulate(use_selection=True)
        df4["setting"] = "Accumulate + Select"
        return pd.concat([df1, df2, df3, df4], ignore_index=True)


def plot_main_metric(df: pd.DataFrame, save_dir: str, num_runs: int, metric: str = "dispersion_ratio"):
    os.makedirs(save_dir, exist_ok=True)
    _use_plot_style()
    stats_list = []
    for setting in ["Replace", "Accumulate", "Replace + Select", "Accumulate + Select"]:
        setting_data = df[df["setting"] == setting]
        for iteration in setting_data["iteration"].unique():
            iter_data = setting_data[setting_data["iteration"] == iteration]
            values = []
            for run_id in iter_data["run"].unique():
                run_data = iter_data[iter_data["run"] == run_id]
                if len(run_data) > 0:
                    values.append(run_data[metric].iloc[0])
            values = np.array(values)
            stats_list.append(
                {
                    "setting": setting,
                    "iteration": iteration,
                    "mean": values.mean(),
                    "std": values.std(),
                    "sem": values.std() / np.sqrt(num_runs),
                }
            )
    stats_df = pd.DataFrame(stats_list)

    import matplotlib as mpl
    import matplotlib.font_manager as fm
    from matplotlib.lines import Line2D

    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = ["DejaVu Serif", "Computer Modern", "Times New Roman"]
    mpl.rcParams["mathtext.fontset"] = "cm"
    mpl.rcParams["font.size"] = 21
    mpl.rcParams["axes.labelsize"] = 24
    mpl.rcParams["axes.titlesize"] = 27
    mpl.rcParams["legend.fontsize"] = 20
    mpl.rcParams["xtick.labelsize"] = 21
    mpl.rcParams["ytick.labelsize"] = 21
    mpl.rcParams["axes.linewidth"] = 2.0
    mpl.rcParams["grid.alpha"] = 0.3

    fig, ax = plt.subplots(1, 1, figsize=(9, 8))
    fig.patch.set_facecolor("white")

    style_map = {
        "Replace": {"color": "#0005E6", "linestyle": "-", "marker": "o", "linewidth": 3.0, "alpha": 0.9, "fill_alpha": 0.2},
        "Replace + Select": {"color": "#000000", "linestyle": "-", "marker": "o", "linewidth": 3.5, "alpha": 0.9, "fill_alpha": 0.2},
        "Accumulate": {"color": "#800080", "linestyle": "-", "marker": "o", "linewidth": 3.0, "alpha": 0.9, "fill_alpha": 0.2},
        "Accumulate + Select": {"color": "#F60507", "linestyle": "-", "marker": "o", "linewidth": 3.5, "alpha": 0.9, "fill_alpha": 0.2},
    }

    name_map = {
        "Replace": "Replace",
        "Replace + Select": "Replace & Selection",
        "Accumulate": "Accumulate",
        "Accumulate + Select": "Accumulate & Selection",
    }

    legend_handles = []
    legend_labels = []
    for setting in ["Replace", "Accumulate", "Replace + Select", "Accumulate + Select"]:
        data = stats_df[stats_df["setting"] == setting]
        style = style_map[setting]
        display_name = name_map[setting]
        ax.plot(
            data["iteration"],
            data["mean"],
            label=display_name,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            alpha=style["alpha"],
            zorder=10,
        )
        legend_proxy = Line2D(
            [0],
            [0],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            label=display_name,
        )
        if style["marker"] is not None:
            legend_proxy.set_marker(style["marker"])
            legend_proxy.set_markersize(8)
            legend_proxy.set_markeredgewidth(2.0)
            legend_proxy.set_markeredgecolor("white")
        legend_handles.append(legend_proxy)
        legend_labels.append(display_name)
        ax.fill_between(
            data["iteration"],
            data["mean"] - data["sem"],
            data["mean"] + data["sem"],
            color=style["color"],
            alpha=style["fill_alpha"],
            linewidth=0,
            zorder=5,
        )

    ax.set_xlabel(r"Iteration $t$", fontsize=30, fontweight="bold")
    ax.set_ylabel("Dispersion Ratio", fontsize=28, fontweight="bold")

    paradigm_indices = [i for i, label in enumerate(legend_labels) if "Selection" not in label]
    selection_indices = [i for i, label in enumerate(legend_labels) if "Selection" in label]
    final_handles = [legend_handles[i] for i in paradigm_indices] + [legend_handles[i] for i in selection_indices]
    final_labels = [legend_labels[i] for i in paradigm_indices] + [legend_labels[i] for i in selection_indices]

    legend_kwargs = {
        "ncol": 2,
        "frameon": True,
        "fontsize": 23,
        "prop": fm.FontProperties(weight="bold"),
        "handlelength": 1.2,
        "handletextpad": 0.5,
        "columnspacing": 1.5,
        "edgecolor": "#B0B0B0",
    }

    ax.set_ylim(bottom=-0.02)
    ax.axhline(y=1.0, color="black", linestyle=":", linewidth=2.5, alpha=0.6, zorder=0)
    ax.grid(True, alpha=0.2, linestyle="--", linewidth=1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout(pad=2.0, rect=[0, 0.15, 1, 1])
    if final_handles:
        legend_pad = 0.07
        legend = fig.legend(
            final_handles,
            final_labels,
            loc="lower center",
            bbox_to_anchor=(legend_pad, 0.085, 1.0 - 2 * legend_pad, 0.0),
            mode="expand",
            **legend_kwargs,
        )
        legend.get_frame().set_linewidth(0.8)
        legend.get_frame().set_facecolor("white")
        legend.set_zorder(5)

    plt.savefig(os.path.join(save_dir, "dispersion_collapse.png"), dpi=800, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.savefig(os.path.join(save_dir, "dispersion_collapse.pdf"), bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    print(f"\nSaved dispersion plot to: {save_dir}")


def export_metric_tables(df: pd.DataFrame, save_dir: str):
    summary = (
        df.groupby(["setting", "iteration"])
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.to_csv(os.path.join(save_dir, "metric_summary_by_iteration.csv"), index=False)

    final_iter = int(df["iteration"].max())
    final_metrics = (
        df[df["iteration"] == final_iter]
        .groupby("setting")
        .agg(["mean", "std"])
        .reset_index()
    )
    final_metrics.to_csv(os.path.join(save_dir, "final_iteration_summary.csv"), index=False)


def run_one_distribution(args, distribution_type: str, save_dir: str):
    random_seeds = [42, 123, 456, 789, 2024, 2025, 3141, 1618, 9999, 1103][: args.num_runs]
    config = {
        "dim": args.dim,
        "num_iterations": args.num_iterations,
        "samples_per_iteration": args.samples_per_iteration,
        "selection_ratio": args.selection_ratio,
        "distribution_type": distribution_type,
        "gmm_components": args.gmm_components,
        "gmm_n_init": args.gmm_n_init,
        "gmm_max_iter": args.gmm_max_iter,
        "eval_sample_size": args.eval_sample_size,
        "swd_num_projections": args.swd_num_projections,
    }

    all_results = []
    for i, seed in enumerate(random_seeds, 1):
        exp = NonGaussianExperiment(**config, random_seed=seed)
        result_df = exp.run_all()
        result_df["run"] = i
        result_df["seed"] = seed
        all_results.append(result_df)

    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df.to_csv(os.path.join(save_dir, "all_runs.csv"), index=False)
    plot_main_metric(combined_df, save_dir, args.num_runs, metric="dispersion_ratio")
    export_metric_tables(combined_df, save_dir)

    final_iter = combined_df["iteration"].max()
    print(f"distribution={distribution_type}, final_iter={final_iter}")
    columns = ["dispersion_ratio", "swd_to_truth", "mean_error", "variance"]
    extra_cols = [
        col
        for col in [
            "component_weight_entropy",
            "smallest_component_weight",
            "component_mean_distance",
            "mean_abs_scale",
        ]
        if col in combined_df.columns
    ]
    report_cols = columns + extra_cols
    summary = combined_df[combined_df["iteration"] == final_iter].groupby("setting")[report_cols].mean().round(4)
    print(summary.to_string())
    print(f"saved_to={save_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--distribution_type",
        type=str,
        default="gmm",
        choices=["gmm", "unequal_gmm", "laplace", "anisotropic", "all"],
    )
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--num_iterations", type=int, default=200)
    parser.add_argument("--samples_per_iteration", type=int, default=300)
    parser.add_argument("--selection_ratio", type=float, default=0.05)
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--gmm_components", type=int, default=2)
    parser.add_argument("--gmm_n_init", type=int, default=1)
    parser.add_argument("--gmm_max_iter", type=int, default=100)
    parser.add_argument("--eval_sample_size", type=int, default=4096)
    parser.add_argument("--swd_num_projections", type=int, default=64)
    parser.add_argument("--output_root", type=str, default=None)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.distribution_type == "all":
        root_dir = args.output_root if args.output_root else os.path.join(script_dir, f"results_{timestamp}")
        os.makedirs(root_dir, exist_ok=True)
        for dist in ["gmm", "unequal_gmm", "laplace", "anisotropic"]:
            dist_dir = os.path.join(root_dir, dist)
            os.makedirs(dist_dir, exist_ok=True)
            run_one_distribution(args, dist, dist_dir)
    else:
        save_dir = args.output_root if args.output_root else os.path.join(script_dir, f"results_{args.distribution_type}_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)
        run_one_distribution(args, args.distribution_type, save_dir)


if __name__ == "__main__":
    main()
