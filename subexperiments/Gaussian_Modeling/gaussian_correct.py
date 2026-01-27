"""
Gaussian Model Collapse Experiment - Correct Experimental Design

4 Settings (Consistent Data Volume):
1. Replace: Generate N samples per generation, completely replace previous data.
2. Accumulate: Generate N samples per generation, accumulate all previous data.
3. Replace + Select: Generate N/alpha samples per generation, select N samples, replace previous data.
4. Accumulate + Select: Generate N/alpha samples per generation, select N samples, accumulate.

Key: The data volume after selection is consistent with the non-selection case!
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from tqdm import tqdm
import os
from datetime import datetime
import pandas as pd
from matplotlib.patches import Rectangle


def _use_plot_style():
    for style in ("seaborn-v0_8", "seaborn", "seaborn-whitegrid", "default"):
        try:
            plt.style.use(style)
            break
        except OSError:
            continue


class GaussianExperiment:
    """
    Correct Gaussian Model Collapse Experiment
    Maintains consistent data volume to compare the effect of selection vs no selection.
    """
    
    def __init__(
        self,
        dim: int = 10,
        num_iterations: int = 100,
        samples_per_iteration: int = 100,  # Number of samples used for training per generation
        selection_ratio: float = 0.1,      # Selection ratio
        random_seed: int = 42
    ):
        self.dim = dim
        self.num_iterations = num_iterations
        self.n_train = samples_per_iteration  # Samples for training per generation
        self.alpha = selection_ratio
        self.n_generate = int(samples_per_iteration / selection_ratio)  # Number of samples to generate
        
        np.random.seed(random_seed)
        
        # True Distribution N(mu*, Sigma*)
        self.mu_star = np.random.randn(dim)
        A = np.random.randn(dim, dim)
        self.sigma_star = A @ A.T + np.eye(dim) * 0.1
        
        # Biased Target u_bias (for selection)
        self.u_bias = self.mu_star.copy()
        bias_dims = np.random.choice(dim, size=dim//2, replace=False)
        self.u_bias[bias_dims] += np.random.randn(len(bias_dims)) * 3.0
        
        # Covariance for selection (smaller, simulating local observation)
        self.sigma_bias = self.sigma_star * 0.3
    
    def sample_gaussian(self, mu: np.ndarray, sigma: np.ndarray, n: int) -> np.ndarray:
        """Sample from Gaussian distribution"""
        return np.random.multivariate_normal(mu, sigma, size=n)
    
    def fit_gaussian(self, samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """MLE Estimation"""
        mu = np.mean(samples, axis=0)
        sigma = np.cov(samples.T, bias=True)  # Use MLE (divide by N) instead of unbiased (N-1)
        sigma = sigma + np.eye(self.dim) * 1e-6  # Regularization
        return mu, sigma
    
    def select_samples(self, samples: np.ndarray) -> np.ndarray:
        """
        Top-alpha selection towards u_bias
        Score: U(x) = -||x - u_bias||^2_{Sigma_bias^(-1)}
        """
        diff = samples - self.u_bias
        sigma_inv = np.linalg.inv(self.sigma_bias)
        mahalanobis = np.sum(diff @ sigma_inv * diff, axis=1)
        scores = -mahalanobis  # Negative distance as score
        
        # Select top-alpha
        top_indices = np.argsort(scores)[-self.n_train:]
        return samples[top_indices]
    
    def run_replace(self, use_selection: bool) -> pd.DataFrame:
        """
        Replace Strategy
        
        Args:
            use_selection: True=Generate more then select, False=Generate n_train directly
        """
        results = []
        
        # Initialize
        mu_t = self.mu_star.copy()
        sigma_t = self.sigma_star.copy()
        
        # Generation 0
        results.append({
            'iteration': 0,
            'mean_error': np.linalg.norm(mu_t - self.mu_star) ** 2,
            'variance': np.trace(sigma_t),
            'use_selection': use_selection
        })
        
        # Iterate
        desc = "Replace + Select" if use_selection else "Replace"
        for t in tqdm(range(1, self.num_iterations + 1), desc=desc):
            if use_selection:
                # Generate more, then select
                samples = self.sample_gaussian(mu_t, sigma_t, self.n_generate)
                samples = self.select_samples(samples)  # Select n_train
            else:
                # Generate n_train directly
                samples = self.sample_gaussian(mu_t, sigma_t, self.n_train)
            
            # Refit (using only current batch)
            mu_t, sigma_t = self.fit_gaussian(samples)
            
            # Record
            results.append({
                'iteration': t,
                'mean_error': np.linalg.norm(mu_t - self.mu_star) ** 2,
                'variance': np.trace(sigma_t),
                'use_selection': use_selection
            })
        
        return pd.DataFrame(results)
    
    def run_accumulate(self, use_selection: bool) -> pd.DataFrame:
        """
        Accumulate Strategy
        
        Args:
            use_selection: True=Generate more then select per generation, False=Generate n_train directly
        """
        results = []
        
        # Initialize
        mu_t = self.mu_star.copy()
        sigma_t = self.sigma_star.copy()
        accumulated = []
        
        # Generation 0
        results.append({
            'iteration': 0,
            'mean_error': np.linalg.norm(mu_t - self.mu_star) ** 2,
            'variance': np.trace(sigma_t),
            'use_selection': use_selection
        })
        
        # Iterate
        desc = "Accumulate + Select" if use_selection else "Accumulate"
        for t in tqdm(range(1, self.num_iterations + 1), desc=desc):
            if use_selection:
                # Generate more, then select
                samples = self.sample_gaussian(mu_t, sigma_t, self.n_generate)
                samples = self.select_samples(samples)  # Select n_train
            else:
                # Generate n_train directly
                samples = self.sample_gaussian(mu_t, sigma_t, self.n_train)
            
            # Accumulate
            accumulated.append(samples)
            all_samples = np.vstack(accumulated)
            
            # Refit (using all accumulated samples)
            mu_t, sigma_t = self.fit_gaussian(all_samples)
            
            # Record
            results.append({
                'iteration': t,
                'mean_error': np.linalg.norm(mu_t - self.mu_star) ** 2,
                'variance': np.trace(sigma_t),
                'use_selection': use_selection,
                'num_samples': len(all_samples)
            })
        
        return pd.DataFrame(results)
    
    def run_all(self) -> pd.DataFrame:
        """Run all 4 settings"""
        print("=" * 70)
        print("Running Gaussian Model Collapse Experiments")
        print("=" * 70)
        # print(f"\nConfiguration:")
        print(f"  Dimension: {self.dim}")
        print(f"  Iterations: {self.num_iterations}")
        print(f"  Samples per iteration: {self.n_train}")
        print(f"  Selection ratio: {self.alpha}")
        print(f"  Samples to generate (with selection): {self.n_generate}")
        print()
        
        # 1. Replace
        df1 = self.run_replace(use_selection=False)
        df1['setting'] = 'Replace'
        
        # 2. Replace + Select
        df2 = self.run_replace(use_selection=True)
        df2['setting'] = 'Replace + Select'
        
        # 3. Accumulate
        df3 = self.run_accumulate(use_selection=False)
        df3['setting'] = 'Accumulate'
        
        # 4. Accumulate + Select
        df4 = self.run_accumulate(use_selection=True)
        df4['setting'] = 'Accumulate + Select'
        
        # Combine
        return pd.concat([df1, df2, df3, df4], ignore_index=True)


def plot_results_with_error_bars(df: pd.DataFrame, save_dir: str, num_runs: int):
    """Beautified plot - Only shows variance collapse with confidence intervals"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Calculate mean and SEM for each setting (using respective initial variances)
    stats_list = []
    for setting in ['Replace', 'Accumulate', 'Replace + Select', 'Accumulate + Select']:
        setting_data = df[df['setting'] == setting]
        for iteration in setting_data['iteration'].unique():
            iter_data = setting_data[setting_data['iteration'] == iteration]
            
            # Calculate variance ratio for each run
            variance_ratios = []
            for run_id in iter_data['run'].unique():
                run_data = iter_data[iter_data['run'] == run_id]
                if len(run_data) > 0:
                    # Get initial variance for this run
                    initial_var = df[(df['setting'] == setting) & 
                                    (df['run'] == run_id) & 
                                    (df['iteration'] == 0)]['variance'].iloc[0]
                    current_var = run_data['variance'].iloc[0]
                    variance_ratios.append(current_var / initial_var)
            
            variance_ratios = np.array(variance_ratios)
            
            stats_list.append({
                'setting': setting,
                'iteration': iteration,
                'mean': variance_ratios.mean(),
                'std': variance_ratios.std(),
                'sem': variance_ratios.std() / np.sqrt(num_runs)  # Standard Error of Mean
            })
    
    stats_df = pd.DataFrame(stats_list)
    
    # Professional style settings
    import matplotlib as mpl
    _use_plot_style()
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['DejaVu Serif', 'Computer Modern', 'Times New Roman']
    # ICML style math font
    mpl.rcParams['mathtext.fontset'] = 'cm'
    mpl.rcParams['font.size'] = 21
    mpl.rcParams['axes.labelsize'] = 24
    mpl.rcParams['axes.titlesize'] = 27
    mpl.rcParams['legend.fontsize'] = 20
    mpl.rcParams['xtick.labelsize'] = 21
    mpl.rcParams['ytick.labelsize'] = 21
    mpl.rcParams['axes.linewidth'] = 2.0
    mpl.rcParams['grid.alpha'] = 0.3
    
    # Create single plot (Variance only)
    fig, ax = plt.subplots(1, 1, figsize=(9, 8))
    fig.patch.set_facecolor('white')
    
    # Color scheme (User specified)
    style_map = {
        'Replace': {
            'color': '#0005E6',  # Blue
            'linestyle': '-', 
            'marker': 'o',
            'linewidth': 3.0,
            'alpha': 0.9,
            'fill_alpha': 0.2
        },
        'Replace + Select': {
            'color': '#000000',  # Purple/Black
            'linestyle': '-',
            'marker': 'o',
            'linewidth': 3.5,
            'alpha': 0.9,
            'fill_alpha': 0.2
        },
        'Accumulate': {
            'color': '#800080',  # Darker Orange/Purple
            'linestyle': '-', 
            'marker': 'o',
            'linewidth': 3.0,
            'alpha': 0.9,
            'fill_alpha': 0.2
        },
        'Accumulate + Select': {
            'color': '#F60507',  # Red
            'linestyle': '-',
            'marker': 'o',
            'linewidth': 3.5,
            'alpha': 0.9,
            'fill_alpha': 0.2
        },
    }
    
    # Name mapping
    name_map = {
        'Replace': 'Replace',
        'Replace + Select': 'Replace & Selection',
        'Accumulate': 'Accumulate',
        'Accumulate + Select': 'Accumulate & Selection'
    }
    
    # Manually collect legend handles and labels
    legend_handles = []
    legend_labels = []

    # === Plot curves with error bars ===
    from matplotlib.lines import Line2D
    for setting in ['Replace', 'Accumulate', 'Replace + Select', 'Accumulate + Select']:
        data = stats_df[stats_df['setting'] == setting]
        style = style_map[setting]
        display_name = name_map[setting]
        
        # Main curve (no marker)
        plot_kwargs = {
            'label': display_name,
            'color': style['color'],
            'linestyle': style['linestyle'],
            'linewidth': style['linewidth'],
            'alpha': style['alpha'],
            'zorder': 10
        }
        # Remove marker settings to keep line smooth
        ax.plot(data['iteration'], data['mean'], **plot_kwargs)
        
        # Create proxy handle for legend (with marker)
        legend_proxy = Line2D([0], [0], 
                            color=style['color'], 
                            linestyle=style['linestyle'],
                            linewidth=style['linewidth'],
                            label=display_name)
        
        if style['marker'] is not None:
            legend_proxy.set_marker(style['marker'])
            legend_proxy.set_markersize(8)
            legend_proxy.set_markeredgewidth(2.0)
            legend_proxy.set_markeredgecolor('white')
            
        legend_handles.append(legend_proxy)
        legend_labels.append(display_name)
        
        # Confidence interval (Mean +/- SEM)
        ax.fill_between(data['iteration'], 
                        data['mean'] - data['sem'], 
                        data['mean'] + data['sem'],
                        color=style['color'],
                        alpha=style['fill_alpha'],
                        linewidth=0,
                        zorder=5)
    
    ax.set_xlabel(r'Iteration $t$', fontsize=30, fontweight='bold')
    ax.set_ylabel(r'$\mathbf{Tr}(\mathbf{Sigma}_t) / \mathbf{Tr}(\mathbf{Sigma}_0)$', 
                  fontsize=30, fontweight='bold')
    # ax.set_title(f'Variance', 
    #              fontsize=30, fontweight='bold', pad=20)
    
    # Grouping: Based on whether name contains 'Selection'
    paradigm_indices = [i for i, l in enumerate(legend_labels) if 'Selection' not in l]
    selection_indices = [i for i, l in enumerate(legend_labels) if 'Selection' in l]
    
    # Combine handles and labels order
    final_handles = [legend_handles[i] for i in paradigm_indices] + [legend_handles[i] for i in selection_indices]
    final_labels = [legend_labels[i] for i in paradigm_indices] + [legend_labels[i] for i in selection_indices]

    # Unified legend parameters
    import matplotlib.font_manager as fm
    legend_kwargs = {
        'ncol': 2,
        'frameon': True, 
        'fontsize': 23,
        'prop': fm.FontProperties(weight='bold'),
        'handlelength': 1.2,
        'handletextpad': 0.5,
        'columnspacing': 1.5,
        'edgecolor': '#B0B0B0',
    }
    
    # Set Y-axis limits
    # ax.set_ylim(0.01, 1.0)
    ax.set_ylim(-0.02, 1.3)
    
    # Linear scale
    ax.set_yscale('linear')
    
    # Reference line (True Variance Ratio = 1)
    ax.axhline(y=1.0, color='black', linestyle=':', linewidth=2.5, alpha=0.6, 
               label='True Variance', zorder=0)
    
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=1.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Adjust layout
    # User request: Legend move down. We reserve more space at the bottom.
    plt.tight_layout(pad=2.0, rect=[0, 0.15, 1, 1])

    if final_handles:
        legend_pad = 0.07
        legend = fig.legend(
            final_handles,
            final_labels,
            loc='lower center',
            # Move legend slightly down
            bbox_to_anchor=(legend_pad, 0.085, 1.0 - 2 * legend_pad, 0.0),
            mode='expand',
            **legend_kwargs,
        )
        legend.get_frame().set_linewidth(0.8)
        legend.get_frame().set_facecolor('white')
        legend.set_zorder(5)
    
    # Save high-quality images
    plt.savefig(os.path.join(save_dir, 'variance_collapse.png'), dpi=800, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig(os.path.join(save_dir, 'variance_collapse.pdf'), bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"\n✓ High-quality variance collapse plot with confidence intervals saved to: {save_dir}")


def plot_results(df: pd.DataFrame, save_dir: str):
    """Beautified plot - Variance collapse only (Single run version)"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Initial variance
    initial_variance = df[df['iteration'] == 0]['variance'].iloc[0]
    
    # Style settings
    import matplotlib as mpl
    _use_plot_style()
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['DejaVu Serif', 'Computer Modern', 'Times New Roman']
    mpl.rcParams['mathtext.fontset'] = 'cm'
    mpl.rcParams['font.size'] = 24
    mpl.rcParams['axes.labelsize'] = 27
    mpl.rcParams['axes.titlesize'] = 30
    mpl.rcParams['legend.fontsize'] = 23
    mpl.rcParams['xtick.labelsize'] = 24
    mpl.rcParams['ytick.labelsize'] = 24
    mpl.rcParams['axes.linewidth'] = 2.0
    mpl.rcParams['grid.alpha'] = 0.3
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 10))
    fig.patch.set_facecolor('white')
    
    # Color scheme
    style_map = {
        'Replace': {
            'color': '#1f77b4',  # Blue
            'linestyle': '-', 
            'marker': 'o',
            'linewidth': 3.0,
            'alpha': 0.9
        },
        'Replace + Select': {
            'color': '#d62728',  # Red
            'linestyle': '--', 
            'marker': 's',
            'linewidth': 3.5,
            'alpha': 0.9
        },
        'Accumulate': {
            'color': '#2ca02c',  # Green
            'linestyle': '-', 
            'marker': '^',
            'linewidth': 3.0,
            'alpha': 0.9
        },
        'Accumulate + Select': {
            'color': '#ff7f0e',  # Orange
            'linestyle': '--', 
            'marker': 'D',
            'linewidth': 3.5,
            'alpha': 0.9
        },
    }
    
    # === Plot Variance Collapse ===
    ax.set_facecolor('#f8f9fa')
    
    # Name map
    name_map = {
        'Replace': 'Replace',
        'Replace + Select': 'Replace & Selection',
        'Accumulate': 'Accumulate',
        'Accumulate + Select': 'Accumulate & Selection'
    }

    from matplotlib.lines import Line2D
    legend_handles = []
    
    for setting in ['Replace', 'Accumulate', 'Replace + Select', 'Accumulate + Select']:
        data = df[df['setting'] == setting]
        variance_ratio = data['variance'] / initial_variance
        style = style_map[setting]
        display_name = name_map[setting]
        
        # Remove marker from line
        ax.plot(data['iteration'], variance_ratio, 
                label=display_name,
                color=style['color'],
                linestyle=style['linestyle'],
                linewidth=style['linewidth'],
                alpha=style['alpha'])
                
        # Create legend proxy with marker
        legend_proxy = Line2D([0], [0], 
                            color=style['color'], 
                            linestyle=style['linestyle'],
                            linewidth=style['linewidth'],
                            label=display_name)
        
        if style['marker'] is not None:
            legend_proxy.set_marker(style['marker'])
            legend_proxy.set_markersize(8)
            legend_proxy.set_markeredgewidth(2.0)
            legend_proxy.set_markeredgecolor('white')
            
        legend_handles.append(legend_proxy)
    
    ax.set_xlabel(r'Iteration $\mathbf{t}$', fontsize=36, fontweight='bold')
    ax.set_ylabel(r'Variance Ratio $\mathbf{Tr}(\mathbf{\hat{Sigma}}_t) / \mathbf{Tr}(\mathbf{Sigma}^*)$', 
                  fontsize=36, fontweight='bold')
    ax.set_title('Variance Collapse', 
                 fontsize=40, pad=20)
    
    # Legend
    import matplotlib.font_manager as fm
    legend = ax.legend(
        handles=legend_handles,
        fontsize=20,
        prop=fm.FontProperties(weight='bold'),
        loc='upper right',
        frameon=True,
        fancybox=True,
        shadow=True,
        framealpha=0.95,
        edgecolor='gray',
        borderpad=1.2
    )
    legend.get_frame().set_facecolor('white')
    
    # Y-axis
    ax.set_ylim(-0.02, 1.3)
    ax.set_yscale('linear')
    
    # Reference line
    ax.axhline(y=1.0, color='black', linestyle=':', linewidth=2.5, alpha=0.6, 
               label='True Variance', zorder=0)
    
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=1.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout(pad=2.0)
    
    # Save
    plt.savefig(os.path.join(save_dir, 'variance_collapse.png'), dpi=800, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig(os.path.join(save_dir, 'variance_collapse.pdf'), dpi=800, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"\n✓ High-quality variance collapse plot saved to: {save_dir}")


def main():
    """Main Function - Run multiple experiments and compute confidence intervals"""
    print("\n" + "=" * 70)
    print("Gaussian Model Collapse - Multiple Runs with Error Bars")
    print("=" * 70)
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, f"results_correct_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Configuration: Correct parameters (satisfying n >> d)
    config = {
        'dim': 10,
        'num_iterations': 300,          # Long enough iterations
        'samples_per_iteration': 300,   # Larger N ensures stable estimation
        'selection_ratio': 0.05,        # 5% selection ratio
    }
    
    # Run multiple experiments with different seeds
    num_runs = 10  # Run 10 times
    random_seeds = [42, 123, 456, 789, 2023, 2024, 2025, 3141, 1618, 9999]
    
    all_results = []
    
    print(f"\nRunning {num_runs} experiments with different random seeds...")
    for i, seed in enumerate(random_seeds, 1):
        print(f"\n--- Run {i}/{num_runs} (seed={seed}) ---")
        exp = GaussianExperiment(**config, random_seed=seed)
        results_df = exp.run_all()
        results_df['run'] = i
        results_df['seed'] = seed
        all_results.append(results_df)
    
    # Combine results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Save full data
    combined_df.to_csv(os.path.join(save_dir, 'all_runs.csv'), index=False)
    print(f"\n✓ All results saved to: {save_dir}/all_runs.csv")
    
    # Plotting (with error bars)
    print("\nGenerating plots with confidence intervals...")
    plot_results_with_error_bars(combined_df, save_dir, num_runs)
    
    # Summary
    print("\n" + "=" * 70)
    print(f"Summary Statistics (averaged over {num_runs} runs, Iteration 200):")
    print("=" * 70)
    
    final_iter = combined_df['iteration'].max()
    initial_var = combined_df[(combined_df['iteration'] == 0) & (combined_df['run'] == 1)]['variance'].iloc[0]
    
    for setting in ['Replace', 'Replace + Select', 'Accumulate', 'Accumulate + Select']:
        final_data = combined_df[(combined_df['setting'] == setting) & 
                                 (combined_df['iteration'] == final_iter)]
        
        mean_errors = final_data['mean_error'].values
        variances = final_data['variance'].values / initial_var
        
        print(f"\n{setting}:")
        print(f"  Mean error: {mean_errors.mean():.2f} ± {mean_errors.std():.2f}")
        print(f"  Variance ratio: {variances.mean():.4f} ± {variances.std():.4f} "
              f"({variances.mean()*100:.2f}% ± {variances.std()*100:.2f}%)")
    
    print("\n" + "=" * 70)
    print(f"All results saved in: {save_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
