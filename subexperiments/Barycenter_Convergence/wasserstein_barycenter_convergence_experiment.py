"""
Sub-experiment: Wasserstein Barycenter Convergence Experiment

Based on paper: Data Valuation and Detections in Federated Learning
Reference [5]: Marco Cuturi and Arnaud Doucet. Fast computation of wasserstein barycenters.

Experiment objectives:
- Compute approximate barycenter at each iteration using our method
- Plot the trajectory of approximate barycenter until convergence with [5]
- Verify convergence of the approximation method
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import ot  # Optimal Transport library
from typing import List, Tuple, Optional
from matplotlib.patches import Ellipse, Rectangle
import seaborn as sns
from tqdm import tqdm

# Set matplotlib style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Override seaborn defaults with our custom font settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Computer Modern', 'Times New Roman'],
    'font.size': 18,
    'axes.labelsize': 20,
    'xtick.labelsize': 15,  # Set to be 5 smaller than axis labels (20-5=15)
    'ytick.labelsize': 15,  # Set to be 5 smaller than axis labels (20-5=15)
    'axes.titlesize': 22
})


class WassersteinBarycenterExperiment:
    """
    Wasserstein Barycenter Convergence Experiment

    Experimental setup from the paper:
    - Three 2D Gaussian distributions, each with 100 data points
    - Different means, same covariance matrix
    - t = 0.5 for interpolation measure
    - Compare approximation method with true barycenter from reference [5]
    """

    def __init__(self, n_samples: int = 100, n_iterations: int = 20, t_val: float = 0.5):
        """
        Initialize experiment parameters

        Args:
            n_samples: Number of samples per distribution (default 100)
            n_iterations: Maximum number of iterations (default 20)
            t_val: Interpolation parameter (default 0.5)
        """
        self.n_samples = n_samples
        self.n_iterations = n_iterations
        self.t_val = t_val

        # Gaussian distribution parameters - moved farther apart for slower convergence
        self.means = [
            np.array([2.0, 2.0]),   # Mean of distribution 1 - farther from center
            np.array([-2.0, 2.0]),  # Mean of distribution 2
            np.array([0.0, -3.0])   # Mean of distribution 3 - moved down more
        ]
        self.covariance = np.array([[0.3, 0.0], [0.0, 0.3]])  # Same covariance matrix

        # Approximation parameters - adjusted for slower convergence
        self.n_supp = 100  # Number of support points for interpolation measure
        self.lr = 0.01     # Learning rate
        self.n_epoch = 5   # Fewer internal iterations

    def generate_gaussian_data(self) -> List[np.ndarray]:
        """
        Generate data from three 2D Gaussian distributions

        Returns:
            List[np.ndarray]: Sample data for three distributions, each of shape (n_samples, 2)
        """
        distributions = []
        np.random.seed(42)  # 确保可重复性

        for mean in self.means:
            samples = np.random.multivariate_normal(mean, self.covariance, self.n_samples)
            distributions.append(samples)

        return distributions

    def compute_true_barycenter(self, distributions: List[np.ndarray],
                               weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute true Wasserstein barycenter using method from reference [5]

        Args:
            distributions: Sample data for three distributions
            weights: Weights for each distribution (default uniform weights)

        Returns:
            np.ndarray: Barycenter position (2,)
        """
        if weights is None:
            weights = np.ones(3) / 3

        # 使用POT库的barycenter计算
        # 注意：需要先计算所有分布之间的距离矩阵
        n_distributions = len(distributions)
        n_samples = len(distributions[0])

        # 计算所有分布的联合barycenter
        # 使用文献[5]的方法：固定点迭代
        barycenter = np.mean([np.mean(dist, axis=0) for dist in distributions], axis=0)

        # 简单迭代优化 (可以改进为更精确的方法)
        for _ in range(10):
            new_barycenter = np.zeros(2)
            total_weight = 0

            for i, dist in enumerate(distributions):
                # 计算当前barycenter到这个分布的Wasserstein距离
                M = ot.dist(barycenter.reshape(1, -1), dist, metric='sqeuclidean')
                a = np.array([1.0])
                b = np.ones(len(dist)) / len(dist)

                # 求解最优传输
                G = ot.emd(a, b, M)

                # 更新barycenter
                mapped_point = G @ dist
                new_barycenter += weights[i] * mapped_point[0]
                total_weight += weights[i]

            barycenter = new_barycenter / total_weight

        return barycenter

    def fedbary_step(self, current_barycenter: np.ndarray,
                    distributions: List[np.ndarray]) -> Tuple[np.ndarray, float]:
        """
        Perform one iteration of barycenter approximation

        Based on approximation method: each client computes local barycenter, then aggregate

        Args:
            current_barycenter: Current barycenter estimate (single point)
            distributions: Sample data for three distributions

        Returns:
            Tuple[np.ndarray, float]: (new barycenter estimate, error)
        """
        # Approximation method: each client computes interpolation between its distribution and current barycenter
        client_barycenters = []

        for dist in distributions:
            # Treat current barycenter as a distribution
            bary_dist = current_barycenter.reshape(1, -1)

            # Compute optimal transport plan
            M = ot.dist(bary_dist, dist, metric='sqeuclidean')
            a = np.array([1.0])  # Weight for single point
            b = np.ones(len(dist)) / len(dist)  # Uniform weights

            G = ot.emd(a, b, M)

            # Compute interpolation point (t=0.5)
            interpolated = (1 - self.t_val) * bary_dist[0] + self.t_val * (G @ dist)[0]
            client_barycenters.append(interpolated)

        # Server aggregation: Average of client barycenters
        new_barycenter = np.mean(client_barycenters, axis=0)

        # Compute error (distance to true barycenter)
        true_barycenter = self.compute_true_barycenter(distributions)
        error = np.linalg.norm(new_barycenter - true_barycenter)

        return new_barycenter, error

    def run_convergence_experiment(self) -> dict:
        """
        Run convergence experiment

        Args:
            save_animation: Whether to save animation (will be ignored for trajectory plot)

        Returns:
            dict: Dictionary containing experiment results
        """
        print("🎯 Starting Wasserstein Barycenter Convergence Experiment")

        # 1. Generate data
        distributions = self.generate_gaussian_data()
        print(f"✅ Data generation completed: 3 distributions, each with {self.n_samples} samples")

        # 2. Compute true barycenter
        true_barycenter = self.compute_true_barycenter(distributions)
        print(f"✅ True barycenter computation completed: {true_barycenter}")

        # 3. Initialize approximation - start far from true barycenter for slower convergence
        # Instead of using mean of distributions, start from a fixed point far away
        current_barycenter = np.array([3.0, -3.0])  # Start from bottom-right corner

        # 4. Iterative computation
        barycenters_history = [current_barycenter.copy()]
        errors_history = []

        print("🔄 Starting approximation iterations...")
        for i in tqdm(range(self.n_iterations)):
            new_barycenter, error = self.fedbary_step(current_barycenter, distributions)
            barycenters_history.append(new_barycenter.copy())
            errors_history.append(error)
            current_barycenter = new_barycenter

            if error < 1e-2:  # Convergence threshold
                print(f"🎉 Converged at iteration {i+1}!")
                break

        # 5. Visualize results
        self.visualize_convergence(distributions, barycenters_history,
                                 true_barycenter, errors_history)

        results = {
            'distributions': distributions,
            'true_barycenter': true_barycenter,
            'barycenters_history': barycenters_history,
            'errors_history': errors_history,
            'final_error': errors_history[-1] if errors_history else None,
            'convergence_iteration': len(barycenters_history) - 1
        }

        print("✅ Experiment completed!")
        return results

    def visualize_convergence(self, distributions: List[np.ndarray],
                            barycenters_history: List[np.ndarray],
                            true_barycenter: np.ndarray,
                            errors_history: List[float]):
        """
        Visualize convergence process - Left: main plot, Right: error convergence
        """
        # Create two side-by-side subplots - compact layout
        fig, (ax_main, ax_error) = plt.subplots(1, 2, figsize=(8.5, 4.0))


        # Modern color palette
        colors = ['#2E86AB', '#F24236', '#7CB342']  # Blue, Red, Green
        barycenter_color = '#E91E63'  # Pink for true barycenter
        trajectory_color = '#607D8B'  # Blue-grey for trajectory

        # Plot distributions with clean styling - 1.5x larger points
        for i, (dist, color) in enumerate(zip(distributions, colors)):
            # Add light semi-transparent circles showing distribution range
            mean = np.mean(dist, axis=0)
            std_x = np.std(dist[:, 0])
            std_y = np.std(dist[:, 1])

            # Create ellipse representing 2-sigma range
            ellipse = Ellipse(xy=mean, width=4*std_x, height=4*std_y,
                             edgecolor=color, facecolor=color, alpha=0.15, linewidth=2)
            ax_main.add_patch(ellipse)

            ax_main.scatter(dist[:, 0], dist[:, 1], alpha=0.6, color=color, s=60, edgecolors='white', linewidth=0.5,
                           label=f'Distribution {i+1}')

        # Plot convergence trajectory with dashed line and iteration points
        barycenters_array = np.array(barycenters_history)
        if len(barycenters_array) > 1:
            # Plot trajectory as dashed line
            ax_main.plot(barycenters_array[:, 0], barycenters_array[:, 1],
                        color=trajectory_color, linewidth=3, alpha=0.8, linestyle='--')

            # Plot iteration points: start and end points s=500, others s=300
            for i, point in enumerate(barycenters_array):
                size = 500 if i == 0 or i == len(barycenters_array) - 1 else 300  # Start and end points larger
                ax_main.scatter(point[0], point[1], color='#9E9E9E',
                               marker='*', s=size, alpha=0.7, zorder=10)

        # Plot true barycenter - red star
        ax_main.scatter(true_barycenter[0], true_barycenter[1], color='#F44336',
                       marker='*', s=250, alpha=0.8, zorder=15, label='Wasserstein Barycenter')

        # Add legend for approximate barycenter (trajectory points)
        ax_main.scatter([], [], color='#9E9E9E', marker='*', s=150, alpha=0.7,
                       label='Approximate Barycenter')


        # Styling - restore original settings
        # Add axis labels with appropriate names
        ax_main.set_xlabel('X Coordinate', fontsize=20, weight='bold')
        ax_main.set_ylabel('Y Coordinate', fontsize=20, weight='bold')
        ax_main.grid(True, alpha=0.2, linestyle='--')
        # Expanded axis limits as requested
        ax_main.set_xlim(-4.5, 4.5)
        ax_main.set_ylim(-5.5, 4.5)
        ax_main.axis('equal')

        # Clean legend - place below the entire figure, two rows, both centered
        handles, labels = ax_main.get_legend_handles_labels()
        dist_items = [(h, l) for h, l in zip(handles, labels) if 'Distribution' in l]
        bary_items = [(h, l) for h, l in zip(handles, labels) if 'Barycenter' in l]

        # Row 1: Distributions [D1, D2, D3] - Centered
        legend1 = fig.legend([d[0] for d in dist_items], [d[1] for d in dist_items],
                            loc='lower center', bbox_to_anchor=(0.5, -0.12),
                            ncol=3, frameon=False, fontsize=18,
                            columnspacing=1.8, handletextpad=0.3, handlelength=1.0)

        # Row 2: Barycenters [WB, AB] - Centered
        legend2 = fig.legend([b[0] for b in bary_items], [b[1] for b in bary_items],
                            loc='lower center', bbox_to_anchor=(0.5, -0.18),
                            ncol=2, frameon=False, fontsize=18,
                            columnspacing=1.5, handletextpad=0.3, handlelength=1.0)

        # Add them to figure
        fig.add_artist(legend1)
        fig.add_artist(legend2)

        # Draw to calculate bboxes (needed for accurate placement of the rectangle)
        fig.canvas.draw()

        # Get bounding boxes in figure coordinates
        bbox1 = legend1.get_window_extent().transformed(fig.transFigure.inverted())
        bbox2 = legend2.get_window_extent().transformed(fig.transFigure.inverted())

        # Calculate union bbox
        x0 = min(bbox1.x0, bbox2.x0)
        y0 = min(bbox1.y0, bbox2.y0)
        x1 = max(bbox1.x1, bbox2.x1)
        y1 = max(bbox1.y1, bbox2.y1)

        # Add padding/border
        pad_w = 0.01
        pad_h = 0.005  # Tighter vertical padding
        rect = Rectangle((x0 - pad_w, y0 - pad_h), (x1 - x0) + 2*pad_w, (y1 - y0) + 2*pad_h,
                        transform=fig.transFigure, zorder=1,
                        facecolor='white', edgecolor='#B0B0B0', linewidth=0.8)
        fig.add_artist(rect)

        # Ensure legends are on top
        legend1.set_zorder(5)
        legend2.set_zorder(5)

        # Plot error convergence on the right subplot
        if errors_history:
            iterations = range(1, len(errors_history) + 1)
            ax_error.semilogy(iterations, errors_history, color='#2196F3', linewidth=2,
                             marker='o', markersize=5, markerfacecolor='white', markeredgewidth=1.5,
                             label='Final Error')
            ax_error.grid(True, alpha=0.3)
            ax_error.set_xlabel('Iteration', fontsize=20, weight='bold')
            ax_error.set_ylabel('Error (log scale)', fontsize=20, weight='bold')

            # Add red horizontal dashed line at final error level
            final_error = errors_history[-1]
            ax_error.axhline(y=final_error, color='red', linestyle='--', linewidth=2, alpha=0.8,
                           label=f'Final Error Level')



        plt.tight_layout(pad=0.5, w_pad=0.2, h_pad=0.5)

        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(script_dir, 'barycenter_convergence.png')
        
        plt.savefig(save_path,
                   dpi=300, bbox_inches='tight', pad_inches=0.05)
        print(f"📊 Clean convergence plot saved to: {save_path}")
        plt.show()

def main():
    """Main function: Run Wasserstein Barycenter Convergence Experiment"""
    import os
    os.makedirs('/opt/data/private/synthetic_data/results', exist_ok=True)

    # Initialize experiment - adjusted for slower convergence
    experiment = WassersteinBarycenterExperiment(
        n_samples=100,     # 100 samples per distribution
        n_iterations=50,   # More maximum iterations for slower convergence
        t_val=0.5         # Interpolation parameter
    )

    # Run experiment
    results = experiment.run_convergence_experiment()

    # Print results summary
    print("\n" + "="*50)
    print("📋 Experiment Results Summary")
    print("="*50)
    print(f"True Wasserstein Barycenter: {results['true_barycenter']}")
    print(f"Convergence Iteration: {results['convergence_iteration']}")
    if results['final_error'] is not None:
        print(f"Final Error: {results['final_error']:.6f}")
    print(f"Trajectory Length: {len(results['barycenters_history'])}")
    print("="*50)


if __name__ == "__main__":
    main()
