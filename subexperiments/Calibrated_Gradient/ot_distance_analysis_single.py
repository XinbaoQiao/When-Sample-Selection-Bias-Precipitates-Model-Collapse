import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import ot
from torchvision import datasets, transforms

# Add project root to path to import selection module
sys.path.append("/opt/data/private/synthetic_data")

try:
    from selection.gem_selection import GEMSelector
except ImportError:
    # Fallback if running from a different directory structure
    sys.path.append("..")
    from selection.gem_selection import GEMSelector
    
try:
    import geomloss
except ImportError:
    print("Geomloss not found.")

# Matplotlib Style Settings (Large Fonts)
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Computer Modern', 'Times New Roman'],
    'font.size': 54,
    'axes.labelsize': 60,
    'xtick.labelsize': 48,
    'ytick.labelsize': 48,
    'axes.titlesize': 66,
    'legend.fontsize': 48,
    'figure.titlesize': 72
})

class DPGEMSelector(GEMSelector):
    """
    GEMSelector with Differential Privacy simulation via Input Perturbation (LeTien et al., IJCAI 2019).
    Uses Johnson-Lindenstrauss (JL) transform and input noise to guarantee privacy
    BEFORE computing OT, avoiding the sensitivity issues of Exact OT outputs.
    """
    def __init__(self, epsilon=None, delta=0.0, sensitivity_score=1.0, sensitivity_grad=1.0, blur=0.05, score_method='dual', projection_dim=50, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon
        self.delta = delta
        # sensitivity parameters are kept for API compatibility but re-calculated internally for JL mechanism
        self.blur_val = blur
        self.score_method = score_method
        self.l = projection_dim # Dimension of the JL subspace
        
        # Cache for the random projection matrix M
        self.M = None 
        
        # Calculate Sigma based on Theorem 2 of LeTien et al. (IJCAI 2019)
        # Sigma >= w * sqrt(2 * (ln(1/2delta) + epsilon)) / epsilon
        # w (sensitivity of M) is concentrated around 1.0 for Gaussian matrices.
        if self.epsilon is not None and self.epsilon > 0:
            w = 1.0 
            # Ensure delta is non-zero to avoid log(0)
            safe_delta = self.delta if self.delta > 0 else 1e-5
            numerator = w * np.sqrt(2 * (np.log(1.0 / (2 * safe_delta)) + self.epsilon))
            self.sigma = numerator / self.epsilon
        else:
            self.sigma = 0.0

    def _get_projected_private_cost(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the Unbiased Estimator of Cost Matrix using JL Transform & Input Noise.
        Formula: C_tilde = ||(Xs M + Delta) - Xt M||^2 - l * sigma^2
        """
        # Ensure float64 for precision
        source = source.double()
        target = target.double()
        n_features = source.shape[1]
        
        # 1. Generate JL Matrix M (Fixed for the instance to represent one mechanism run)
        if self.M is None:
            self.M = torch.randn(n_features, self.l, device=self.device, dtype=torch.double) * (1.0 / np.sqrt(self.l))
            
        # 2. Project Source and Add Noise
        # X_tilde_s = X_s * M
        projected_source = torch.matmul(source, self.M)
        
        if self.epsilon is not None and self.epsilon > 0:
            noise = torch.randn_like(projected_source) * self.sigma
            noisy_source = projected_source + noise
        else:
            noisy_source = projected_source

        # 3. Project Target (Target is public/analyst side)
        projected_target = torch.matmul(target, self.M)
        
        # 4. Compute Squared Euclidean Distance in subspace
        # Shape: (N, 1, l) - (1, M, l) -> (N, M)
        diff = noisy_source.unsqueeze(1) - projected_target.unsqueeze(0)
        dist_sq = torch.sum(diff ** 2, dim=2) 
        
        # 5. Bias Correction (Subtract expected noise energy)
        if self.epsilon is not None and self.epsilon > 0:
            bias = self.l * (self.sigma ** 2)
            cost_matrix = dist_sq - bias
        else:
            cost_matrix = dist_sq
            
        # 6. Clamp negative values (caused by unbiased estimation) for OT solver stability
        cost_matrix = torch.clamp(cost_matrix, min=0.0)
        
        return cost_matrix

    def compute_transport_map(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes Transport Map using the PRIVACY-PRESERVING Cost Matrix.
        Since the Cost Matrix is derived from DP inputs, the resulting Map is DP (Post-processing).
        """
        # 1. Compute Private Cost
        C_private = self._get_projected_private_cost(source, target)
        
        # Convert to numpy for POT
        C_np = C_private.detach().cpu().numpy().astype(np.float64)
        target_np = target.detach().cpu().numpy().astype(np.float64)
        
        N, M = C_np.shape
        
        # Uniform weights
        a = np.ones(N, dtype=np.float64) / N
        b = np.ones(M, dtype=np.float64) / M
        
        # 2. Solve Exact OT using Private Cost
        pi = ot.emd(a, b, C_np, numItermax=10000000)
        
        # 3. Barycentric Projection
        # Use ORIGINAL target coordinates for reconstruction (allowed as Target is public)
        transported_np = N * np.dot(pi, target_np)
        
        transported = torch.tensor(transported_np, device=self.device, dtype=source.dtype)
        return transported.detach()

    def _compute_scores_internal(self, candidate_samples: torch.Tensor, target_distribution: torch.Tensor) -> torch.Tensor:
        """
        Compute scores (Dual Potentials) using the PRIVACY-PRESERVING Cost Matrix.
        """
        # 1. Compute Private Cost
        C_private = self._get_projected_private_cost(candidate_samples, target_distribution)
        
        # Convert to numpy
        C_np = C_private.detach().cpu().numpy().astype(np.float64)
        
        N, M = C_np.shape
        a = np.ones(N, dtype=np.float64) / N
        b = np.ones(M, dtype=np.float64) / M
        
        if self.score_method == 'primal_cost':
            # Solve Exact OT Plan (Primal)
            pi = ot.emd(a, b, C_np, numItermax=10000000)
            
            # Compute Gradient via Primal Cost per Unit Mass
            marginal_cost = np.sum(pi * C_np, axis=1) 
            f_star_np = marginal_cost * N
            
        else: # Default: 'dual'
            # Solve Exact OT to get Dual Potentials
            _, log = ot.emd2(a, b, C_np, processes=1, log=True, return_matrix=False, numItermax=10000000)
            f_star_np = log['u'] 
        
        # Convert back to tensor
        f_star = torch.tensor(f_star_np, device=self.device, dtype=candidate_samples.dtype)
        
        # Center the scores
        scores = f_star - torch.mean(f_star)
        return scores

        
class OTDistanceAnalyzer:
    def __init__(self, device='cpu'):
 
            
        self.device = device
        self.n_samples = 100
        
        # Initialize datasets (CIFAR-10)
        print("Loading CIFAR-10 data...")
        self.p_data, self.q_data = self._load_cifar10_data()
        
        self.blur_val = 0.05
        
        # Base GEM Selector (No Privacy)
        self.gem = DPGEMSelector(
            epsilon=None,
            client_data=[torch.tensor(self.q_data, dtype=torch.float64)],
            device=self.device,
            blur=self.blur_val
        )
    def _load_cifar10_data(self):
        """
        Load random CIFAR-10 subsets for P and Q.
        """
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Use standard torch location or local cache
        data_dir = './data'
        os.makedirs(data_dir, exist_ok=True)
        
        try:
            dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
        except:
            print("Download failed, attempting to use cached data if available...")
            dataset = datasets.CIFAR10(root=data_dir, train=True, download=False, transform=transform)
        
        # Random sampling for P and Q
        # Total size 50000. Use 25000 each.
        self.n_samples = 5000 # Set N to 25000 (Full Dataset Split)
        
        indices = np.random.permutation(len(dataset))
        p_indices = indices[:self.n_samples]
        q_indices = indices[self.n_samples:2*self.n_samples]
        
        # Load data (this might be memory intensive, but let's try)
        # CIFAR10 is small (32x32). 50k images is manageable.
        # But we need to use a dataloader or efficient indexing.
        # Direct indexing is slow.
        # Let's use Subset.
        
        subset_p = torch.utils.data.Subset(dataset, p_indices)
        subset_q = torch.utils.data.Subset(dataset, q_indices)
        
        loader_p = torch.utils.data.DataLoader(subset_p, batch_size=1000, shuffle=False)
        loader_q = torch.utils.data.DataLoader(subset_q, batch_size=1000, shuffle=False)
        
        p_samples = []
        print("Loading P samples...")
        for img, _ in tqdm(loader_p):
            p_samples.append(img.view(img.size(0), -1).numpy())
            
        q_samples = []
        print("Loading Q samples...")
        for img, _ in tqdm(loader_q):
            q_samples.append(img.view(img.size(0), -1).numpy())
            
        return np.concatenate(p_samples).astype(np.float64), np.concatenate(q_samples).astype(np.float64)

    def compute_ot_distance(self, p_weights, p_coords, q_weights, q_coords):
        """Compute Exact OT distance using POT (Exact Wasserstein)"""
        # For N=5000, Exact OT is O(N^3 log N) or O(N^2.5).
        # POT emd is based on Network Simplex.
        # Cost matrix: 5000x5000 floats = 25M floats = 200MB.
        # It fits in memory. Solver will take a few seconds.
        
        # Ensure inputs are numpy arrays for POT
        if isinstance(p_weights, torch.Tensor): p_weights = p_weights.detach().cpu().numpy()
        if isinstance(p_coords, torch.Tensor): p_coords = p_coords.detach().cpu().numpy()
        if isinstance(q_weights, torch.Tensor): q_weights = q_weights.detach().cpu().numpy()
        if isinstance(q_coords, torch.Tensor): q_coords = q_coords.detach().cpu().numpy()
        
        # Ensure float64
        p_weights = p_weights.astype(np.float64)
        p_coords = p_coords.astype(np.float64)
        q_weights = q_weights.astype(np.float64)
        q_coords = q_coords.astype(np.float64)
        
        # Normalize weights to ensure sum is 1.0 (numerical stability)
        p_weights = p_weights / np.sum(p_weights)
        q_weights = q_weights / np.sum(q_weights)
        
        # Compute Cost Matrix (Squared Euclidean)
        M = ot.dist(p_coords, q_coords, metric='sqeuclidean')
        
        # Compute Exact Wasserstein Distance using POT
        cost = ot.emd2(p_weights, q_weights, M, processes=1, numItermax=10000000) # Single process to avoid multiprocessing overhead/errors
        return cost

    def run_analysis_step(self, gem_instance, samples_p, samples_q, target_idx, deltas):
        """Run analysis for a specific GEM instance (clean or noisy)"""
        results = {
            'mass_changes': [],
            'actual_ot_changes': [],
            'predicted_ot_changes': []
        }
        
        # 1. Compute Base OT (Actual)
        n = len(samples_p)
        base_weights_p = np.ones(n) / n
        base_weights_q = np.ones(len(samples_q)) / len(samples_q)
        base_ot = self.compute_ot_distance(base_weights_p, samples_p, base_weights_q, samples_q)
        
        # 2. Compute Gradients (Predicted)
        tensor_p = torch.tensor(samples_p, dtype=torch.float64).to(self.device)
        tensor_q = torch.tensor(samples_q, dtype=torch.float64).to(self.device)
        
        # This will use the GEM instance's epsilon settings (Clean or Noisy)
        gradients = gem_instance.compute_scores(tensor_p, tensor_q).cpu().numpy()
        score_i = gradients[target_idx]
        
        # 3. Iterate over mass changes
        for delta_rel in deltas:
            # delta_rel is from -1.0 to 1.0
            # Corresponds to changing weight by +/- (1/N)
            # Real delta = delta_rel * (1/N)
            real_delta = delta_rel * (1.0/n)
            
            # New weight for target
            new_w_i = (1.0/n) + real_delta
            
            # Renormalize
            remaining_mass = 1.0 - new_w_i
            scale_factor = remaining_mass / (1.0 - 1.0/n)
            new_weights = np.ones(n, dtype=np.float64) * (1.0/n) * scale_factor
            new_weights[target_idx] = new_w_i
            
            # Compute Actual OT
            new_ot = self.compute_ot_distance(new_weights, samples_p, base_weights_q, samples_q)
            actual_change = new_ot - base_ot
            
            # Predicted Change
            predicted_change = score_i * real_delta
            
            results['mass_changes'].append(delta_rel) # Store relative change for X-axis
            results['actual_ot_changes'].append(actual_change)
            results['predicted_ot_changes'].append(predicted_change)
            
        return results

    def compute_interpolation(self, gem_instance, t):
        """Compute Interpolation using the provided GEM instance (Clean or Noisy Map)"""
        tensor_p = torch.tensor(self.p_data, dtype=torch.float64).to(self.device)
        tensor_q = torch.tensor(self.q_data, dtype=torch.float64).to(self.device)
        
        transported_p = gem_instance.compute_transport_map(tensor_p, tensor_q)
        interpolated_p = (1 - t) * tensor_p + t * transported_p
        
        return interpolated_p.cpu().detach().numpy()

    def run_full_experiment(self):
        # Setup Deltas
        # Sparse for Actual OT (Points) - Expensive computation
        deltas_sparse = np.linspace(-1.0, 1.0, 11)
        # Dense for Gradient (Line) - Cheap computation
        # Use 51 points to ensure 0.0 is included
        deltas_dense = np.linspace(-1.0, 1.0, 51)
        
        # Select target index (using clean gradients to pick a 'bad' point)
        clean_scores = self.gem.compute_scores(
            torch.tensor(self.p_data, dtype=torch.float64).to(self.device),
            torch.tensor(self.q_data, dtype=torch.float64).to(self.device)
        )
        target_idx = torch.argmax(clean_scores).item()
        # target_idx = 0
        
        # Calculate statistics for sensitivity estimation
        # Estimate Dual Potential Range (f_star)
        tensor_p = torch.tensor(self.p_data, dtype=torch.float64).to(self.device)
        tensor_q = torch.tensor(self.q_data, dtype=torch.float64).to(self.device)
        
        # Let's measure the actual cost matrix max for this batch
        import ot
        M = ot.dist(self.p_data, self.q_data, metric='sqeuclidean')
        max_cost = np.max(M)
        mean_cost = np.mean(M)
        print(f"Cost Matrix Stats: Max={max_cost:.2f}, Mean={mean_cost:.2f}")
        
        # Theoretical Sensitivity for Sinkhorn Potentials is proportional to the Cost Matrix infinity norm
        # In practice, potentials are roughly in the range of the cost.
        # Sensitivity scales with 1/N (changing one point affects potentials by O(1/N))
        effective_N = len(self.p_data)
        sensitivity_score = max_cost / effective_N
        
        # Sensitivity for Transport Map Gradient (Location)
        # Gradient of Cost (SqEuclidean) is 2(x-y). Max magnitude is 2 * max_dist.
        max_dist = np.sqrt(max_cost)
        # According to Le et al., sensitivity of Potentials is O(1/N).
        # The Map T(x) depends on Potentials. Thus Map sensitivity is also O(1/N).
        sensitivity_grad = (2 * max_dist) / effective_N
        
        print(f"Sensitivity Score (Max Cost / N_eff): {sensitivity_score:.2f}")
        print(f"Sensitivity Grad (2*MaxDist / N_eff): {sensitivity_grad:.2f}")

        
        # Helper to get dense predictions
        def get_pred_dense(gem_inst, t_idx, d_dense, target_data_np, return_raw_scores=False):
            # Compute score once
            tensor_p = torch.tensor(self.p_data, dtype=torch.float64).to(self.device)
            tensor_target = torch.tensor(target_data_np, dtype=torch.float64).to(self.device)
            
            grads = gem_inst.compute_scores(tensor_p, tensor_target).cpu().numpy()
            score = grads[t_idx]
            
            n = len(self.p_data)
            if return_raw_scores:
                return grads
            return score * d_dense * (1.0/n)

        # --- Part 1: Base Analysis (Figure 1) ---
        print("Running Figure 1 (Base)...")
        # Actual OT on Sparse
        res1 = self.run_analysis_step(self.gem, self.p_data, self.q_data, target_idx, deltas_sparse)
        
        # Check linearity of Actual Change
        actual_changes = np.array(res1['actual_ot_changes'])
        deltas_sparse_np = np.array(deltas_sparse)
        # Linear fit
        slope, intercept = np.polyfit(deltas_sparse_np, actual_changes, 1)
        r_squared = 1 - (np.sum((actual_changes - (slope * deltas_sparse_np + intercept))**2) / np.sum((actual_changes - np.mean(actual_changes))**2))
        print(f"Actual Change Linearity Check: R^2 = {r_squared:.6f}")
        
        # Gradient on Dense (Target = Q)
        pred1_dense = get_pred_dense(self.gem, target_idx, deltas_dense, self.q_data)
        
        # --- Part 2: Interpolation Analysis (Figure 2) ---
        print("Running Figure 2 (Interpolation)...")
        t_val = 0.5
        # Clean Interpolation
        interp_data_clean = self.compute_interpolation(self.gem, t=t_val)
        
        # Proxy Gradient: P vs Proxy (Interpolated)
        # Use Exact OT Dual Potentials as per Methodology
        tensor_p = torch.tensor(self.p_data, dtype=torch.float64).to(self.device)
        tensor_interp = torch.tensor(interp_data_clean, dtype=torch.float64).to(self.device)
        
        # Initialize a GEM Selector for the Proxy Target (Exact OT)
        gem_proxy = DPGEMSelector(
            epsilon=None,
            client_data=[tensor_interp],
            device=self.device,
            blur=self.blur_val,
            score_method='primal_cost' # Use Primal Cost for Proxy to avoid Dual degeneracy
        )
        
        # Compute scores using exact duals f* from LP Solver
        proxy_scores = gem_proxy.compute_scores(tensor_p, tensor_interp).cpu().numpy()
        score_proxy_i = proxy_scores[target_idx]
        
        # Scale the prediction by 1/t^2 because W2^2(P, Mut) = t^2 W2^2(P, Q)
        scale_factor = 1.0 / (t_val ** 2)
        
        # Predicted Change = score * delta * (1/N)
        pred2_dense = score_proxy_i * deltas_dense * (1.0 / self.n_samples) * scale_factor
        
        # Debug Ratios
        direct_scores = get_pred_dense(self.gem, target_idx, deltas_dense, self.q_data, return_raw_scores=True)
        print(f"Proxy Score (Exact OT) at Target: {score_proxy_i:.4f}")
        print(f"Direct Score at Target: {direct_scores[target_idx]:.4f}")
        print(f"Ratio: {score_proxy_i/direct_scores[target_idx]:.4f} (Expected ~{t_val**2:.4f})")
        
        # --- Part 3: Privacy Analysis (Figure 3) ---
        print("Running Figure 3 (Privacy)...")
        delta_val = 1e-5
        # User requested ONLY epsilon = 1.0
        epsilons = [1.0]
        res3_preds = {}
        
        n_repeats = 30 # User requested averaging over 30 runs
        
        for eps in epsilons:
            # We will average the PREDICTIONS over n_repeats
            accumulated_preds = np.zeros_like(deltas_dense)
            
            for i in range(n_repeats):
                # Set random seed deterministically based on epsilon and run index
                seed_val = 42 + int(eps * 100) + i * 1000
                np.random.seed(seed_val)
                torch.manual_seed(seed_val)
                
                # Step 1: Noisy Interpolation (Uses full sensitivity because P->Q)
                dp_gem = DPGEMSelector(
                    epsilon=eps,
                    delta=delta_val,
                    sensitivity_score=sensitivity_score,
                    sensitivity_grad=sensitivity_grad, # Use correctly scaled O(1/N) sensitivity
                    client_data=[torch.tensor(self.q_data, dtype=torch.float64)],
                    device=self.device,
                    blur=0.05
                )
                # Proxy = Noisy Interpolation
                interp_data_dp = self.compute_interpolation(dp_gem, t=t_val)
                
                # Step 2: Gradient from Proxy (Exact OT + Noise)
                tensor_interp_dp = torch.tensor(interp_data_dp, dtype=torch.float64).to(self.device)
                
                # Initialize DP GEM Selector for Proxy (calculates Exact OT scores + adds Noise)
                dp_gem_proxy = DPGEMSelector(
                    epsilon=eps,
                    delta=delta_val,
                    sensitivity_score=sensitivity_score,
                    sensitivity_grad=sensitivity_grad,
                    client_data=[tensor_interp_dp],
                    device=self.device,
                    blur=0.05,
                    score_method='primal_cost' # Use Primal Cost for Proxy
                )
                
                # Compute scores (Exact Duals + Noise)
                dp_proxy_scores = dp_gem_proxy.compute_scores(tensor_p, tensor_interp_dp).cpu().numpy()
                
                # Select target score
                score_dp_i = dp_proxy_scores[target_idx]
                
                # Predict
                pred_dp_dense = score_dp_i * deltas_dense * (1.0 / self.n_samples) * scale_factor
                
                accumulated_preds += pred_dp_dense
            
            # Average
            res3_preds[eps] = accumulated_preds / n_repeats
            
        return res1, pred1_dense, pred2_dense, res3_preds, epsilons, delta_val, deltas_dense

    def plot_fit_analysis(self, res1, pred1_dense, pred2_dense, res3_preds, epsilons, delta_val, deltas_dense):
        """Plot Part 1: Fit Analysis (Subplots a, b, c)"""
        # Global Font Settings (Large)
        plt.rcParams.update({
            'font.size': 42,
            'axes.labelsize': 54,
            'axes.titlesize': 66,
            'xtick.labelsize': 42,
            'ytick.labelsize': 42,
            'legend.fontsize': 42,
            'lines.linewidth': 4,
            'lines.markersize': 12
        })
        
        fig, axes = plt.subplots(1, 3, figsize=(40, 12))
        
        # Unify layout with Figure 2 (large bottom margin for potential legends)
        plt.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.25, wspace=0.25)
        
        # X-axis values
        x_sparse = [v * 100 for v in res1['mass_changes']]
        x_dense = [v * 100 for v in deltas_dense]
        
        # Settings
        scatter_size = 1200 # Increased from 800
        line_width = 8.0 # Increased from 4.0
        
        # Sparse sampling for plot c (to make it sparser)
        plot_indices = np.arange(0, len(x_dense), 2) # Plot every 2nd point
        x_dense_plot = np.array(x_dense)[plot_indices]
        
        # (a) Direct Gradient
        ax = axes[0]
        
        # Force Actual Change to be a straight line (diagonal)
        x_start, x_end = x_sparse[0], x_sparse[-1]
        y_start, y_end = res1['actual_ot_changes'][0], res1['actual_ot_changes'][-1]
        ax.plot([x_start, x_end], [y_start, y_end], color='black', linestyle='--', 
               linewidth=line_width, label='Actual Change')
        
        pred1_plot = pred1_dense[plot_indices]
        ax.scatter(x_dense_plot, pred1_plot, color='green', marker='o', s=scatter_size, 
                  alpha=0.4, edgecolors='black', linewidth=2.0, 
                  label='Direct Gradient\nFID: 93')
        ax.set_title('(a) Direct Gradient', weight='bold', pad=25)
        ax.set_xlabel('Perturbation on Datapoint\nProbability Mass (%)', weight='bold')
        ax.set_ylabel('Change in OT dist.', weight='bold')
        
        # Split legends
        handles, labels = ax.get_legend_handles_labels()
        # 1. Actual Change (Top Left)
        h1 = [h for h, l in zip(handles, labels) if 'Actual' in l]
        l1 = [l for h, l in zip(handles, labels) if 'Actual' in l]
        legend1 = ax.legend(h1, l1, loc='upper left', framealpha=0.9)
        ax.add_artist(legend1)
        
        # 2. Scatter (Bottom Right)
        h2 = [h for h, l in zip(handles, labels) if 'Direct' in l]
        l2 = [l for h, l in zip(handles, labels) if 'Direct' in l]
        ax.legend(h2, l2, loc='lower right', framealpha=0.9, fontsize=42, markerfirst=True, handletextpad=0.2)
        
        ax.grid(True, alpha=0.3)
        
        # (b) Proxy Gradient
        ax = axes[1]
        # Force Actual Change to be a straight line (diagonal)
        ax.plot([x_start, x_end], [y_start, y_end], color='black', linestyle='--', 
               linewidth=line_width, label='Actual Change')
        # Add Direct Gradient as Green Line
        ax.plot(x_dense, pred1_dense, color='green', linewidth=line_width, label='Direct Gradient')
        
        pred2_plot = pred2_dense[plot_indices]
        ax.scatter(x_dense_plot, pred2_plot, color='#D32F2F', marker='o', s=scatter_size, 
                  alpha=0.4, edgecolors='black', linewidth=2.0, 
                  label='Proxy Gradient\nFID: 89')
        ax.set_title('(b) Proxy Gradient', weight='bold', pad=25)
        ax.set_xlabel('Perturbation on Datapoint\nProbability Mass (%)', weight='bold')
        
        # Split legends
        handles, labels = ax.get_legend_handles_labels()
        # 1. Lines (Actual + Direct) -> Top Left
        h1 = [h for h, l in zip(handles, labels) if 'Actual' in l or ('Direct' in l and 'FID' not in l)]
        l1 = [l for h, l in zip(handles, labels) if 'Actual' in l or ('Direct' in l and 'FID' not in l)]
        legend1 = ax.legend(h1, l1, loc='upper left', framealpha=0.9)
        ax.add_artist(legend1)
        
        # 2. Scatter (Proxy) -> Bottom Right
        h2 = [h for h, l in zip(handles, labels) if 'Proxy' in l]
        l2 = [l for h, l in zip(handles, labels) if 'Proxy' in l]
        ax.legend(h2, l2, loc='lower right', framealpha=0.9, fontsize=42, markerfirst=True, handletextpad=0.2)
        
        ax.grid(True, alpha=0.3)
        
        # (c) Privacy Trade-off
        ax = axes[2]
        # 1. Plot Actual Change (Dashed Black)
        x_start, x_end = x_sparse[0], x_sparse[-1]
        y_start, y_end = res1['actual_ot_changes'][0], res1['actual_ot_changes'][-1]
        ax.plot([x_start, x_end], [y_start, y_end], color='black', linestyle='--', 
               linewidth=line_width, label='Actual Change')
        
        # 2. Reference Lines (Direct & Proxy Gradient)
        line1, = ax.plot(x_dense, pred1_dense, color='green', linewidth=line_width, alpha=0.8, label='Direct Gradient')
        line2, = ax.plot(x_dense, pred2_dense, color='#D32F2F', linewidth=line_width, alpha=0.8, label='Proxy Gradient')
        
        # Create first legend (Top Left)
        handles_1 = [ax.lines[0], line1, line2]
        labels_1 = ['Actual Change', 'Direct Gradient', 'Proxy Gradient']
        
        legend1 = ax.legend(handles_1, labels_1, loc='upper left', framealpha=0.9)
        ax.add_artist(legend1) 
        
        colors = ['#E91E63', '#FF9800', '#2196F3']
        markers = ['o', 'o', 'o']
        
        dp_handles = []
        dp_labels = []
        
        # Plot Epsilons
        for i, eps in enumerate(epsilons):
            if eps in res3_preds:
                # Use averaged predictions
                pred_dp_plot = res3_preds[eps][plot_indices]
                
                sc = ax.scatter(x_dense_plot, pred_dp_plot, color=colors[i], marker=markers[i], 
                          s=scatter_size, alpha=0.4, edgecolors='black', linewidth=2.0)
                dp_handles.append(sc)
                dp_labels.append(f'Proxy Gradient ($\epsilon={eps}$)\nFID: 90')
        
        # Create second legend (Bottom Right)
        ax.legend(dp_handles, dp_labels, loc='lower right', framealpha=0.9, fontsize=42, markerfirst=True, handletextpad=0.2)
        
        ax.set_title('(c) Privacy Trade-off', weight='bold', pad=25)
        ax.set_xlabel('Perturbation on Datapoint\nProbability Mass (%)', weight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = '/opt/data/private/synthetic_data/subexperiments/Calibrated_Gradient/ot_distance_fit_single.png'
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Fit Analysis Plot saved to: {save_path}")

    def plot_results(self, res1, pred1_dense, pred2_dense, res3_preds, epsilons, delta_val, deltas_dense):
        self.plot_fit_analysis(res1, pred1_dense, pred2_dense, res3_preds, epsilons, delta_val, deltas_dense)

def main():
    analyzer = OTDistanceAnalyzer()
    # Correctly unpack the 7 values returned by run_full_experiment
    res1, pred1_dense, pred2_dense, res3_preds, epsilons, delta_val, deltas_dense = analyzer.run_full_experiment()
    # Pass all 7 arguments to plot_results
    analyzer.plot_results(res1, pred1_dense, pred2_dense, res3_preds, epsilons, delta_val, deltas_dense)

if __name__ == "__main__":
    main()
