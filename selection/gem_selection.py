"""
GEM (Geometric Ensemble Matching) Selector
Based on: "When Sample Selection Bias Precipitates Model Collapse" (Chapter 4)
"""

import torch
import numpy as np
import sys
import os
from typing import List, Tuple, Optional, Union

# Try to import geomloss, if not found, add RefCode to path
try:
    from geomloss import SamplesLoss
    GEOMLOSS_AVAILABLE = True
except ImportError:
    # Attempt to find geomloss in RefCode
    ref_path = "/opt/data/private/synthetic_data/RefCode/Data Valuation and Detections in Federated Learning/FedBary-main"
    if os.path.exists(ref_path):
        sys.path.append(ref_path)
        try:
            from geomloss import SamplesLoss
            GEOMLOSS_AVAILABLE = True
        except ImportError:
            GEOMLOSS_AVAILABLE = False
    else:
        GEOMLOSS_AVAILABLE = False

if not GEOMLOSS_AVAILABLE:
    print("Warning: geomloss not available. GEMSelector requires geomloss.")

class GEMSelector:
    def __init__(
        self,
        client_data: List[torch.Tensor],
        client_weights: Optional[List[float]] = None,
        device: str = "cuda",
        blur: float = 0.05,
        scaling: float = 0.9,
        debias: bool = True,
        feature_extractor=None
    ):
        """
        Initialize GEM Selector.

        Args:
            client_data: List of tensors representing data from each client (or validation sets).
                         Each tensor should be [M, D].
            client_weights: Weights for each client. Defaults to uniform.
            device: Computation device.
            blur: Sinkhorn blur parameter (epsilon).
            scaling: Sinkhorn scaling parameter.
            debias: Whether to use debiased Sinkhorn loss.
            feature_extractor: Optional callable to extract features from candidate samples.
        """
        if not GEOMLOSS_AVAILABLE:
            raise ImportError("geomloss is required for GEMSelector.")

        self.client_data = [d.to(device) for d in client_data]
        self.num_clients = len(self.client_data)
        self.device = device
        self.feature_extractor = feature_extractor
        
        if client_weights is None:
            self.client_weights = [1.0 / self.num_clients] * self.num_clients
        else:
            assert len(client_weights) == self.num_clients
            total_weight = sum(client_weights)
            self.client_weights = [w / total_weight for w in client_weights]

        self.blur = blur
        self.scaling = scaling
        self.debias = debias

        # Loss for computing gradients/potentials
        self.loss_fn = SamplesLoss(
            loss="sinkhorn",
            p=2,
            blur=blur,
            scaling=scaling,
            debias=debias,
            potentials=True,
            backend="tensorized"
        )
        
        # Loss for computing barycenter (no potentials needed)
        self.loss_fn_val = SamplesLoss(
            loss="sinkhorn",
            p=2,
            blur=blur,
            scaling=scaling,
            debias=debias,
            potentials=False,
            backend="tensorized"
        )

    def _extract_features(self, samples: Union[torch.Tensor, List]) -> torch.Tensor:
        """Helper to extract features if extractor is present"""
        if self.feature_extractor is None:
            # Assume samples is already a tensor
            if not isinstance(samples, torch.Tensor):
                # Try to convert list of tensors to tensor
                if isinstance(samples, list) and isinstance(samples[0], torch.Tensor):
                    return torch.stack(samples).to(self.device)
                raise ValueError("Samples must be Tensor if no feature_extractor provided")
            return samples.to(self.device)
            
        # Apply feature extractor
        # If samples is list (e.g. text), extractor should handle it
        # If samples is tensor (e.g. images), extractor should handle it
        features = self.feature_extractor(samples)
        return features.to(self.device)

    def compute_transport_map(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Approximates the Monge map T(x) = x - \nabla_x W(x, y) using Sinkhorn gradients.
        Ref: plot_wasserstein_barycenters_2D.py
        """
        source = source.clone().requires_grad_(True)
        # Uniform weights
        a = torch.ones(len(source), device=self.device) / len(source)
        b = torch.ones(len(target), device=self.device) / len(target)
        
        # Compute Loss
        loss = self.loss_fn_val(a, source, b, target)
        
        # Compute Gradient w.r.t source coordinates
        [grad_x] = torch.autograd.grad(loss, [source])
        
        # Transport map: x - N * grad_x (since a_i = 1/N)
        transported = source - grad_x * len(source)
        return transported.detach()

    def compute_barycenter(self, candidate_samples: torch.Tensor, n_epochs: int = 10) -> torch.Tensor:
        """
        Computes the Wasserstein Barycenter of the clients using the candidate samples as initialization.
        Iteratively updates the support points to minimize the sum of Wasserstein distances.
        
        Ref: "Collaborative Wasserstein Barycenter Estimation" (Scheme II in Methodology)
        """
        # Initialize barycenter support with candidate samples (or a subset if N is large)
        current_support = candidate_samples.clone()
        
        for epoch in range(n_epochs):
            transported_sum = torch.zeros_like(current_support)
            
            for k, client_data in enumerate(self.client_data):
                # Compute transport map T_k(x) = x - grad_x W(x, y_k)
                # This moves current_support towards client_data
                transported = self.compute_transport_map(current_support, client_data)
                transported_sum += self.client_weights[k] * transported
            
            # Update barycenter: X^(t+1) = \sum w_k T_k(X^t)
            current_support = transported_sum
            
        return current_support

    def compute_calibrated_gradient(self, f_star: torch.Tensor) -> torch.Tensor:
        """
        Computes the calibrated gradient (score) for each sample.
        Ref: detections.ipynb "values" function in RefCode.
        
        Formula: (1 + 1/(N-1)) * f_i - sum(f)/(N-1)
        
        Interpretation:
        - Positive score: Sample increases divergence (Bad, remove it)
        - Negative score: Sample decreases divergence (Good, keep it)
        """
        # Ensure f_star is 1D [N]
        if f_star.dim() > 1:
            f_star = f_star.squeeze()
            
        N = len(f_star)
        if N <= 1:
            return torch.zeros_like(f_star)
            
        sum_f = f_star.sum()
        # Note: In the notebook, it returns list(trainGradient). We return Tensor.
        # trainGradient = (1 + 1 / (training_size - 1)) * f1k - sum(f1k) / (training_size - 1)
        term1 = (1 + 1.0 / (N - 1)) * f_star
        term2 = sum_f / (N - 1)
        return term1 - term2

    def _compute_scores_internal(self, candidate_samples: torch.Tensor, target_distribution: torch.Tensor) -> torch.Tensor:
        """
        Computes GEM scores (calibrated gradients) of candidate_samples w.r.t target_distribution.
        Lower score (negative) means the sample effectively reduces the Sinkhorn divergence (high value).
        """
        N = len(candidate_samples)
        M = len(target_distribution)
        
        a = torch.ones(N, device=self.device) / N
        b = torch.ones(M, device=self.device) / M
        
        # Compute dual potentials
        # f_star (for x), g_star (for y)
        f_star, _ = self.loss_fn(a, candidate_samples, b, target_distribution)
        
        # Compute calibrated gradient
        scores = self.compute_calibrated_gradient(f_star)
        return scores

    def compute_scores(self, candidate_samples: Union[torch.Tensor, List], target_distribution: torch.Tensor = None) -> torch.Tensor:
        """
        Public interface for computing scores.
        Handles feature extraction.
        """
        features = self._extract_features(candidate_samples)
        
        if target_distribution is None:
             # Default to first client
             target_distribution = self.client_data[0]
             
        return self._compute_scores_internal(features, target_distribution)

    def select_gem_barycenter(
        self, 
        candidate_features: torch.Tensor, 
        selection_ratio: float = 0.3
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        GEM-Barycenter Selection (Method 1).
        Selects samples that are closest to the Wasserstein Barycenter of the clients.
        """
        candidate_features = candidate_features.to(self.device)
        N = len(candidate_features)
        num_select = int(N * selection_ratio)
        
        # 1. Compute Barycenter (Iterative)
        barycenter = self.compute_barycenter(candidate_features, n_epochs=10)
        
        # 2. Compute Scores w.r.t Barycenter
        scores = self._compute_scores_internal(candidate_features, barycenter)
        
        # 3. Select (Lowest scores are best)
        _, indices = torch.topk(scores, num_select, largest=False)
        selected_features = candidate_features[indices]
        
        return selected_features, indices

    def select_gem_local_greedy(
        self, 
        candidate_features: torch.Tensor, 
        selection_ratio: float = 0.3
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        GEM-Local-Greedy Selection (Scheme I).
        """
        candidate_features = candidate_features.to(self.device)
        N = len(candidate_features)
        num_select = int(N * selection_ratio)
        
        # 1. Compute Scores per Client
        all_scores = []
        for client_data in self.client_data:
            scores = self._compute_scores_internal(candidate_features, client_data)
            all_scores.append(scores)
            
        all_scores = torch.stack(all_scores, dim=1) # [N, NumClients]
        
        # 2. Submodular Maximization (Greedy Algorithm)
        selected_indices = self._solve_greedy_max(all_scores, num_select)
            
        selected_features = candidate_features[selected_indices]
        return selected_features, selected_indices

    def _solve_greedy_max(self, all_scores: torch.Tensor, num_select: int) -> torch.Tensor:
        """
        Executes the greedy maximization strategy on precomputed scores.
        """
        N, K = all_scores.shape
        
        # Min-Max Normalization per client (dim=0 is samples, dim=1 is clients)
        min_scores = all_scores.min(dim=0, keepdim=True)[0]
        max_scores = all_scores.max(dim=0, keepdim=True)[0]
        
        # Avoid division by zero
        range_scores = max_scores - min_scores
        range_scores[range_scores == 0] = 1.0
        
        normalized_scores = (all_scores - min_scores) / range_scores
        
        # Utility term: 1 - S_tilde
        utility_matrix = 1.0 - normalized_scores # [N, K]
        utility_matrix = torch.clamp(utility_matrix, min=0.0)
        
        selected_indices = []
        current_client_utilities = torch.zeros(K, device=self.device)
        mask = torch.ones(N, dtype=torch.bool, device=self.device)
        
        # Concave function g(z) = log(1 + z)
        def g(z):
            return torch.log(1 + z)
            
        for _ in range(num_select):
            remaining_indices = torch.nonzero(mask).squeeze()
            if remaining_indices.numel() == 0:
                break
            
            if remaining_indices.dim() == 0:
                 remaining_indices = remaining_indices.unsqueeze(0)

            candidates_utility = utility_matrix[remaining_indices] # [N_remain, K]
            
            # Vectorized gain computation
            # Maximize sum_k [ g(curr_k + util_ik) - g(curr_k) ]
            new_total = current_client_utilities.unsqueeze(0) + candidates_utility
            marginal_gains = g(new_total) - g(current_client_utilities.unsqueeze(0))
            total_gains = marginal_gains.sum(dim=1) # [N_remain]
            
            best_local_idx = torch.argmax(total_gains)
            best_global_idx = remaining_indices[best_local_idx]
            
            selected_indices.append(best_global_idx.item())
            current_client_utilities += utility_matrix[best_global_idx]
            mask[best_global_idx] = False
            
        return torch.tensor(selected_indices, device=self.device)

    def select(
        self,
        candidate_samples: Union[torch.Tensor, List],
        selection_ratio: float = 0.3,
        method: str = "local_greedy" 
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Unified selection interface.
        """
        # Extract features first
        features = self._extract_features(candidate_samples)
        
        if method == "barycenter":
            selected_features, indices = self.select_gem_barycenter(features, selection_ratio)
        elif method == "local_greedy":
            selected_features, indices = self.select_gem_local_greedy(features, selection_ratio)
        else:
            raise ValueError(f"Unknown selection method: {method}")
            
        # Return corresponding original samples
        indices_list = indices.cpu().tolist()
        if isinstance(candidate_samples, torch.Tensor):
            selected_samples = candidate_samples[indices]
        elif isinstance(candidate_samples, list):
            selected_samples = [candidate_samples[i] for i in indices_list]
        else:
            # Fallback for other types
            selected_samples = features[indices] # Return features if type unknown
            
        return selected_samples, indices
