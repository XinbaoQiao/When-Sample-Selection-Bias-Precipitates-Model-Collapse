"""
Image Data Selection Methods
Based on reference implementation: High-dimensional Analysis of Synthetic Data Selection
"""
import torch
import numpy as np
from typing import List, Tuple, Optional, Union
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from scipy import linalg
import warnings
warnings.filterwarnings("ignore")

try:
    from pytorch_fid import fid_score
    PYTORCH_FID_AVAILABLE = True
except ImportError:
    PYTORCH_FID_AVAILABLE = False
    print("Warning: pytorch_fid not available. FID-based methods will not work.")

try:
    from selfrepresentation import SparseSubspaceClusteringOMP
    from ole import OLELoss
    OLE_AVAILABLE = True
except ImportError:
    OLE_AVAILABLE = False
    print("Warning: OLE modules not available. Using alternative density estimation.")


class RandomFilter:
    """
    Random selection method
    Reference: random_selection function
    """
    def __init__(self, device="cuda"):
        self.device = device

    def select(self, images: torch.Tensor, selection_ratio: float = 0.3, used=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Randomly select samples
        """
        available_indices = torch.where(~used)[0] if used is not None else torch.arange(len(images))
        if len(available_indices) < int(len(images) * selection_ratio):
            num_select = len(available_indices)
        else:
            num_select = int(len(images) * selection_ratio)

        selected_indices = torch.randperm(len(available_indices))[:num_select]
        matches = available_indices[selected_indices]

        selected_images = images[matches]
        return selected_images, matches


class CenterMatchingFilter:
    """
    Selection method based on center matching (He et al., 2023)
    Select ns images closest to the center of real training features
    Reference: Center_matching function
    """
    def __init__(self, reference_data: torch.Tensor, feature_extractor, device="cuda"):
        self.reference_data = reference_data
        self.feature_extractor = feature_extractor
        self.device = device
        # Only PyTorch models need eval(), feature extractors might not
        if hasattr(self.feature_extractor, 'eval'):
            self.feature_extractor.eval()

        # Calculate center of reference data
        with torch.no_grad():
            ref_features = self.feature_extractor(self.reference_data.to(device))
            if ref_features.dim() > 2:
                ref_features = ref_features.view(len(ref_features), -1)
            self.ref_center = ref_features.mean(dim=0)

    def select(self, images: torch.Tensor, selection_ratio: float = 0.3, used=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select samples closest to reference center
        Implementation reference: Center_matching function
        """
        with torch.no_grad():
            features = self.feature_extractor(images.to(self.device))
            if features.dim() > 2:
                features = features.view(len(features), -1)

            # Calculate distance to reference center (L2 norm)
            distances = torch.norm(features - self.ref_center.unsqueeze(0), dim=1)

        # Handle used samples
        if used is not None:
            distances[used] = float('inf')

        # Select samples with minimum distance
        num_select = int(len(images) * selection_ratio)
        sorted_indices = torch.argsort(distances)
        nearest_indices = sorted_indices[:num_select]

        selected_images = images[nearest_indices]
        return selected_images, nearest_indices


class CenterSamplingFilter:
    """
    Selection method based on center sampling (Lin et al., 2023)
    Sample with probability proportional to cosine similarity
    Reference: Center_sampling function
    """
    def __init__(self, reference_data: torch.Tensor, feature_extractor, device="cuda"):
        self.reference_data = reference_data
        self.feature_extractor = feature_extractor
        self.device = device
        # Only PyTorch models need eval(), feature extractors might not
        if hasattr(self.feature_extractor, 'eval'):
            self.feature_extractor.eval()

        # Calculate center of reference data
        with torch.no_grad():
            ref_features = self.feature_extractor(self.reference_data.to(device))
            if ref_features.dim() > 2:
                ref_features = ref_features.view(len(ref_features), -1)
            self.ref_center = ref_features.mean(dim=0)
            # Normalize reference center
            self.ref_center = self.ref_center / torch.norm(self.ref_center)

    def select(self, images: torch.Tensor, selection_ratio: float = 0.3, used=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample with cosine similarity as probability
        Implementation reference: Center_sampling function
        """
        with torch.no_grad():
            features = self.feature_extractor(images.to(self.device))
            if features.dim() > 2:
                features = features.view(len(features), -1)

            # Normalize feature vectors
            features_norm = features / torch.norm(features, dim=1, keepdim=True)

            # Calculate cosine similarity
            similarities = torch.matmul(features_norm, self.ref_center)

            # Ensure similarities are positive (avoid negative similarity)
            similarities = torch.clamp(similarities, min=1e-8)

        # Handle used samples
        if used is not None:
            similarities[used] = 0

        # Convert to probability distribution
        probabilities = similarities / torch.sum(similarities)

        # Sample
        num_select = int(len(images) * selection_ratio)
        selected_indices = torch.multinomial(probabilities.float(), num_select, replacement=False)

        selected_images = images[selected_indices]
        return selected_images, selected_indices


class DS3Filter:
    """
    DS3 selection method (Hulkund et al., 2025)
    Cluster generation pool into 200 clusters; for each real image, retain its closest cluster;
    then uniformly sample ns images from the retained set
    Reference: ds3 function
    """
    def __init__(self, reference_data: torch.Tensor, feature_extractor, device="cuda", n_clusters: int = 200):
        self.reference_data = reference_data
        self.feature_extractor = feature_extractor
        self.device = device
        self.n_clusters = n_clusters
        # Only PyTorch models need eval(), feature extractors might not
        if hasattr(self.feature_extractor, 'eval'):
            self.feature_extractor.eval()

        # Pre-calculate reference features
        with torch.no_grad():
            self.ref_features = self.feature_extractor(self.reference_data.to(device))
            if self.ref_features.dim() > 2:
                self.ref_features = self.ref_features.view(len(self.ref_features), -1)

    def select(self, images: torch.Tensor, selection_ratio: float = 0.3, used=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        DS3 algorithm implementation
        Implementation reference: ds3 function
        """
        with torch.no_grad():
            gen_features = self.feature_extractor(images.to(self.device))
            if gen_features.dim() > 2:
                gen_features = gen_features.view(len(gen_features), -1)

        # Apply K-means clustering to generated features
        gen_features_np = gen_features.cpu().numpy()
        kmeans = KMeans(n_clusters=self.n_clusters, n_init='auto', random_state=0)
        cluster_assignments = kmeans.fit_predict(gen_features_np)
        cluster_centers = torch.from_numpy(kmeans.cluster_centers_).to(self.device)

        # Find closest cluster center for each real sample
        ref_features_np = self.ref_features.cpu().numpy()
        distances_to_centers = np.linalg.norm(
            ref_features_np[:, np.newaxis, :] - kmeans.cluster_centers_[np.newaxis, :, :], axis=2
        )
        closest_centers = np.argmin(distances_to_centers, axis=1)
        unique_closest_centers = np.unique(closest_centers)

        # Collect all generated samples belonging to these clusters
        retained_indices = []
        for center_idx in unique_closest_centers:
            cluster_samples = np.where(cluster_assignments == center_idx)[0]
            retained_indices.extend(cluster_samples)

        # Handle used samples
        if used is not None:
            retained_indices = [idx for idx in retained_indices if not used[idx]]

        retained_indices = torch.tensor(retained_indices, device=self.device)
        num_select = int(len(images) * selection_ratio)

        if len(retained_indices) > num_select:
            # Uniformly sample from retained samples
            selected_indices = retained_indices[torch.randperm(len(retained_indices))[:num_select]]
        elif len(retained_indices) > 0:
            # If retained samples are insufficient, recursively reduce number of clusters
            if self.n_clusters > 2:
                self.n_clusters = self.n_clusters // 2
                return self.select(images, selection_ratio, used)
            else:
                selected_indices = retained_indices
        else:
            # If no samples retained, return random selection
            available_indices = torch.where(~used)[0] if used is not None else torch.arange(len(images))
            selected_indices = available_indices[torch.randperm(len(available_indices))[:num_select]]

        selected_images = images[selected_indices]
        return selected_images, selected_indices


class KMeansFilter:
    """
    K-means selection method (Lin et al., 2023)
    Cluster generation pool into ns clusters, select one random representative from each cluster
    Reference: K_mean function
    """
    def __init__(self, device="cuda"):
        self.device = device

    def select(self, images: torch.Tensor, selection_ratio: float = 0.3, used=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        K-means clustering selection
        Implementation reference: K_mean function
        """
        with torch.no_grad():
            # Assume input is already features; if images, extract features first
            if images.dim() > 2:
                # Flatten if image data
                features = images.view(len(images), -1)
            else:
                features = images

        num_select = int(len(images) * selection_ratio)
        features_np = features.cpu().numpy()

        # Apply K-means clustering
        kmeans = KMeans(n_clusters=num_select, n_init='auto', random_state=0)
        cluster_labels = kmeans.fit_predict(features_np)
        cluster_centers = torch.from_numpy(kmeans.cluster_centers_).to(self.device)

        # Select closest sample for each cluster
        selected_indices = []
        used_mask = torch.zeros(len(features), dtype=torch.bool, device=self.device)
        if used is not None:
            used_mask = used.clone()

        # Calculate distances from all samples to all cluster centers
        # features: [N, D], cluster_centers: [K, D]
        # dists: [N, K]
        dists = torch.cdist(features, cluster_centers)
        
        # Set distance of used samples to infinity
        if used is not None:
            dists[used] = float('inf')

        for cluster_idx in range(num_select):
            # Get indices of all samples in this cluster
            # Actually can find sample closest to center belonging to this cluster
            # Or simpler: find closest in samples belonging to this cluster
            
            # Find samples belonging to this cluster
            cluster_mask = (torch.from_numpy(cluster_labels).to(self.device) == cluster_idx)
            
            # Combine cluster membership and unused status
            candidate_mask = cluster_mask & (~used_mask)
            
            candidate_indices = torch.where(candidate_mask)[0]
            
            if len(candidate_indices) > 0:
                # Find closest to center among candidates
                # cluster_centers[cluster_idx] is [D]
                # features[candidate_indices] is [M, D]
                
                # Use pre-calculated distances
                candidate_dists = dists[candidate_indices, cluster_idx]
                best_candidate_idx_in_subset = torch.argmin(candidate_dists)
                best_idx = candidate_indices[best_candidate_idx_in_subset]
                
                selected_indices.append(best_idx.item())
                used_mask[best_idx] = True
            else:
                # If cluster has no available samples (rare), select one closest to center from global remaining
                available_indices = torch.where(~used_mask)[0]
                if len(available_indices) > 0:
                    candidate_dists = dists[available_indices, cluster_idx]
                    best_candidate_idx_in_subset = torch.argmin(candidate_dists)
                    best_idx = available_indices[best_candidate_idx_in_subset]
                    selected_indices.append(best_idx.item())
                    used_mask[best_idx] = True

        selected_indices = torch.tensor(selected_indices, device=self.device)
        selected_images = images[selected_indices]

        return selected_images, selected_indices


class CovarianceMatchingFilter:
    """
    Selection method based on covariance matching
    Reference: Covariance_matching function
    """
    def __init__(
        self,
        reference_data: torch.Tensor,
        feature_extractor,
        device="cuda",
        pca_dim: int = 32,
        constraint_scale: bool = False
    ):
        self.reference_data = reference_data
        self.feature_extractor = feature_extractor
        self.device = device
        self.pca_dim = pca_dim
        self.constraint_scale = constraint_scale
        # Only PyTorch models need eval(), feature extractors might not
        if hasattr(self.feature_extractor, 'eval'):
            self.feature_extractor.eval()

        # Calculate covariance matrix of reference data
        with torch.no_grad():
            ref_features = self.feature_extractor(self.reference_data.to(device))
            if ref_features.dim() > 2:
                ref_features = ref_features.view(len(ref_features), -1)

            # Optional: Use PCA for dimensionality reduction
            if pca_dim is not None and ref_features.shape[1] > pca_dim:
                ref_features_np = ref_features.cpu().numpy()
                self.pca = PCA(n_components=pca_dim)
                ref_features_pca = self.pca.fit_transform(ref_features_np)
                ref_features = torch.from_numpy(ref_features_pca).to(device)
                self.use_pca = True
            else:
                self.use_pca = False
                pca_dim = ref_features.shape[1]

            # Calculate target covariance
            X = ref_features
            X_centered = X - X.mean(dim=0, keepdim=True)
            self.ref_cov = X_centered.t() @ X_centered / (X.shape[0] - 1)
            self.p = self.ref_cov.shape[0]

            if constraint_scale:
                self.ref_cov = self.ref_cov / torch.norm(self.ref_cov, p='fro')

    def select(self, images: torch.Tensor, selection_ratio: float = 0.3, used=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select samples with best covariance matching (greedy selection)
        Implementation reference: Covariance_matching function
        """
        with torch.no_grad():
            features = self.feature_extractor(images.to(self.device))
            if features.dim() > 2:
                features = features.view(len(features), -1)

            # Apply PCA (if used)
            if self.use_pca:
                features_np = features.cpu().numpy()
                features = torch.from_numpy(self.pca.transform(features_np)).to(self.device)

            num_select = int(len(images) * selection_ratio)
            selected_indices = []
            used = used if used is not None else torch.zeros(len(features), dtype=torch.bool, device=self.device)

            # Initialize: Randomly select one sample
            available_idx = torch.where(~used)[0]
            if len(available_idx) == 0:
                return images[[]], torch.tensor([], dtype=torch.long, device=self.device)

            random_candidate_idx = torch.randint(0, len(available_idx), (1,)).item()
            x = features[available_idx[random_candidate_idx]]
            selected_indices.append(available_idx[random_candidate_idx].item())
            used[available_idx[random_candidate_idx]] = True

            # Maintain cumulative statistics
            sum_x = x.clone()
            sum_xxT = x.unsqueeze(1) @ x.unsqueeze(0)
            selected_count = 1

            # Greedy selection of remaining samples
            for count_idx in range(1, num_select):
                available_idx = torch.where(~used)[0]
                if len(available_idx) == 0:
                    break

                candidates = features[available_idx]

                # Try covariance after adding each candidate
                S_try = sum_x.unsqueeze(0) + candidates
                S2_try = sum_xxT.unsqueeze(0) + candidates.unsqueeze(2) @ candidates.unsqueeze(1)
                n_try = selected_count + 1
                mean_try = S_try / n_try
                cov_try = S2_try / n_try - mean_try.unsqueeze(2) @ mean_try.unsqueeze(1)
                if n_try > 1:
                    cov_try = cov_try * n_try / (n_try - 1)

                scale_loss = 0
                if self.constraint_scale:
                    scale_loss = torch.norm(cov_try, p='fro', dim=(1,2))
                    cov_try = cov_try / torch.norm(cov_try, p='fro', dim=(1,2)).unsqueeze(1).unsqueeze(2)

                diff = cov_try - self.ref_cov.unsqueeze(0)
                frob_norms = torch.norm(diff, p='fro', dim=(1, 2))

                if self.constraint_scale:
                    loss = frob_norms - 1/10_000 * scale_loss
                else:
                    loss = frob_norms

                # Select minimum difference
                best_idx_in_batch = torch.argmin(loss)
                best_idx = available_idx[best_idx_in_batch]

                # Update cumulative statistics
                x = features[best_idx]
                sum_x += x
                sum_xxT += x.unsqueeze(1) @ x.unsqueeze(0)
                selected_count += 1
                selected_indices.append(best_idx.item())
                used[best_idx] = True

            selected_indices = torch.tensor(selected_indices, device=self.device)
            selected_images = images[selected_indices]

            return selected_images, selected_indices


class MatchingAlphaFilter:
    """
    Selection method based on matching alpha (Core of GEM)
    Reference: matching_alpha function
    """
    def __init__(
        self,
        reference_data: torch.Tensor,
        feature_extractor,
        device="cuda",
        pca_dim: int = 32,
        use_orig: bool = False
    ):
        self.reference_data = reference_data
        self.feature_extractor = feature_extractor
        self.device = device
        self.pca_dim = pca_dim
        self.use_orig = use_orig
        # Only PyTorch models need eval(), feature extractors might not
        if hasattr(self.feature_extractor, 'eval'):
            self.feature_extractor.eval()

        # Calculate statistics of reference data
        with torch.no_grad():
            ref_features = self.feature_extractor(self.reference_data.to(device))
            if ref_features.dim() > 2:
                ref_features = ref_features.view(len(ref_features), -1)

            orig_dim = ref_features.shape[1]
            if pca_dim is None:
                self.real_feats_pca = ref_features
                pca_dim = ref_features.shape[1]
            else:
                pca = PCA(n_components=pca_dim)
                self.real_feats_pca = torch.from_numpy(pca.fit_transform(ref_features.cpu().numpy())).to(device)
                if use_orig:
                    pca_dim = orig_dim

            self.n_2 = self.real_feats_pca.shape[0]
            cov_real = self._empirical_covariance_torch(self.real_feats_pca)
            eigvals, eigvecs = torch.linalg.eigh(cov_real)
            eigvals = torch.clamp(eigvals, min=1e-6)
            self.S_inv_sqrt = eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals)) @ eigvecs.t()

    def _empirical_covariance_torch(self, x):
        """Calculate empirical covariance"""
        x = x - x.mean(dim=0)
        cov = x.t() @ x / (x.shape[0] - 1)
        return cov

    def _fixed_point_eq(self, alpha, lambdas, n, p, n1):
        alpha = alpha.unsqueeze(-1)/n
        denom = lambdas * alpha + 1 - (p / n) - alpha
        lhs = torch.sum(1.0 / denom, dim=1)
        rhs = (p + n * alpha.squeeze(-1) - n1) / (1 - (p / n) - alpha.squeeze(-1))
        return lhs - rhs

    def _solve_alpha_batch(self, lambdas_batch, n, p, n1, tol=1e-5, max_iter=30):
        """
        lambdas_batch: [B, D]
        Returns: [B] (alpha solutions)
        """
        batch_size = lambdas_batch.shape[0]
        lower = torch.full((batch_size,), 1e-8, dtype=torch.float64, device=self.device)
        upper = torch.full((batch_size,), n - p - 1e-6, dtype=torch.float64, device=self.device)

        for _ in range(max_iter):
            mid = (lower + upper) / 2
            f_mid = self._fixed_point_eq(mid, lambdas_batch, n, p, n1)
            f_lower = self._fixed_point_eq(lower, lambdas_batch, n, p, n1)

            mask = f_mid * f_lower < 0
            upper = torch.where(mask, mid, upper)
            lower = torch.where(~mask, mid, lower)

            if torch.max(torch.abs(upper-lower)) < tol:
                break

        return mid/n

    def _calculate_alpha_estimate(self, eigvalues, n, n1, p, max_iter=30):
        alpha_solutions = self._solve_alpha_batch(eigvalues, n=n, p=p, n1=n1, max_iter=max_iter)
        return alpha_solutions

    def select(self, images: torch.Tensor, selection_ratio: float = 0.3, used=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select samples based on matching alpha
        Implementation reference: matching_alpha function
        """
        with torch.no_grad():
            features = self.feature_extractor(images.to(self.device))
            if features.dim() > 2:
                features = features.view(len(features), -1)

            if self.pca_dim is None:
                gen_feats_pca = features
                pca_dim = features.shape[1]
            else:
                pca = PCA(n_components=self.pca_dim)
                gen_feats_pca = torch.from_numpy(pca.fit_transform(features.cpu().numpy())).to(self.device)
                if self.use_orig:
                    pca_dim = features.shape[1]

            used = torch.zeros(len(gen_feats_pca), dtype=torch.bool, device=self.device) if used is None else used
            sum_x = torch.zeros(gen_feats_pca.shape[1], device=self.device)
            sum_xxT = torch.zeros(gen_feats_pca.shape[1], gen_feats_pca.shape[1], device=self.device)
            selected_count = 0
            matches = []

            num_select = int(len(images) * selection_ratio)

            for count_idx in range(num_select):
                available_idx = torch.where(~used)[0]
                if len(available_idx) == 0:
                    return images[torch.tensor(matches, device=self.device)], torch.tensor(matches, device=self.device)

                candidates = gen_feats_pca[available_idx]

                S_try = sum_x.unsqueeze(0) + candidates
                S2_try = sum_xxT.unsqueeze(0) + candidates.unsqueeze(2) @ candidates.unsqueeze(1)
                n_try = selected_count + 1
                mean_try = S_try / n_try
                cov_try = S2_try / n_try - mean_try.unsqueeze(2) @ mean_try.unsqueeze(1)

                if n_try > 1: cov_try = cov_try * n_try / (n_try - 1)
                product = torch.matmul(self.S_inv_sqrt.unsqueeze(0), cov_try)
                product = torch.matmul(product, self.S_inv_sqrt.unsqueeze(0))
                eigvals = torch.linalg.eigvalsh(product)
                alpha_values = self._calculate_alpha_estimate(eigvals, n=count_idx+1+self.n_2, n1=count_idx+1, p=pca_dim)

                best_idx_in_batch = torch.argmin(alpha_values)
                best_idx = available_idx[best_idx_in_batch]

                x = gen_feats_pca[best_idx]
                sum_x += x
                sum_xxT += x.unsqueeze(1) @ x.unsqueeze(0)
                used[best_idx] = True
                selected_count += 1
                matches.append(best_idx.item())

            selected_indices = torch.tensor(matches, device=self.device)
            selected_images = images[selected_indices]

            return selected_images, selected_indices



        self.n_clusters = n_clusters
        return self.select(images, selection_ratio, used)


class FIDFilter:
    """Selection method based on FID."""
    def __init__(self, reference_data: torch.Tensor, feature_extractor, device="cuda"):
        if not PYTORCH_FID_AVAILABLE:
            raise ImportError("FIDFilter requires pytorch_fid. Install with: pip install pytorch-fid")

        self.reference_data = reference_data
        self.feature_extractor = feature_extractor
        self.device = device

        # Pre-calculate statistics of reference data
        # Only PyTorch models need eval(), feature extractors might not
        if hasattr(self.feature_extractor, 'eval'):
            self.feature_extractor.eval()
        with torch.no_grad():
            # Process in batches to avoid OOM
            ref_features_list = []
            batch_size = 64 # Safe batch size
            
            for i in range(0, len(self.reference_data), batch_size):
                batch = self.reference_data[i:i+batch_size].to(device)
                features = self.feature_extractor(batch)
                
                # pytorch_fid InceptionV3 returns a list of tensors
                if isinstance(features, list):
                    features = features[0]
                
                if features.dim() > 2:
                    features = features.view(len(features), -1)
                ref_features_list.append(features.cpu())
            
            ref_features = torch.cat(ref_features_list, dim=0)

            self.ref_mu = ref_features.mean(dim=0).numpy()
            self.ref_sigma = np.cov(ref_features.numpy(), rowvar=False)

    def compute_fid_batch(self, images: torch.Tensor) -> float:
        """Calculate FID between a batch of images and reference data"""
        # Only PyTorch models need eval(), feature extractors might not
        if hasattr(self.feature_extractor, 'eval'):
            self.feature_extractor.eval()

        with torch.no_grad():
            features = self.feature_extractor(images.to(self.device))
            if features.dim() > 2:
                features = features.view(len(features), -1)

            # Calculate statistics of batch data
            batch_mu = features.mean(dim=0).cpu().numpy()
            batch_sigma = np.cov(features.cpu().numpy(), rowvar=False)

            # Calculate FID
            diff = self.ref_mu - batch_mu
            covmean = linalg.sqrtm(self.ref_sigma @ batch_sigma)

            if np.iscomplexobj(covmean):
                covmean = covmean.real

            fid = np.sum(diff ** 2) + np.trace(self.ref_sigma + batch_sigma - 2 * covmean)

        return float(fid)
    
    def select(self, images: torch.Tensor, selection_ratio: float = 0.3, 
               batch_size: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select batch with lowest FID"""
        # Calculate FID for image batches, select batch with lowest FID
        num_batches = len(images) // batch_size
        batch_fids = []
        batch_indices = []
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch = images[start_idx:end_idx]
            fid = self.compute_fid_batch(batch)
            batch_fids.append(fid)
            batch_indices.append((start_idx, end_idx))
        
        # Select batch with lowest FID
        num_select_batches = int(num_batches * selection_ratio)
        sorted_indices = sorted(range(len(batch_fids)), key=lambda i: batch_fids[i])
        selected_batch_indices = sorted_indices[:num_select_batches]
        
        # Collect selected images
        selected_indices_list = []
        for idx in selected_batch_indices:
            start, end = batch_indices[idx]
            selected_indices_list.extend(range(start, end))
        
        selected_indices = torch.tensor(selected_indices_list)
        selected_images = images[selected_indices]
        
        return selected_images, selected_indices


class LatentSpaceFilter:
    """
    Selection method based on latent space density estimation (LSF)
    Reference: Stabilizing Self-Consuming Diffusion Models with Latent Space Filtering
    Implementation reference: filter_utils.py -> filter_dataset_by_ole function
    """
    def __init__(self, encoder, device="cuda", filter_num_classes: int = 10, timesteps: List[int] = [200, 400, 600, 800]):
        self.encoder = encoder
        self.device = device
        self.filter_num_classes = filter_num_classes
        self.timesteps = timesteps
        self.encoder.eval()

    def get_bottleneck_features(self, model, images):
        """Get bottleneck features"""
        actual_model = model.module if hasattr(model, 'module') else model
        features = actual_model.bottleneck_features
        features = features.view(features.size(0), -1)  # [batch_size, feature_dim]
        return features

    def filter_dataset_by_ole(self, images: torch.Tensor, target_size: int, batch_size: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Filter dataset using OLE
        Reference: filter_dataset_by_ole function
        """
        dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(images),
            batch_size=batch_size,
            shuffle=False
        )

        current_idx = 0
        filter_scores = np.zeros(len(images))

        # compute scores for each batch
        for batch_idx, (batch_images,) in enumerate(dataloader):
            batch_images = batch_images.to(self.device)
            current_batch_size = batch_images.shape[0]

            batch_scores = []
            for t in self.timesteps:
                t_batch = torch.ones(current_batch_size, device=self.device) * t
                y_batch = torch.zeros_like(torch.zeros(current_batch_size, dtype=torch.long, device=self.device))

                with torch.no_grad():
                    self.encoder(batch_images, t_batch, y=y_batch)
                    features = self.get_bottleneck_features(self.encoder, batch_images)

                    # Max-pooling features on last two dimensions
                    if features.dim() > 2:
                        features = features.amax(dim=(-2, -1))

                    # Use SSC to get cluster labels
                    if OLE_AVAILABLE:
                        ssc = SparseSubspaceClusteringOMP(n_clusters=self.filter_num_classes)
                        batch_features = features.cpu().numpy()
                        batch_labels = torch.from_numpy(ssc.fit_predict(batch_features)).to(self.device)

                        # Calculate OLE loss
                        ole_loss = OLELoss.apply(features, batch_labels, self.device)
                        batch_scores.append(ole_loss.item())
                    else:
                        # Use k-NN density as alternative
                        nbrs = NearestNeighbors(n_neighbors=5, metric='euclidean')
                        nbrs.fit(features.cpu().numpy())
                        distances, _ = nbrs.kneighbors(features.cpu().numpy())
                        density = 1.0 / (distances[:, 1:].mean(axis=1) + 1e-10)
                        batch_scores.append(-density.mean().item())

            batch_score = torch.tensor(batch_scores).mean().item()
            filter_scores[current_idx:current_idx + current_batch_size] = batch_score
            current_idx += current_batch_size

        # Sort indices based on filter scores from smallest to largest (lower OLE loss is better)
        sorted_indices = torch.argsort(torch.tensor(filter_scores))
        filtered_indices = sorted_indices[:target_size]

        selected_images = images[filtered_indices]
        return selected_images, filtered_indices

    def select(self, images: torch.Tensor, selection_ratio: float = 0.3) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select samples with highest latent space density (using OLE loss)
        Reference: filter_dataset_by_ole function
        """
        target_size = int(len(images) * selection_ratio)
        return self.filter_dataset_by_ole(images, target_size)


class ThresholdDecayFilter:
    """
    Threshold Decay Filter (Diversity Filter)
    As described in 'A Closer Look at Model Collapse'
    Iteratively selects images that are far from the already selected ones.
    """
    def __init__(self, feature_extractor=None, device="cuda", initial_threshold: float = 60.0, decay_rate: float = 0.95):
        self.feature_extractor = feature_extractor
        self.device = device
        self.initial_threshold = initial_threshold
        self.decay_rate = decay_rate
        if self.feature_extractor and hasattr(self.feature_extractor, 'eval'):
            self.feature_extractor.eval()

    def select(self, images: torch.Tensor, selection_ratio: float = 0.3, used=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Selects samples based on distance threshold with decay.
        """
        num_select = int(len(images) * selection_ratio)
        
        with torch.no_grad():
            # If images are not features, extract them
            if self.feature_extractor:
                features = self.feature_extractor(images.to(self.device))
                if features.dim() > 2:
                    features = features.view(len(features), -1)
            else:
                # Assume images are already features if no extractor provided
                features = images.view(len(images), -1) if images.dim() > 2 else images
            
            features_np = features.cpu().numpy()
        
        selected_indices = []
        
        # We can pick the first one randomly or just the first available
        available_indices = np.where(~used.cpu().numpy())[0] if used is not None else np.arange(len(images))
        # Shuffle to ensure randomness in initial selection
        np.random.shuffle(available_indices)
        
        if len(available_indices) == 0:
             return images[[]], torch.tensor([], device=self.device)

        # Initialize pool with one random sample
        pool_indices = [available_indices[0]]
        selected_indices.append(available_indices[0])
        # Keep pool features as a list of arrays for easier appending, or vstack
        pool_features = [features_np[available_indices[0]]]
        
        threshold = self.initial_threshold
        
        # We iterate until we fill the buffer or run out of candidates
        while len(selected_indices) < num_select:
            added_in_this_round = False
            
            # Prepare pool for vectorization
            current_pool_features = np.array(pool_features)
            
            # Filter candidates that are not in selected_indices
            candidates = [idx for idx in available_indices if idx not in selected_indices]
            
            if not candidates:
                break
            
            # Check candidates
            # Optimization: check in chunks if needed, but here we do simple loop
            for idx in candidates:
                if len(selected_indices) >= num_select:
                    break
                
                feat = features_np[idx]
                # Compute distances to all in pool
                dists = np.linalg.norm(current_pool_features - feat, axis=1)
                
                if np.all(dists > threshold):
                    selected_indices.append(idx)
                    pool_features.append(feat)
                    # Update current_pool_features for next candidate? 
                    # Re-stacking every time is slow (O(N^2)). 
                    # But correct implementation requires checking against ALL currently selected.
                    current_pool_features = np.vstack([current_pool_features, feat])
                    added_in_this_round = True
            
            if not added_in_this_round and len(selected_indices) < num_select:
                # Decay threshold if no progress
                threshold *= self.decay_rate
                # print(f"Decaying threshold to {threshold:.2f}")
                
                if threshold < 1e-3:
                    break
        
        selected_indices = torch.tensor(selected_indices, device=self.device)
        return images[selected_indices], selected_indices
