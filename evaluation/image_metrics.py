"""
Image evaluation metrics
"""
import torch
import numpy as np
from typing import Tuple, Dict, List

try:
    from pytorch_fid import fid_score
    PYTORCH_FID_AVAILABLE = True
except ImportError:
    PYTORCH_FID_AVAILABLE = False
    print("Warning: pytorch_fid not available. FID computation will be limited.")

from sklearn.metrics.pairwise import cosine_similarity
from scipy import linalg


def compute_fid(
    real_images: torch.Tensor,
    generated_images: torch.Tensor,
    feature_extractor,
    device: str = "cuda"
) -> float:
    """
    Compute Fréchet Inception Distance (FID)
    
    Args:
        real_images: Real images
        generated_images: Generated images
        feature_extractor: Feature extractor (e.g., InceptionV3)
        device: Device
    
    Returns:
        FID score
    """
    # Only PyTorch models need eval(), feature extractor might not
    if hasattr(feature_extractor, 'eval'):
        feature_extractor.eval()
    
    with torch.no_grad():
        # Extract features - batch processing to avoid OOM
        batch_size = 50
        
        real_features_list = []
        for i in range(0, len(real_images), batch_size):
            batch = real_images[i:i+batch_size].to(device)
            # Ensure input range is [0, 1] (pytorch_fid expects [0, 1])
            
            # Case 1: [0, 255] -> [0, 1]
            if batch.max() > 20.0:
                 batch = batch.float() / 255.0
            
            # Case 2: [-1, 1] -> [0, 1]
            elif batch.min() < -0.001:
                 batch = (batch + 1.0) / 2.0
                 batch = torch.clamp(batch, 0.0, 1.0)
            
            # Convert to 3 channels if single channel
            if batch.shape[1] == 1:
                batch = batch.repeat(1, 3, 1, 1)

            features = feature_extractor(batch)
            
            # pytorch_fid returns a list, take the first element
            if isinstance(features, list):
                features = features[0]
            
            # InceptionV3 returns [N, 2048, 1, 1] or [N, 2048]
            if len(features.shape) > 2:
                features = features.squeeze()
            if len(features.shape) == 1: # Handle batch size 1 case
                 features = features.unsqueeze(0)
                 
            real_features_list.append(features.cpu())
            # Clear GPU memory
            del batch, features
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        real_features = torch.cat(real_features_list, dim=0)
        
        gen_features_list = []
        for i in range(0, len(generated_images), batch_size):
            batch = generated_images[i:i+batch_size].to(device)
            
            # Case 1: [0, 255] -> [0, 1]
            if batch.max() > 20.0:
                 batch = batch.float() / 255.0
            
            # Case 2: [-1, 1] -> [0, 1]
            elif batch.min() < -0.001:
                 batch = (batch + 1.0) / 2.0
                 batch = torch.clamp(batch, 0.0, 1.0)

            if batch.shape[1] == 1:
                batch = batch.repeat(1, 3, 1, 1)
                
            features = feature_extractor(batch)

            if isinstance(features, list):
                features = features[0]

            if len(features.shape) > 2:
                features = features.squeeze()
            if len(features.shape) == 1: # Handle batch size 1 case
                 features = features.unsqueeze(0)

            gen_features_list.append(features.cpu())
            # Clear GPU memory
            del batch, features
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        gen_features = torch.cat(gen_features_list, dim=0)
        
        # Compute mean and covariance
        mu1 = real_features.numpy().mean(axis=0)
        sigma1 = np.cov(real_features.numpy(), rowvar=False)
        
        mu2 = gen_features.numpy().mean(axis=0)
        sigma2 = np.cov(gen_features.numpy(), rowvar=False)

        
        # Compute FID
        diff = mu1 - mu2
        covmean = linalg.sqrtm(sigma1 @ sigma2)
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = np.sum(diff ** 2) + np.trace(sigma1 + sigma2 - 2 * covmean)
    
    return float(fid)


def compute_precision_recall(
    real_images: torch.Tensor,
    generated_images: torch.Tensor,
    feature_extractor,
    k: int = 5,
    device: str = "cuda"
) -> Tuple[float, float]:
    """
    Compute Precision and Recall
    
    Precision: Proportion of generated samples falling on real data manifold
    Recall: Proportion of real samples covered by generated distribution
    
    Args:
        real_images: Real images
        generated_images: Generated images
        feature_extractor: Feature extractor (default: InceptionV3)
        k: k-NN parameter
        device: Device
    
    Returns:
        (precision, recall)
    """
    from sklearn.neighbors import NearestNeighbors
    
    # Only PyTorch models need eval(), feature extractor might not
    if hasattr(feature_extractor, 'eval'):
        feature_extractor.eval()
    
    # Check if feature extractor is InceptionV3 (FID default)
    # Papers usually use VGG-16 or InceptionV3 fc layer
    # Here we use the passed feature_extractor, usually InceptionV3
    
    with torch.no_grad():
        # Extract features - batch processing to avoid OOM
        batch_size = 50
        
        real_features_list = []
        for i in range(0, len(real_images), batch_size):
            batch = real_images[i:i+batch_size].to(device)
            
            # Case 1: [0, 255] -> [0, 1]
            if batch.max() > 20.0:
                 batch = batch.float() / 255.0
            # Case 2: [-1, 1] -> [0, 1]
            elif batch.min() < -0.001:
                 batch = (batch + 1.0) / 2.0
                 batch = torch.clamp(batch, 0.0, 1.0)

            features = feature_extractor(batch)
            if len(features.shape) > 2:
                features = features.squeeze()
            if len(features.shape) == 1: # Handle batch size 1 case
                features = features.unsqueeze(0)
            real_features_list.append(features.cpu())
            # Clear GPU memory
            del batch, features
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        real_features = torch.cat(real_features_list, dim=0).numpy()
        
        gen_features_list = []
        for i in range(0, len(generated_images), batch_size):
            batch = generated_images[i:i+batch_size].to(device)
            
            # Case 1: [0, 255] -> [0, 1]
            if batch.max() > 20.0:
                 batch = batch.float() / 255.0
            # Case 2: [-1, 1] -> [0, 1]
            elif batch.min() < -0.001:
                 batch = (batch + 1.0) / 2.0
                 batch = torch.clamp(batch, 0.0, 1.0)
                 
            features = feature_extractor(batch)
            if len(features.shape) > 2:
                features = features.squeeze()
            if len(features.shape) == 1: # Handle batch size 1 case
                features = features.unsqueeze(0)
            gen_features_list.append(features.cpu())
            # Clear GPU memory
            del batch, features
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        gen_features = torch.cat(gen_features_list, dim=0).numpy()
    
    # Precision: Find k nearest real samples for each generated sample
    # If distance is less than threshold, considered on manifold
    nbrs_real = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(real_features)
    distances_gen_to_real, _ = nbrs_real.kneighbors(gen_features)
    
    # Use average distance as threshold
    threshold = distances_gen_to_real.mean()
    precision = (distances_gen_to_real.mean(axis=1) < threshold).mean()
    
    # Recall: Find k nearest generated samples for each real sample
    nbrs_gen = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(gen_features)
    distances_real_to_gen, _ = nbrs_gen.kneighbors(real_features)
    
    threshold = distances_real_to_gen.mean()
    recall = (distances_real_to_gen.mean(axis=1) < threshold).mean()
    
    return float(precision), float(recall)


def compute_category_coverage(
    generated_images: torch.Tensor,
    classifier,
    num_classes: int,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Compute Category Coverage
    
    Args:
        generated_images: Generated images
        classifier: Classifier
        num_classes: Number of classes
        device: Device
    
    Returns:
        Dictionary containing coverage info
    """
    classifier.eval()
    
    with torch.no_grad():
        outputs = classifier(generated_images.to(device))
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()
    
    unique_classes, counts = np.unique(predictions, return_counts=True)
    
    covered_classes = len(unique_classes)
    coverage_ratio = covered_classes / num_classes
    
    # Compute entropy of class distribution (measure uniformity)
    class_probs = counts / counts.sum()
    entropy = -np.sum(class_probs * np.log(class_probs + 1e-10))
    max_entropy = np.log(num_classes)
    normalized_entropy = entropy / max_entropy
    
    return {
        "coverage_ratio": float(coverage_ratio),
        "covered_classes": int(covered_classes),
        "total_classes": num_classes,
        "entropy": float(entropy),
        "normalized_entropy": float(normalized_entropy),
        "class_distribution": dict(zip(unique_classes.tolist(), counts.tolist()))
    }

