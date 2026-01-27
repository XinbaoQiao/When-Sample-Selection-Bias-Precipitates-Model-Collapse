"""
Configuration File: Defines experiment parameters and hyperparameters.

Model configurations based on reference papers:

=== Image Model Configuration ===

1. High-dimensional Analysis of Synthetic Data Selection (2024)
   - Datasets: CIFAR-10 (32x32), CelebA (64x64)
   - Generative Model: DiT-XL/2 (ImageNet Pretrained)
   - Feature Extraction: ResNet-50 + PCA dimensionality reduction to 32 dimensions
   - Selection Methods: Center Matching, Covariance Matching, Matching Alpha

2. Stabilizing Self-Consuming Diffusion Models with Latent Space Filtering (2024)
   - Datasets: CIFAR-10, CelebA (gender/smiling), MNIST
   - Generative Model: UNet with bottleneck features (DDPM)
   - Training Parameters: batch_size=128, lr=1e-4, epochs=500, diffusion_steps=1000
   - Selection Methods: OLE Loss Clustering Filter, k-NN Density Estimation

=== Text Model Configuration ===

3. Beyond Model Collapse: Scaling Up with Synthesized Data Requires Verification (2024)
   - Dataset: XLSUM dataset (Hasan et al., 2021) - Multilingual summarization dataset
   - Models: GPT-2 Series (117M, 345M, 762M, 1.5B parameters)
   - Selection Methods: ROUGE Score, Perplexity, BERTScore

4. The Anti-Ouroboros Effect: Emergent Resilience in Large Language Models (2024)
   - Dataset: CNN/DailyMail (news summarization)
   - Models: GPT-2 Series (117M-1.5B parameters)
   - Training Strategy: Recursive Selective Feedback, RAG Enhancement

=== Hyperparameter Settings ===

Image Generation (Diffusion Models):
- Diffusion steps: 1000 (LSF Paper)
- Batch size: 128 (LSF Paper, High-dim Analysis)
- Learning rate: 1e-4 (LSF Paper)
- Epochs: 500 (LSF Paper)
- Image size: 32x32 (CIFAR-10), 64x64 (CelebA)

Text Generation (GPT-2):
- Model sizes: 117M, 345M, 762M, 1.5B (Beyond Collapse Paper)
- Batch size: 8 (GPT-2 Training Standard)
- Learning rate: 5e-5 (GPT-2 Fine-tuning)
- Max length: 1024 tokens
- Generation length: 50 tokens

Feature Extraction:
- ResNet: 18/50 layers (High-dim Analysis)
- CLIP: ViT-B/32 (High-dim Analysis, LSF)
- DINOv2: vitb14 (High-dim Analysis)
- PCA Dimensions: 32 (High-dim Analysis)
"""
import argparse
from dataclasses import dataclass
from typing import Optional


@dataclass
class DataConfig:
    """Data Configuration"""
    # Image Tasks
    image_dataset: str = "cifar10"  # "cifar10" or "celeba"
    image_data_dir: str = "./data"
    
    # Text Tasks (Based on reference paper: Beyond Model Collapse)
    # Option: "xlsum" (Only this dataset is supported, following paper config)
    text_dataset: str = "xlsum"  
    text_data_dir: str = "./data"
    
    # Bias Construction
    num_clients: int = 10
    non_iid_alpha: float = 0.5  # Dirichlet distribution parameter, smaller value means more Non-IID
    bias_type: str = "class"  # "class" or "attribute"
    
    # Data Splitting
    # Correction: Set training ratio to 1.0 to use the full CIFAR-10 dataset (50,000 images)
    # Validation and test sets will use the official test set (10,000 images) or be split from it
    # Corresponding logic in main.py is needed to handle test set as val/test
    train_ratio: float = 1.0 
    val_ratio: float = 0.0
    test_ratio: float = 0.0
    
    # User Request: Keep CIFAR-10 unchanged, adjust hyperparameters for CelebA and STL-10
    # Image size and training parameters will be dynamically adjusted in the main logic based on the dataset
    # CIFAR-10: 32x32
    # CelebA: 64x64, Non-IID alpha=0.5
    # STL-10: 64x64 (resized from 96x96), Non-IID alpha=0.1


@dataclass
class ModelConfig:
    """Model Configuration - Based on specific configurations from reference papers"""

    # ===== Image Generation Models =====
    # High-dimensional Analysis Paper Config (CIFAR-10)
    # Continue using CIFAR-10, CelebA, FFHQ + DDPM models
    image_model_type: str = "unet"  # "unet" for DDPM
    # Use pre-trained CIFAR-10 DDPM model as a starting point to avoid training from scratch
    image_model_name: str = "google/ddpm-cifar10-32"  
    image_size: int = 32  # Image resolution, 32x32 for CIFAR-10, 64x64 for CelebA/STL-10

    # LSF Paper Config (CelebA, CIFAR-10, MNIST)
    diffusion_steps: int = 1000  # Diffusion steps
    sampling_steps: int = 1000  # Sampling steps
    use_ddim: bool = False  # DDIM sampling

    # Training Parameters - Optimized based on "A Closer Look at Model Collapse" paper
    # CIFAR-10: 1e-4, 501 epochs, 500 warmup
    # CelebA/STL-10: Needs adjustment
    batch_size: int = 128  # Paper uses 128
    learning_rate: float = 1e-4  # Paper uses 1e-4 (reverted to standard value)
    epochs: int = 501  # Paper uses 501 epochs
    warmup_steps: int = 500  # Paper uses 500 warmup steps
    
    # Optimizer Configuration
    optimizer: str = "adamw"  # adamw, sgd
    weight_decay: float = 1e-6
    adam_beta1: float = 0.95
    adam_beta2: float = 0.999
    lr_scheduler: str = "cosine"  # cosine, linear, constant

    # LoRA Configuration
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_target_modules: list = None  # Default to all linear layers

    # ===== Text Generation Models =====
    # Beyond Model Collapse Paper Config (Strictly followed)
    # Model: Llama-2-7B (referenced in paper)
    # Use NousResearch version to avoid authentication issues, architecture is identical
    text_model_type: str = "llama"  # "gpt2", "llama", "gemma"
    text_model_name: str = "NousResearch/Llama-2-7b-hf"  # Llama-2-7B
    
    # Llama-2 Training Parameters (Beyond Model Collapse)
    text_batch_size: int = 32  # Updated per implementation details
    text_learning_rate: float = 5e-5  # Initial LR
    text_epochs: int = 1  # Updated per implementation details
    text_max_length: int = 1024  # 
    text_generation_length: int = 50  # 

    # LoRA Configuration (Text)
    text_use_lora: bool = True
    text_lora_rank: int = 16
    text_lora_alpha: int = 32
    text_lora_target_modules: list = None  # GPT-2: ["c_attn", "c_proj", "c_fc"]

    # ===== Feature Extraction Models =====
    # High-dimensional Analysis Paper Config
    feature_extractor_type: str = "resnet"  # "resnet", "clip", "dinov2"
    feature_extractor_name: str = "resnet50"  # resnet18, resnet50, clip-vit-b/32, dinov2_vitb14
    feature_dim: int = 32  # Dimensions after PCA reduction

    # CLIP Configuration
    clip_model_name: str = "ViT-B/32"  # ViT-B/32, ViT-L/14
    clip_batch_size: int = 64

    # ===== Classifier Models (For Selection) =====
    # LSF Paper Config
    classifier_type: str = "resnet"  # "resnet", "efficientnet", "vit"
    classifier_name: str = "resnet18"  # resnet18, resnet50

    # ===== Device and Precision =====
    torch_dtype: str = "float16"  # "float32", "float16", "bfloat16"
    device_map: str = "auto"  # "auto", "balanced", None
    max_memory: dict = None  # GPU memory limit
    
    # VRAM Optimization
    gradient_checkpointing: bool = True  # Enable gradient checkpointing to save significant VRAM
    gradient_accumulation_steps: int = 4  # Gradient accumulation allows for smaller batch_size


@dataclass
class SelectionConfig:
    """Selection Configuration"""
    # Selection Ratio - Optimized Config: Increase selection ratio to ensure quality
    selection_ratio: float = 0.5  # Top-50% (Increased from 30% to 50%)
    
    # Image Selection Methods
    image_selection_method: str = "fid"  # "entropy", "center_matching", "fid", "lsf"
    
    # Text Selection Methods
    text_selection_method: str = "rouge"  # "rouge", "perplexity", "bertscore"
    
    # GEM Configuration
    gem_regularization: float = 0.1
    diversity_weight: float = 0.5


@dataclass
class TrainingConfig:
    """Training Configuration"""
    # Iteration Settings
    num_iterations: int = 10
    num_generations_per_iter: int = 10000  # Optimized Config: Increased from 5000 to 10000 to improve data quality
    
    # Dynamic Data Generation Ratio Config (N = Training Set Size)
    # Number of candidate samples generated per round = generation_multiplier * N
    generation_multiplier: int = 4  # User Request: "Generate 4N synthetic data pool"
    
    # Training Parameters
    batch_size: int = 128
    learning_rate: float = 2e-4
    num_epochs: int = 500  # Default
    num_warmup_steps: int = 500
    
    # Data Strategy
    data_strategy: str = "replace"  # "replace", "accumulate", "accumulate_subsample"
    # User Request: "Fix all training DDPM datasets to n=50000... filter to n"
    # Revert to production value
    subsample_budget: Optional[int] = 50000


@dataclass
class EvaluationConfig:
    """Evaluation Configuration"""
    # Image Evaluation
    compute_fid: bool = True
    compute_precision_recall: bool = True
    compute_category_coverage: bool = True
    
    # Text Evaluation
    compute_rouge: bool = True
    compute_bertscore: bool = True
    compute_mauve: bool = True
    compute_entropy: bool = True
    compute_topic_shift: bool = True
    
    # Evaluation Frequency
    eval_frequency: int = 1  # Evaluate every N rounds


@dataclass
class ExperimentConfig:
    """Experiment Configuration"""
    experiment_name: str = "biased_verification"
    experiment_type: str = "biased_verification"  # "biased_verification" or "gem"
    
    # Experiment 1: Biased Verification
    biased_client_id: int = 0  # Which client to use as the biased verifier
    
    # Experiment 2: GEM
    gem_num_clients: int = 10
    gem_aggregation: str = "barycenter"  # "barycenter" or "simple_average"
    
    # Logging and Saving
    use_wandb: bool = True
    wandb_project: str = "synthetic_data_collapse"
    save_dir: str = "./results"
    log_dir: str = "./logs"
    
    # Device
    device: str = "cuda"
    num_workers: int = 4
    seed: int = 42


def get_config_from_args(args):
    """Build configuration from parsed command line arguments"""
    
    # Set configuration based on dataset type
    if hasattr(args, 'dataset') and args.dataset in ["cifar10", "celeba"]:
        data_config = DataConfig(
            image_dataset=args.dataset,
            num_clients=args.num_clients,
            non_iid_alpha=getattr(args, 'non_iid_alpha', 0.5)
        )
    else:
        data_config = DataConfig(
            text_dataset=getattr(args, 'dataset', 'cnn_dailymail'),
            num_clients=args.num_clients,
            non_iid_alpha=getattr(args, 'non_iid_alpha', 0.5)
        )
    
    training_config = TrainingConfig(
        num_iterations=args.num_iterations,
        data_strategy=args.data_strategy
    )
    
    selection_config = SelectionConfig(
        selection_ratio=args.selection_ratio
    )
    
    experiment_config = ExperimentConfig(
        experiment_name=getattr(args, 'experiment_name', 'exp'),
        experiment_type=args.experiment_type,
        device=args.device,
        seed=args.seed,
        use_wandb=args.use_wandb,
        save_dir=getattr(args, 'save_dir', './results')
    )
    
    return {
        "data": data_config,
        "model": ModelConfig(
            use_lora=getattr(args, 'use_lora', True),
            torch_dtype=getattr(args, 'torch_dtype', 'float32'),
            device_map=getattr(args, 'device_map', 'auto')
        ),
        "selection": selection_config,
        "training": training_config,
        "evaluation": EvaluationConfig(),
        "experiment": experiment_config
    }


def get_config():
    """Get default configuration (without parsing command line arguments)"""
    return {
        "data": DataConfig(),
        "model": ModelConfig(),
        "selection": SelectionConfig(),
        "training": TrainingConfig(),
        "evaluation": EvaluationConfig(),
        "experiment": ExperimentConfig()
    }
