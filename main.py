"""
Main program: Run experiments
"""
import argparse
import torch
import numpy as np
import random
import os
from config import get_config, get_config_from_args
from data.data_loader import (
    load_cifar10, load_celeba, load_cnn_dailymail, load_wikitext,
    split_data_non_iid, split_train_val_test, get_data_loader
)
from experiments.biased_verification_experiment import BiasedVerificationExperiment
from experiments.gem_experiment import GEMExperiment


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_ddpm_cifar10_model(config, device, torch_dtype, from_scratch=False):
    """Create DDPM CIFAR-10 model (google/ddpm-cifar10-32) - Use DDIM sampling"""
    try:
        from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel, DDIMScheduler, DDIMPipeline
        
        if from_scratch:
            print("Initializing DDPM model from scratch (Random Weights)...")
            # Load config without loading weights
            model_id = "google/ddpm-cifar10-32"
            unet_config = UNet2DModel.load_config(model_id)
            unet = UNet2DModel.from_config(unet_config)
            
            try:
                # Use DDIMScheduler for faster sampling
                scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
            except Exception as e:
                print(f"Warning: Failed to load scheduler from {model_id}: {e}")
                print("Using default DDIMScheduler configuration for CIFAR-10")
                # Use default CIFAR-10 configuration (beta_start=0.0001, beta_end=0.02, num_train_timesteps=1000)
                scheduler = DDIMScheduler(
                    num_train_timesteps=1000,
                    beta_start=0.0001,
                    beta_end=0.02,
                    beta_schedule="linear",
                    clip_sample=True,
                    prediction_type="epsilon"
                )
            
            pipeline = DDIMPipeline(unet=unet, scheduler=scheduler)
        else:
            print("Loading google/ddpm-cifar10-32 model (Pre-trained)...")
            # Load pre-trained DDPM pipeline
            # Note: The original is DDPMPipeline. We load it and convert to DDIMPipeline for sampling.
            ddpm_pipeline = DDPMPipeline.from_pretrained(
                "google/ddpm-cifar10-32",
                torch_dtype=torch.float16 if torch_dtype == "float16" else torch.float32
            )
            
            # Convert to DDIMPipeline
            scheduler = DDIMScheduler.from_config(ddpm_pipeline.scheduler.config)
            pipeline = DDIMPipeline(unet=ddpm_pipeline.unet, scheduler=scheduler)
            
            print("Converted DDPMPipeline to DDIMPipeline for faster sampling")
        
        pipeline.to(device)
        
        class DiffusionWrapper:
            def __init__(self, pipeline, device):
                self.pipeline = pipeline
                self.unet = pipeline.unet
                self.scheduler = pipeline.scheduler
                self.device = device
                
            def train(self):
                self.unet.train()
                
            def eval(self):
                self.unet.eval()
                
            def compute_loss(self, x, labels=None):
                # x is [B, 3, 32, 32]
                # Sample noise to add to the images
                noise = torch.randn(x.shape, device=x.device)
                bs = x.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, self.scheduler.config.num_train_timesteps, (bs,), device=x.device
                ).long()

                # Add noise to the clean images according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_images = self.scheduler.add_noise(x, noise, timesteps)

                # Predict the noise residual
                noise_pred = self.unet(noisy_images, timesteps).sample

                # Compute loss
                loss = torch.nn.functional.mse_loss(noise_pred, noise)
                return loss
            
            # Allow calling the instance directly
            def __call__(self, batch_size, num_inference_steps=None):
                return self.generate_images(batch_size, num_inference_steps)

            def generate_images(self, num_samples, num_inference_steps=None):
                # Use the pipeline for generation
                # DDIM pipeline uses num_inference_steps
                
                kwargs = {}
                if num_inference_steps is not None:
                     kwargs["num_inference_steps"] = num_inference_steps
                
                # DDIM/DDPM pipeline __call__ args: batch_size, output_type, etc.
                images = self.pipeline(
                    batch_size=num_samples,
                    output_type="tensor",
                    **kwargs
                ).images
                
                # Ensure return type is Tensor
                if isinstance(images, np.ndarray):
                    images = torch.from_numpy(images)
                    if images.ndim == 4 and images.shape[-1] == 3:
                        # NHWC -> NCHW
                        images = images.permute(0, 3, 1, 2)
                
                return images

        return DiffusionWrapper(pipeline, device)
    except ImportError:
        raise ImportError("diffusers not available. Install with: pip install diffusers")

def create_model(config, device, torch_dtype="float32", device_map="auto", experiment_type="biased_verification", dataset="cifar10", from_scratch=False):
    """
    Create model based on reference papers

    Reference papers:
    - High-dimensional Analysis: DiT-XL/2, UNet-DDPM
    - Stabilizing Self-Consuming: UNet with bottleneck features (ADM)
    """

    if experiment_type in ["biased_verification", "gem"]:
        # Image tasks - based on reference papers
        # Fix: Case-insensitive check for dataset names
        # Add STL-10 support
        if dataset.lower() in ["cifar10", "celeba", "mnist", "fashion_mnist", "ffhq", "stl10"]:
            try:
                # Prioritize loading DiT model (High-dimensional Analysis paper)
                if config.image_model_type == "dit":
                    model = create_dit_model(config, device, torch_dtype)
                elif config.image_model_type in ["unet", "ddpm"]:
                    # User Request: Don't use pretrained for CelebA/STL-10.
                    # Force from_scratch=True if dataset is CelebA or STL-10
                    # This reduces Round 0 performance but ensures fairness/no data leakage
                    
                    actual_from_scratch = from_scratch
                    if dataset.lower() in ["celeba", "stl10"]:
                        print(f"Forcing from_scratch=True for {dataset} as requested.")
                        actual_from_scratch = True
                    
                    if not actual_from_scratch:
                         # Use Google's DDPM CIFAR-10 model as a strong baseline
                         # This is a standard UNet DDPM, perfectly matching the reference paper settings (32x32)
                         # Avoids OOM issues caused by Stable Diffusion upsampling
                         print("Using google/ddpm-cifar10-32 (Standard DDPM UNet) as base model")
                         model = create_ddpm_cifar10_model(config, device, torch_dtype, from_scratch=False)
                    else:
                         # Initialize generic UNet from scratch
                         # We can reuse create_ddpm_cifar10_model logic but with from_scratch=True
                         # This will create a fresh UNet2DModel with config parameters
                         print(f"Initializing {dataset} model from scratch...")
                         model = create_ddpm_cifar10_model(config, device, torch_dtype, from_scratch=True)
                    
                elif config.image_model_type == "stable_diffusion":
                    # Use only when SD is explicitly requested
                    print("Using Stable Diffusion v1.5 (Warning: High VRAM usage for CIFAR-10)")
                    model = create_stable_diffusion_model(config, device, torch_dtype)
                else:
                    raise ValueError(f"Unknown image model type: {config.image_model_type}")

                print(f"Loaded {config.image_model_type.upper()} model for {dataset}")

            except Exception as e:
                print(f"Failed to load {config.image_model_type} model: {e}")
                print("Using simple ConvNet placeholder (for research prototyping)")
                # Create placeholder model for rapid prototyping
                unet = torch.nn.Sequential(
                    torch.nn.Conv2d(3, 64, 3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(64, 128, 3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(128, 3, 3, padding=1)
                ).to(device)

                class SimpleDiffusionModel:
                    def __init__(self, unet):
                        self.unet = unet

                    def train(self):
                        self.unet.train()

                    def eval(self):
                        self.unet.eval()

                    def compute_loss(self, x, labels=None):
                        """Simplified loss calculation"""
                        return torch.nn.functional.mse_loss(self.unet(x), x)

                    def generate_images(self, num_samples):
                        """Generate random noise as backup"""
                        print("Warning: Using SimpleDiffusionModel generate_images (random noise)")
                        # Generate random images [B, C, H, W]
                        samples = torch.randn(num_samples, 3, 32, 32).to(device)
                        return samples

                model = SimpleDiffusionModel(unet)

        # Text tasks - based on reference papers
        elif dataset in ["cnn_dailymail", "wikitext", "pile", "natural_instructions", "xlsum"]:
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM

                # GPT-2 configuration based on 'Beyond Model Collapse' paper
                if config.text_model_name:
                    model_name = config.text_model_name
                else:
                    model_name = "gpt2"  # Default to GPT-2 Small (117M)

                print(f"Loading GPT-2 model: {model_name}")
                
                # Convert string dtype to torch dtype
                dtype_map = {
                    "float16": torch.float16,
                    "bfloat16": torch.bfloat16,
                    "float32": torch.float32,
                    "auto": "auto"
                }
                torch_dtype_val = dtype_map.get(torch_dtype, torch.float32)

                if device_map == "auto":
                    print(f"Using device_map='auto' for {model_name}")
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name, 
                        device_map="auto", 
                        torch_dtype=torch_dtype_val
                    )
                else:
                    # Load directly to specified device without device_map='auto'
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch_dtype_val
                    )
                    model = model.to(device)
                
                tokenizer = AutoTokenizer.from_pretrained(model_name)

                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                model.tokenizer = tokenizer

                def compute_loss(self, input_ids, labels=None, attention_mask=None):
                    if labels is None:
                        labels = input_ids
                    outputs = self(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
                    return outputs.loss

                def generate_text(self, input_ids, max_length=50, num_return_sequences=1):
                    """Text generation method based on paper"""
                    outputs = self.generate(
                        input_ids=input_ids,
                        max_length=max_length,
                        num_return_sequences=num_return_sequences,
                        do_sample=True,
                        temperature=0.8,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                    return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

                model.compute_loss = compute_loss.__get__(model, type(model))
                model.generate_text = generate_text.__get__(model, type(model))

                print(f"Loaded GPT-2 model: {model_name}")

            except ImportError:
                print("transformers not available, using dummy text model")
                model = DummyTextModel().to(device)
            except Exception as e:
                print(f"Failed to load GPT-2 model: {e}")
                print("Using dummy text model")
                model = DummyTextModel().to(device)
        else:
            raise NotImplementedError(f"Dataset {dataset} not implemented")
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")

    # For HuggingFace models, if device_map='auto' is used, skip subsequent processing
    # as device_map='auto' already handles device distribution
    # Fix: Ensure hf_device_map is not None before accessing len()
    hf_device_map = getattr(model, 'hf_device_map', None)
    if hf_device_map is not None and device_map == "auto":
        print(f"Model automatically distributed across {len(hf_device_map)} devices")
    else:
        # Apply torch_dtype and device_map to other models
        if hasattr(model, 'parameters'):
            if torch_dtype != "float32":
                dtype_map = {
                    "float16": torch.float16,
                    "bfloat16": torch.bfloat16,
                    "auto": "auto"
                }
                if torch_dtype in dtype_map and torch_dtype != "auto":
                    model = model.to(dtype=dtype_map[torch_dtype])

            # Implement multi-GPU distribution (only for non-HuggingFace models)
            if device_map == "auto" and torch.cuda.device_count() > 1:
                print(f"Using DataParallel across {torch.cuda.device_count()} GPUs")
                # For wrapper classes like DiffusionModel, we need to wrap the unet part
                if hasattr(model, 'unet'):
                    model.unet = torch.nn.DataParallel(model.unet)
                else:
                    # For other model types
                    model = torch.nn.DataParallel(model)
            elif device_map == "auto":
                print("Single GPU mode (only 1 GPU available)")

    return model


def create_dit_model(config, device, torch_dtype):
    """Create DiT model (based on High-dimensional Analysis paper)"""
    try:
        from diffusers import DiTPipeline
        # DiT-XL/2 for CIFAR-10 (paper recommended config)
        model = DiTPipeline.from_pretrained("facebook/DiT-XL-2-32")  # 32x32 version for CIFAR-10
        model = model.to(device)
        return model
    except ImportError:
        raise ImportError("diffusers not available. Install with: pip install diffusers")


def create_unet_diffusion_model(config, device, torch_dtype, experiment_type):
    """Create UNet diffusion model (based on LSF paper)"""
    try:
        from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline

        # Improve model config to enhance generation quality and lower FID
        if experiment_type == "biased_verification":
            # User Request: "CelebA and STL-10 might directly take pre-trained models for the first round"
            # Attempt to load pre-trained model
            model = None
            if hasattr(config, 'image_model_name') and config.image_model_name:
                try:
                    print(f"Attempting to load pretrained model: {config.image_model_name}")
                    # Note: DDPMPipeline contains unet and scheduler
                    pipeline = DDPMPipeline.from_pretrained(config.image_model_name)
                    model = pipeline.unet
                    
                    # Check if input size matches
                    if model.sample_size != config.image_size:
                        print(f"Warning: Pretrained model size ({model.sample_size}) != Config size ({config.image_size})")
                        print("Initializing UNet from scratch due to size mismatch...")
                        model = None
                    else:
                        print("Successfully loaded pretrained UNet.")
                except Exception as e:
                    print(f"Could not load pretrained model: {e}")
                    model = None
            
            if model is None:
                # Enhanced config for biased verification experiment, improving generation quality
                model = UNet2DModel(
                    sample_size=config.image_size,
                    in_channels=3,
                    out_channels=3,
                    layers_per_block=2,  # Increase layers to improve expressivity
                    block_out_channels=(128, 128, 256, 256),  # Increase channels to improve quality
                    down_block_types=(
                        "DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"
                    ),
                    up_block_types=(
                        "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"
                    ),
                )
        else:
            # Full UNet config from LSF paper
            model = UNet2DModel(
                sample_size=config.image_size,
                in_channels=3,
                out_channels=3,
                layers_per_block=2,
                block_out_channels=(128, 128, 256, 256, 512, 512),
                down_block_types=(
                    "DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"
                ),
                up_block_types=(
                    "UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"
                ),
            )

        # Add diffusion scheduler
        scheduler = DDPMScheduler(num_train_timesteps=config.diffusion_steps)

        class DiffusionModel:
            def __init__(self, unet, scheduler):
                self.unet = unet
                self.scheduler = scheduler

            def train(self):
                self.unet.train()

            def eval(self):
                self.unet.eval()
            
            # Allow calling the instance directly (for biased_verification_experiment compatibility)
            def __call__(self, batch_size, num_inference_steps=None):
                return self.generate_images(batch_size, num_inference_steps)

            def compute_loss(self, x, labels=None):
                """Simplified loss calculation for diffusion model"""
                noise = torch.randn_like(x)
                timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (x.shape[0],), device=x.device)
                noisy_x = self.scheduler.add_noise(x, noise, timesteps)
                # Ensure timesteps are on the correct device
                timesteps = timesteps.to(x.device)
                # Ensure noisy_x is on the correct device (scheduler.add_noise might change device)
                noisy_x = noisy_x.to(x.device)
                noise_pred = self.unet(noisy_x, timesteps).sample
                # Ensure noise is on the same device as noise_pred
                noise = noise.to(noise_pred.device)
                return torch.nn.functional.mse_loss(noise_pred, noise)

            def generate_images(self, num_samples, num_inference_steps=None):
                """Generate image samples"""
                self.unet.eval()
                
                # Default to scheduler's max steps if not provided
                if num_inference_steps is None:
                    num_inference_steps = self.scheduler.config.num_train_timesteps

                # Initial noise
                sample_shape = (num_samples, 3, config.image_size, config.image_size)
                samples = torch.randn(sample_shape, device=self.unet.device)

                # Step-by-step denoising
                self.scheduler.set_timesteps(num_inference_steps)

                for t in self.scheduler.timesteps:
                    with torch.no_grad():
                        # Model predicts noise
                        model_output = self.unet(samples, t).sample

                        # Scheduler update
                        samples = self.scheduler.step(model_output, t, samples).prev_sample

                # Ensure output is in [0, 1] range
                samples = torch.clamp(samples, 0, 1)
                return samples

        diffusion_model = DiffusionModel(model, scheduler)
        # Move model to specified device
        diffusion_model.unet = diffusion_model.unet.to(device)
        return diffusion_model

    except ImportError:
        raise ImportError("diffusers not available. Install with: pip install diffusers")


def create_stable_diffusion_model(config, device, torch_dtype):
    """Create Stable Diffusion model"""
    try:
        from diffusers import StableDiffusionPipeline

        # Use smaller Stable Diffusion model
        model = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if torch_dtype == "float16" else torch.float32
        )
        model = model.to(device)
        
        class StableDiffusionWrapper:
            def __init__(self, pipeline, device):
                self.pipeline = pipeline
                self.unet = pipeline.unet
                self.vae = pipeline.vae
                self.scheduler = pipeline.scheduler
                self.text_encoder = pipeline.text_encoder
                self.tokenizer = pipeline.tokenizer
                self.device = device
                
                # Freeze VAE and Text Encoder
                self.vae.requires_grad_(False)
                self.text_encoder.requires_grad_(False)
                
                # Empty text embeddings for unconditional generation
                self.empty_text_embeddings = self._get_empty_text_embeddings()

            def _get_empty_text_embeddings(self):
                # Generate embeddings for empty string (unconditional)
                text_input = self.tokenizer(
                    [""], padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt"
                )
                with torch.no_grad():
                    text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
                return text_embeddings

            def train(self):
                self.unet.train()
                self.vae.eval()
                self.text_encoder.eval()
                # Enable gradient checkpointing to save memory
                if hasattr(self.unet, "enable_gradient_checkpointing"):
                    self.unet.enable_gradient_checkpointing()

            def eval(self):
                self.unet.eval()
                self.vae.eval()
                self.text_encoder.eval()
                # Disable gradient checkpointing
                if hasattr(self.unet, "disable_gradient_checkpointing"):
                    self.unet.disable_gradient_checkpointing()

            def compute_loss(self, x, labels=None):
                # x is [B, 3, H, W]
                # Resize to 256x256 for SD (512x512 might be too heavy for CIFAR upscaling)
                # But SD works best at 512, let's try 512 with smaller batch size first.
                # If OOM persists, we can lower this to 256 or 224.
                x_resized = torch.nn.functional.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False)
                
                # Encode images to latents
                with torch.no_grad():
                    latents = self.vae.encode(x_resized).latent_dist.sample()
                    latents = latents * self.vae.config.scaling_factor

                # Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                
                # Add noise to the latents
                noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = self.empty_text_embeddings.repeat(bsz, 1, 1)

                # Predict the noise residual
                if self.scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif self.scheduler.config.prediction_type == "v_prediction":
                    target = self.scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {self.scheduler.config.prediction_type}")

                model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

                loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="mean")
                return loss

            def generate_images(self, num_samples, num_inference_steps=50):
                # Use the pipeline for generation
                images = self.pipeline(
                    prompt=[""] * num_samples,
                    height=512,
                    width=512,
                    num_inference_steps=num_inference_steps,
                    output_type="tensor"
                ).images
                
                # Resize back to 32x32 for CIFAR
                images = torch.nn.functional.interpolate(images, size=(32, 32), mode='bilinear', align_corners=False)
                return images

        return StableDiffusionWrapper(model, device)
    except ImportError:
        raise ImportError("diffusers not available. Install with: pip install diffusers")


class DummyTextModel(torch.nn.Module):
    """Dummy text model for testing"""
    def __init__(self):
        super().__init__()
        self.vocab_size = 1000

    def compute_loss(self, input_ids, labels=None):
        return torch.tensor(1.0, requires_grad=True)

    def generate_text(self, num_samples):
        return [f"Generated text sample {i}" for i in range(num_samples)]


def main():
    # Set GPU memory management env vars to reduce fragmentation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'

    parser = argparse.ArgumentParser(description="Synthetic Data Collapse Experiments")
    parser.add_argument("--experiment_type", type=str, default="biased_verification",
                       choices=["biased_verification", "gem"],
                       help="Experiment type")
    parser.add_argument("--dataset", type=str, default="cifar10",
                       choices=["cifar10", "mnist", "fashion_mnist", "ffhq", "celeba", "stl10", "cnn_dailymail", "wikitext", "xlsum"],
                       help="Dataset: cifar10/mnist/fashion_mnist/ffhq/celeba (images), cnn_dailymail/wikitext/xlsum (text)")
    parser.add_argument("--num_clients", type=int, default=10,
                       help="Number of clients (GEM experiment only)")
    parser.add_argument("--num_iterations", type=int, default=10,
                       help="Number of iterations")
    parser.add_argument("--data_strategy", type=str, default="replace",
                       choices=["replace", "accumulate", "accumulate_subsample"],
                       help="Data strategy")
    parser.add_argument("--selection_ratio", type=float, default=0.5,
                       help="Selection ratio (Optimized: 0.5)")
    parser.add_argument("--selection_method", type=str, default="fid",
                       choices=["random", "center_matching", "center_sampling", "ds3", "kmeans", "covariance_matching", "matching_alpha", "latent_space", "fid", "rouge", "text_matching", "text_sampling", "perplexity", "bertscore", "no_synthetic", "real_upper_bound"],
                       help="Selection method")
    parser.add_argument("--biased_client_id", type=int, default=0,
                       help="Biased client ID (-1 for full dataset, unbiased)")
    parser.add_argument("--use_biased_verification", action="store_true", default=True,
                       help="Use biased verification set (True=Experiment Group, False=Control Group, Default=True)")
    parser.add_argument("--no_biased_verification", action="store_true",
                       help="Use unbiased verification set (Control Group, overrides --use_biased_verification)")
    parser.add_argument("--feature_extractor", type=str, default="dinov2",
                       choices=["resnet50", "clip", "dinov2"],
                       help="Feature extractor")
    parser.add_argument("--fid_model", type=str, default="inceptionv3",
                       choices=["inceptionv3", "clip"],
                       help="FID computation model")

    parser.add_argument("--device", type=str, default="cuda",
                       help="Device")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--use_wandb", action="store_true",
                       help="Use wandb")
    parser.add_argument("--torch_dtype", type=str, default="float32",
                       choices=["float16", "float32", "bfloat16", "auto"],
                       help="Model precision type")
    parser.add_argument("--device_map", type=str, default="auto",
                       choices=["auto", "sequential", "cpu"],
                       help="Device mapping strategy")
    parser.add_argument("--from_scratch", action="store_true",
                       help="Train model from scratch (random initialization) instead of using pre-trained weights")
    parser.add_argument("--learning_rate", type=float, default=None,
                       help="Learning rate (overrides config default)")
    parser.add_argument("--non_iid_alpha", type=float, default=0.5,
                       help="Dirichlet alpha for Non-IID split (<=0 for extreme/one-class)")

    args = parser.parse_args()
    
    set_seed(args.seed)
    
    config = get_config_from_args(args)

    if args.learning_rate is not None:
        config["model"].learning_rate = args.learning_rate
        
    # User Request: Pass non_iid_alpha to config
    config["data"].non_iid_alpha = args.non_iid_alpha

    print(f"Configuration loaded for {args.experiment_type} on {args.dataset}")

    # Handle validation set selection logic
    use_biased_verification = args.use_biased_verification
    if args.no_biased_verification:
        use_biased_verification = False
        print("Explicitly disabled biased verification (using unbiased validation set)")
    
    # Ensure base_save_dir is defined
    base_save_dir = config["experiment"].save_dir
    
    # User Request: Adjust epochs for STL-10
    if args.dataset == "stl10" and config["training"].num_epochs == 500:
         print("Adjusting epochs for STL-10 to 800 (User Recommendation)")
         config["training"].num_epochs = 800
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print("Loading dataset...")
    
    # Determine if we should use diffusion normalization ([-1, 1])
    # This is critical for Diffusion Models to work correctly and for FID to be valid
    is_diffusion_model = config["model"].image_model_type in ["ddpm", "unet", "dit", "stable_diffusion"]
    use_diffusion_norm = is_diffusion_model and args.experiment_type in ["biased_verification", "gem"]
    
    if use_diffusion_norm:
        print("Using Diffusion Normalization ([-1, 1]) for data loading")
    
    dataset_name = args.dataset.lower()
    if dataset_name == "cifar10":
        dataset = load_cifar10(config["data"].image_data_dir, train=True, use_diffusion_norm=use_diffusion_norm)
    elif dataset_name == "mnist":
        from data.data_loader import load_mnist
        dataset = load_mnist(config["data"].image_data_dir, train=True)
    elif dataset_name == "fashion_mnist":
        from data.data_loader import load_fashion_mnist
        dataset = load_fashion_mnist(config["data"].image_data_dir, train=True)
    elif dataset_name == "ffhq":
        from data.data_loader import load_ffhq
        # User Request: Redirect FFHQ to STL-10
        # STL-10 native is 96x96, but we resize to 64x64 in load_stl10
        dataset = load_ffhq(config["data"].image_data_dir, resolution=64, split="train")
    elif dataset_name == "stl10":
        from data.data_loader import load_stl10
        dataset = load_stl10(config["data"].image_data_dir, split="train")
    elif dataset_name == "celeba":
        dataset = load_celeba(config["data"].image_data_dir, split="train")
    elif dataset_name == "cnn_dailymail":
        dataset = load_cnn_dailymail(config["data"].text_data_dir, split="train")
    elif dataset_name == "xlsum":
        from data.data_loader import load_xlsum
        dataset = load_xlsum(config["data"].text_data_dir, split="train")
    elif dataset_name == "wikitext":
        dataset = load_wikitext(config["data"].text_data_dir, split="train")
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Split dataset (Train/Val/Test)
    # For CIFAR-10, if using full 50k training set, we need to adjust split logic
    # If config["data"].train_ratio == 1.0, skip split and use dataset as train_set
    # And attempt to load official test set as val/test set
    
    train_ratio = config["data"].train_ratio
    val_ratio = config["data"].val_ratio
    test_ratio = config["data"].test_ratio
    
    if args.dataset == "cifar10" and train_ratio >= 0.99:
        print("Using FULL CIFAR-10 Training Set (50k)...")
        train_dataset = dataset
        # Load official test set (10k)
        test_dataset_full = load_cifar10(config["data"].image_data_dir, train=False, use_diffusion_norm=use_diffusion_norm)
        # Split test set into val (5k) and test (5k) or just use test (10k)
        # For simplicity, val = test = full test set
        print("Using Official CIFAR-10 Test Set (10k) for Validation/Test...")
        val_dataset = test_dataset_full
        test_dataset = test_dataset_full
    elif args.dataset == "stl10" and train_ratio >= 0.99:
        # STL-10: Train (5k) + Unlabeled (100k) or just Train?
        # Standard supervised learning uses 'train' split (5000 images, 10 classes)
        # But 5000 is small. Maybe we should use 'train+unlabeled' if unsupervised?
        # Assuming standard 'train' split for now.
        print("Using FULL STL-10 Training Set (Limited to 50k)...")
        from data.data_loader import load_stl10
        # User Request: Limit to 50000
        train_dataset = load_stl10(config["data"].image_data_dir, split="train", max_samples=50000)
        
        test_dataset_full = load_stl10(config["data"].image_data_dir, split="test")
        val_dataset = test_dataset_full
        test_dataset = test_dataset_full
    elif args.dataset == "celeba" and train_ratio >= 0.99:
        print("Using FULL CelebA Training Set (Limited to 50k)...")
        # User Request: 限制到 50000
        train_dataset = load_celeba(config["data"].image_data_dir, split="train", max_samples=50000)
        
        # Load official valid/test splits
        val_dataset = load_celeba(config["data"].image_data_dir, split="valid")
        test_dataset = load_celeba(config["data"].image_data_dir, split="test")
        print(f"CelebA Splits: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    elif args.dataset == "mnist" and train_ratio >= 0.99:
        print("Using FULL MNIST Training Set (60k)...")
        train_dataset = dataset
        from data.data_loader import load_mnist
        test_dataset_full = load_mnist(config["data"].image_data_dir, train=False)
        val_dataset = test_dataset_full
        test_dataset = test_dataset_full
    else:
        # Default split logic
        train_dataset, val_dataset, test_dataset = split_train_val_test(
            dataset, 
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio
        )
    
    print(f"Data splits: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    print("Creating model...")
    model = create_model(config["model"], device, args.torch_dtype, args.device_map, args.experiment_type, args.dataset, from_scratch=args.from_scratch)
    
    # Determine learning rate - based on "A Closer Look at Model Collapse" paper
    if args.learning_rate is not None:
        lr = args.learning_rate
    elif args.from_scratch:
        lr = 1e-4  # From scratch training - paper standard
    else:
        lr = 1e-5  # Finetuning (Lower LR for stability)
    
    print(f"Using Learning Rate: {lr}")

    # Run experiment
    if args.experiment_type == "biased_verification":
        print("Running Biased Verification Experiment...")

        # Create experiment - pass full dataset, let experiment class handle bias logic
        # Explicitly pass dataset type to avoid internal detection failure
        is_image_dataset = args.dataset in ["cifar10", "celeba", "stl10", "ffhq", "mnist", "fashion_mnist"]
        
        experiment = BiasedVerificationExperiment(
            model=model,
            dataset=train_dataset,  # Pass full dataset, let experiment class handle bias
            global_test_dataset=test_dataset,
            biased_client_id=args.biased_client_id,
            selection_method=args.selection_method,
            selection_ratio=config["selection"].selection_ratio,
            data_strategy=config["training"].data_strategy,
            num_iterations=config["training"].num_iterations,
            device=device,
            use_wandb=config["experiment"].use_wandb,
            feature_extractor=args.feature_extractor,
            fid_model=args.fid_model,
            use_biased_verification=use_biased_verification,
            is_image_dataset=is_image_dataset,
            learning_rate=lr,
            bias_type=config["data"].bias_type,
            non_iid_alpha=config["data"].non_iid_alpha,
            num_epochs=config["training"].num_epochs,
            generation_multiplier=config["training"].generation_multiplier
        )

        # Check for checkpoint, resume if exists
        # Use same path generation logic as experiment class
        experiment_dir = experiment.experiment_dir
        checkpoint_path = f"{experiment_dir}/checkpoints/checkpoint_iter_latest.pth"
        
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            start_iteration, loaded_results = experiment.load_checkpoint(checkpoint_path)
            if start_iteration > 0:
                print(f"Resuming from iteration {start_iteration}")
                # Merge loaded results
                results = loaded_results
            else:
                results = None
        else:
            results = None

        # Run (resumes from next round if checkpoint exists)
        # Note: checkpoint_dir is now managed by experiment, no need to pass here
        if results is None:
            results = experiment.run()
        else:
            # Resume from checkpoint and continue running
            remaining_iterations = config["training"].num_iterations - start_iteration - 1
            if remaining_iterations > 0:
                # Create new experiment instance from checkpoint state
                additional_results = experiment.run()
                # Merge results
                for key in additional_results:
                    if key in results:
                        results[key].extend(additional_results[key])
                    else:
                        results[key] = additional_results[key]
        
        # Result saving logic is now handled by Experiment class, just print info here
        print("Experiment completed!")
        print(f"Results saved to {os.path.join(experiment_dir, 'results.json')}")
        print(f"Hyperparameters saved to {os.path.join(experiment_dir, 'hyperparameters.json')}")
        print(f"Experiment directory: {experiment_dir}")
        
    elif args.experiment_type == "gem":
        client_datasets = split_data_non_iid(
            train_dataset,
            num_clients=args.num_clients,
            alpha=config["data"].non_iid_alpha,
            bias_type=config["data"].bias_type
        )
        
        # Create experiment
        experiment = GEMExperiment(
            model=model,
            global_dataset=train_dataset,
            client_datasets=client_datasets,
            global_test_dataset=test_dataset,
            selection_ratio=config["selection"].selection_ratio,
            num_iterations=config["training"].num_iterations,
            device=device,
            use_wandb=config["experiment"].use_wandb
        )
        
        # Run
        results = experiment.run()
        
        # Reorganize GEM experiment result saving structure
        base_save_dir = config["experiment"].save_dir

        # 1. Create subfolder for algorithm name + data type
        algorithm_name = "gem"
        data_type = "images" if args.dataset in ["cifar10", "celeba"] else "text"
        algorithm_dir = os.path.join(base_save_dir, f"{algorithm_name}_{data_type}")
        os.makedirs(algorithm_dir, exist_ok=True)

        # 2. Name sub-subfolder using main hyperparameters
        num_clients = args.num_clients
        selection_ratio = config["selection"].selection_ratio
        num_iterations = config["training"].num_iterations
        non_iid_alpha = config["data"].non_iid_alpha
        bias_type = config["data"].bias_type

        subfolder_name = f"clients{num_clients}_ratio{selection_ratio}_iter{num_iterations}_alpha{non_iid_alpha}_{bias_type}"
        experiment_dir = os.path.join(algorithm_dir, subfolder_name)
        os.makedirs(experiment_dir, exist_ok=True)

        # Save experiment results
        import json
        result_path = os.path.join(experiment_dir, "results.json")
        with open(result_path, "w") as f:
            json.dump(results, f, indent=2)

        # Save all hyperparameters to a separate file
        hyperparameters = {
            "experiment_type": args.experiment_type,
            "dataset": args.dataset,
            "num_clients": args.num_clients,
            "selection_ratio": config["selection"].selection_ratio,
            "num_iterations": config["training"].num_iterations,
            "non_iid_alpha": config["data"].non_iid_alpha,
            "bias_type": config["data"].bias_type,
            "device": args.device,
            "seed": args.seed,
            "use_wandb": args.use_wandb,
            "torch_dtype": args.torch_dtype,
            "device_map": args.device_map
        }

        hyperparam_path = os.path.join(experiment_dir, "hyperparameters.json")
        with open(hyperparam_path, "w") as f:
            json.dump(hyperparameters, f, indent=2)
        
        print("Experiment completed!")
        print(f"Results saved to {result_path}")
        print(f"Hyperparameters saved to {hyperparam_path}")
        print(f"Experiment directory: {experiment_dir}")
    
    else:
        raise ValueError(f"Unknown experiment type: {args.experiment_type}")


if __name__ == "__main__":
    main()

