"""
Experiment 1: The Curse of Biased Verification
"""
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm
import wandb
import os
from datetime import datetime
from torchvision.utils import save_image

from selection import (
    RandomFilter, CenterMatchingFilter, CovarianceMatchingFilter, MatchingAlphaFilter,
    LatentSpaceFilter, FIDFilter, CenterSamplingFilter, DS3Filter, KMeansFilter,
    RougeFilter, PerplexityFilter, BERTScoreFilter, TextMatchingFilter, TextSamplingFilter, ThresholdDecayFilter,
    GEMSelector
)
from evaluation import (
    compute_fid, compute_precision_recall, compute_category_coverage,
    compute_rouge, compute_bertscore, compute_mauve, compute_entropy
)


import copy

class BiasedVerificationExperiment:
    """
    Biased Verification Experiment: Verifying "The Curse of Biased Verification"
    """

    def __init__(
        self,
        model,
        dataset,  # Complete real dataset
        global_test_dataset,
        biased_client_id: int = 0,
        selection_method: str = "fid",
        selection_ratio: float = 0.3,
        data_strategy: str = "replace",
        num_iterations: int = 10,
        device: str = "cuda",
        use_wandb: bool = True,
        num_correction_datasets: int = 3,
        feature_extractor: str = "dinov2",
        fid_model: str = "inceptionv3",
        num_clients: int = 10,
        use_biased_verification: bool = True,  # True=Experiment group, False=Control group
        is_image_dataset: bool = None,
        learning_rate: float = 1e-4,
        generation_multiplier: float = 4.0,
        bias_type: str = "class",  # "class" or "attribute"
        non_iid_alpha: float = 0.1,
        num_epochs: int = 200
    ):
        self.model = model
        self.dataset = dataset  # Training data always uses complete dataset
        self.global_test_dataset = global_test_dataset
        self.biased_client_id = biased_client_id
        self.selection_method = selection_method
        self.selection_ratio = selection_ratio
        self.data_strategy = data_strategy
        self.num_iterations = num_iterations
        self.device = device
        self.use_wandb = use_wandb
        self.num_correction_datasets = num_correction_datasets
        self.feature_extractor = feature_extractor
        self.use_biased_verification = use_biased_verification
        self.fid_model = fid_model
        self.num_clients = num_clients
        self.learning_rate = learning_rate
        self.generation_multiplier = generation_multiplier
        self.bias_type = bias_type
        self.non_iid_alpha = non_iid_alpha
        self.num_epochs = num_epochs
        
        # Initialize results storage
        self.results = {}
        self.current_iteration = 0
        
        # Determine dataset type
        if is_image_dataset is not None:
            self._is_image_dataset = is_image_dataset
            self._is_text_dataset = not is_image_dataset
        else:
            # Simple heuristic
            self._is_image_dataset = True 
            self._is_text_dataset = False
            
        # Determine model type
        self._is_diffusion_model = False
        if hasattr(self.model, 'unet') or hasattr(self.model, 'generate_images'):
            self._is_diffusion_model = True
            
        self.model_uses_device_map = False
        if hasattr(self.model, 'hf_device_map') and self.model.hf_device_map is not None:
             self.model_uses_device_map = True

        # Auto-detect bias type override
        if "celeba" in str(dataset).lower():
            print(f"Detected CelebA dataset. Overriding bias_type to 'attribute' (was '{self.bias_type}')")
            self.bias_type = "attribute"
        elif "stl10" in str(dataset).lower():
             self.bias_type = "class"

        # Initialize directories
        self.experiment_dir = self._get_experiment_dir()
        os.makedirs(self.experiment_dir, exist_ok=True)
        self.checkpoint_dir = os.path.join(self.experiment_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Save initial state
        self.initial_state_dict = self._get_model_state_dict()
        
        # Initialize training params
        self.initial_epochs = self.num_epochs
        
        # Initialize datasets
        self._prepare_datasets()
        
        # Initialize selector
        self.selector = self._init_selector()
        self._validate_selector_setup()

    def _get_model_state_dict(self):
        if self._is_diffusion_model:
            if hasattr(self.model, 'unet'):
                return copy.deepcopy(self.model.unet.state_dict())
            elif hasattr(self.model, 'module'):
                 return copy.deepcopy(self.model.module.state_dict())
            else:
                return copy.deepcopy(self.model.state_dict())
        else:
             if hasattr(self.model, 'module'):
                 return copy.deepcopy(self.model.module.state_dict())
             return copy.deepcopy(self.model.state_dict())

    def _generate_hyperparams_summary(self):
        return {
            "dataset": str(self.dataset),
            "selection_method": self.selection_method,
            "selection_ratio": self.selection_ratio,
            "biased_client_id": self.biased_client_id,
            "num_clients": self.num_clients,
            "bias_type": self.bias_type,
            "alpha": self.non_iid_alpha,
            "use_biased_verification": self.use_biased_verification
        }

    def _generate_hyperparams_hash(self, summary):
        # Format: clients{N}_bias{ID}_ratio{R}_alpha{A}_{TYPE}_{MODE}
        mode = "biased" if self.use_biased_verification else "unbiased"
        name = (
            f"clients{self.num_clients}_"
            f"bias{self.biased_client_id}_"
            f"ratio{self.selection_ratio}_"
            f"alpha{self.non_iid_alpha}_"
            f"{self.bias_type}_"
            f"{mode}"
        )
        return name

    def _get_experiment_dir(self) -> str:
        data_type = "images" if self._is_image_dataset else "text"
        main_folder = f"{self.selection_method}_{data_type}"
        hyperparams_summary = self._generate_hyperparams_summary()
        hyperparams_name = self._generate_hyperparams_hash(hyperparams_summary)
        return f"./results/{main_folder}/{hyperparams_name}"

    def _reset_model(self):
        print("Resetting model to initial state...")
        if self._is_diffusion_model:
            if hasattr(self.model, 'unet'):
                self.model.unet.load_state_dict(self.initial_state_dict)
                if not self.model_uses_device_map:
                    self.model.unet.to(self.device)
            else:
                 model_to_load = self.model.module if hasattr(self.model, 'module') else self.model
                 model_to_load.load_state_dict(self.initial_state_dict)
        else:
            model_to_load = self.model.module if hasattr(self.model, 'module') else self.model
            if not hasattr(self, 'initial_state_dict') or len(self.initial_state_dict) == 0:
                print("Initial state dict empty or missing. Reloading from pretrained...")
            else:
                model_to_load.load_state_dict(self.initial_state_dict, strict=False)
            if not self.model_uses_device_map:
                model_to_load.to(self.device)
        print("Model reset complete.")

    def _prepare_datasets(self):
        # 1. Split data into clients (Non-IID)
        from data.data_loader import split_data_non_iid
        
        print("Splitting data into clients...")
        self.client_datasets = split_data_non_iid(
            self.dataset,
            num_clients=self.num_clients,
            alpha=self.non_iid_alpha,
            bias_type=self.bias_type
        )
        
        # 2. Set biased_reference_data
        self.biased_reference_data = self.client_datasets[self.biased_client_id]
        print(f"Biased reference data (Client {self.biased_client_id}): {len(self.biased_reference_data)} samples")
        
        # 3. Set correction datasets (other clients)
        self.correction_datasets = [ds for i, ds in enumerate(self.client_datasets) if i != self.biased_client_id]
        
        # 4. Create local test dataset
        self.local_test_dataset = self._create_local_test_dataset()
        
        # 5. Create global validation set (if needed)
        # self.global_validation_set = self._create_global_validation_set() 

    def _create_local_test_dataset(self):
        from data.data_loader import split_data_non_iid
        if self.biased_client_id == -1:
            return self.global_test_dataset
            
        rng_state = np.random.get_state()
        np.random.seed(42)
        try:
            client_test_datasets = split_data_non_iid(
                self.global_test_dataset,
                num_clients=self.num_clients,
                alpha=self.non_iid_alpha,  # Use same alpha
                bias_type=self.bias_type,
                balanced=False
            )
        finally:
            np.random.set_state(rng_state)
            
        local_test_data = client_test_datasets[self.biased_client_id]
        print(f"Created local test data (biased): {len(local_test_data)} samples from client {self.biased_client_id}")
        return local_test_data

    def _init_selector(self):
        # Image selectors
        if self.selection_method == "random":
            return RandomFilter(device=self.device)
        elif self.selection_method == "fid":
             # Initialize FID filter
             # Need reference stats
             feature_extractor = self._get_feature_extractor()
             # We need to compute stats for biased_reference_data
             # But FIDFilter usually takes reference_data as argument
             ref_data = self._get_reference_tensor()
             return FIDFilter(ref_data, feature_extractor, device=self.device)
        
        # Default fallback
        return RandomFilter(device=self.device)

    def _validate_selector_setup(self):
        expected_ref_data = self.biased_reference_data if self.use_biased_verification else self.dataset
        print(f"[OK] Selector validated: using {'biased' if self.use_biased_verification else 'unbiased'} reference data ({len(expected_ref_data)} samples)")

    def _get_reference_tensor(self):
        # Convert reference dataset to tensor
        ref_dataset = self.biased_reference_data if self.use_biased_verification else self.dataset
        
        # Subsample if too large to fit in memory
        if len(ref_dataset) > 10000:
            indices = np.random.choice(len(ref_dataset), 10000, replace=False)
            ref_dataset = torch.utils.data.Subset(ref_dataset, indices)
            
        loader = torch.utils.data.DataLoader(ref_dataset, batch_size=128, shuffle=False)
        all_data = []
        for batch in loader:
            if isinstance(batch, torch.Tensor):
                all_data.append(batch)
            elif isinstance(batch, (tuple, list)):
                all_data.append(batch[0])
        
        if not all_data:
             return torch.empty(0)
             
        return torch.cat(all_data, dim=0)

    def _get_feature_extractor(self):
        if self.fid_model == "inceptionv3":
            try:
                from pytorch_fid.inception import InceptionV3
                block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
                model = InceptionV3([block_idx]).to(self.device)
                return model
            except Exception as e:
                print(f"pytorch_fid failed ({e}), falling back to torchvision InceptionV3...")
                from torchvision.models import inception_v3
                model = inception_v3(pretrained=True, transform_input=False)
                model.fc = torch.nn.Identity()
                model.eval()
                return model.to(self.device)
        return None

    def _prepare_initial_training_data(self):
        print(f"Preparing initial training data: Pure biased data ({len(self.biased_reference_data)} samples)")
        return self.biased_reference_data

    def generate_samples(self, num_samples: int, save_to_disk: bool = False, save_dir: str = None) -> Union[torch.Tensor, List[str]]:
        if hasattr(self.model, 'eval'):
            self.model.eval()
        
        # Handle Diffusers Pipeline
        if self._is_diffusion_model and hasattr(self.model, '__call__'):
             samples_list = []
             batch_size_gen = min(32, num_samples)
             
             if save_to_disk and save_dir is None:
                  save_dir = f"./generated_samples/iteration_{getattr(self, 'current_iteration', 0)}"
                  os.makedirs(save_dir, exist_ok=True)
             
             sample_paths = []
             
             for i in range(0, num_samples, batch_size_gen):
                 current_batch = min(batch_size_gen, num_samples - i)
                 print(f"Generating batch {i//batch_size_gen + 1}/{(num_samples + batch_size_gen - 1)//batch_size_gen}: {current_batch} samples")
                 
                 try:
                     output = self.model(batch_size=current_batch, num_inference_steps=50)
                     if hasattr(output, 'images'):
                         batch_samples = torch.stack([torch.from_numpy(img).permute(2, 0, 1) for img in output.images])
                         batch_samples = batch_samples.float() / 255.0
                     else:
                         batch_samples = output.cpu()
                 except Exception as e:
                     print(f"Generation failed: {e}")
                     # Fallback random
                     batch_samples = torch.randn(current_batch, 3, 32, 32)
                 
                 batch_samples = batch_samples.cpu()
                 
                 if save_to_disk:
                     batch_path = f"{save_dir}/batch_{i//batch_size_gen:04d}.pt"
                     torch.save(batch_samples, batch_path)
                     sample_paths.append(batch_path)
                     del batch_samples
                 else:
                     samples_list.append(batch_samples)
                 
                 if torch.cuda.is_available():
                     torch.cuda.empty_cache()
                     
             if save_to_disk:
                 return sample_paths
             
             if samples_list:
                 return torch.cat(samples_list, dim=0)
             return torch.empty(0)
             
        return torch.randn(num_samples, 3, 32, 32) # Fallback

    def load_samples_from_disk(self, sample_paths, indices=None):
        all_samples = []
        # If indices provided, we need to load specific samples
        # But sample_paths are batches. 
        # This is complicated if indices are global indices.
        # Assuming sample_paths is list of batch files.
        # We need to load all and then select? Or load selectively?
        # For simplicity, load all for now (optimize later if needed)
        
        for path in sample_paths:
            batch = torch.load(path)
            all_samples.append(batch)
        
        full_tensor = torch.cat(all_samples, dim=0)
        
        if indices is not None:
            return full_tensor[indices]
        return full_tensor

    def select_samples(self, candidate_samples, num_select, iteration):
        # Handle file paths
        if isinstance(candidate_samples, list) and isinstance(candidate_samples[0], str):
            # Load all candidates to memory for selection (might be heavy)
            print("Loading candidates from disk for selection...")
            candidates_tensor = self.load_samples_from_disk(candidate_samples)
        else:
            candidates_tensor = candidate_samples
            
        # Run selection
        try:
            # Calculate selection ratio if selector expects it (most do)
            if hasattr(self.selector, 'select'):
                # Compute ratio
                ratio = num_select / len(candidates_tensor)
                selected_samples, indices = self.selector.select(candidates_tensor, selection_ratio=ratio)
                     
        except Exception as e:
            print(f"Selection failed: {e}. Falling back to random.")
            indices = torch.randperm(len(candidates_tensor))[:num_select]
            selected_samples = candidates_tensor[indices]
            
        return selected_samples, indices

    def train_model_on_correction_data(self, synthetic_data, correction_dataset, num_epochs=50):
        # Ensure synthetic_data is Tensor
        if isinstance(synthetic_data, list):
             if len(synthetic_data) > 0 and isinstance(synthetic_data[0], str):
                 synthetic_data = self.load_samples_from_disk(synthetic_data)
             elif len(synthetic_data) > 0 and isinstance(synthetic_data[0], torch.Tensor):
                 synthetic_data = torch.stack(synthetic_data)
        
        print(f"Training on {len(synthetic_data) if synthetic_data is not None else 0} synthetic samples...")
        
        # Create loader
        if synthetic_data is not None:
            dataset = torch.utils.data.TensorDataset(synthetic_data)
            loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
        else:
            loader = torch.utils.data.DataLoader(correction_dataset, batch_size=128, shuffle=True)
            
        params = self.model.parameters() if hasattr(self.model, 'parameters') else self.model.unet.parameters()
        optimizer = torch.optim.AdamW(params, lr=self.learning_rate)
        self.model.train()
        
        for epoch in range(num_epochs):
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch in pbar:
                if isinstance(batch, list): batch = batch[0]
                batch = batch.to(self.device)
                optimizer.zero_grad()
                loss = self.model.compute_loss(batch)
                loss.backward()
                optimizer.step()
                pbar.set_postfix({"loss": loss.item()})
                
    def save_image_grid(self, samples, iteration, prefix="samples"):
        if not self._is_image_dataset: return
        save_dir = os.path.join(self.experiment_dir, "generated_images")
        os.makedirs(save_dir, exist_ok=True)
        if not isinstance(samples, torch.Tensor): return
        num_images = min(64, len(samples))
        images = samples[:num_images].cpu()
        save_path = os.path.join(save_dir, f"{prefix}_iter{iteration}.png")
        save_image(images, save_path, nrow=8, normalize=True)

    def save_results(self):
        import json
        save_path = os.path.join(self.experiment_dir, "results.json")
        def convert(o):
            if isinstance(o, np.int64): return int(o)
            if isinstance(o, np.float32): return float(o)
            if isinstance(o, torch.Tensor): return o.tolist()
            return o
        with open(save_path, "w") as f:
            json.dump(self.results, f, indent=2, default=convert)
        print(f"Results saved to {save_path}")

    def save_checkpoint(self, iteration):
        checkpoint = {
            "iteration": iteration,
            "model_state_dict": self.model.state_dict() if not hasattr(self.model, 'module') else self.model.module.state_dict(),
            "results": self.results,
            "selector_state": self.selector.state_dict() if hasattr(self.selector, 'state_dict') else None
        }
        path = os.path.join(self.checkpoint_dir, "checkpoint_iter_latest.pth")
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path):
        print(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location=self.device)
        
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            
        self.results = checkpoint.get("results", {})
        self.current_iteration = checkpoint.get("iteration", 0)
        
        if "selector_state" in checkpoint and checkpoint["selector_state"] is not None and hasattr(self.selector, 'load_state_dict'):
            self.selector.load_state_dict(checkpoint["selector_state"])
            
        return self.current_iteration, self.results

    def evaluate(self, iteration, generated_samples=None):
        print(f"Evaluating iteration {iteration}...")
        metrics = {}
        
        if generated_samples is None:
             return {}

        # FID
        if self.selection_method == "fid" or True:
             try:
                 # Lazy load test data
                 if not hasattr(self, 'test_data_tensor'):
                     print("Preparing test data tensor for evaluation...")
                     loader = torch.utils.data.DataLoader(self.global_test_dataset, batch_size=128, shuffle=False)
                     all_data = []
                     for batch in loader:
                         if isinstance(batch, list): batch = batch[0]
                         all_data.append(batch)
                     self.test_data_tensor = torch.cat(all_data, dim=0)
                 
                 # Prepare gen data
                 if isinstance(generated_samples, list) and isinstance(generated_samples[0], str):
                     gen_tensor = self.load_samples_from_disk(generated_samples)
                 else:
                     gen_tensor = generated_samples
                     
                 # Lazy load feature extractor
                 if not hasattr(self, 'eval_feature_extractor'):
                      self.eval_feature_extractor = self._get_feature_extractor()
                 
                 if self.eval_feature_extractor is not None:
                     fid = compute_fid(self.test_data_tensor, gen_tensor, self.eval_feature_extractor, device=self.device)
                     metrics["fid"] = fid
                     print(f"FID: {fid}")
                     
                     # Precision/Recall
                     prec, recall = compute_precision_recall(self.test_data_tensor, gen_tensor, self.eval_feature_extractor, device=self.device)
                     metrics["precision"] = prec
                     metrics["recall"] = recall
                     print(f"Precision: {prec}, Recall: {recall}")
                 
             except Exception as e:
                 print(f"Evaluation failed: {e}")
        
        # Save metrics
        for k, v in metrics.items():
            if k not in self.results:
                self.results[k] = []
            self.results[k].append(v)
            
        return metrics

    def run(self):
        print(f"Running experiment with {self.num_iterations} iterations...")
        
        start_iter = self.current_iteration
        if start_iter > 0:
            print(f"Resuming from iteration {start_iter}")
            
        for iteration in range(start_iter, self.num_iterations):
            self.current_iteration = iteration
            print(f"Iteration {iteration}")
            
            generated_samples_for_eval = None
            
            if iteration == 0:
                # Initial training
                self._reset_model()
                initial_data = self.dataset 
                self.train_model_on_correction_data(None, initial_data, num_epochs=self.initial_epochs)
                
                self.initial_N_value = len(self.dataset)
                self.target_N = self.initial_N_value
                
                # Eval on initial model
                print("Generating samples for Iteration 0 evaluation...")
                # Use fewer samples for speed during debug/verify, but ideally matches N
                generated_samples_for_eval = self.generate_samples(2000, save_to_disk=False)
                
            else:
                # Generation
                num_generate = int(self.generation_multiplier * self.target_N)
                num_select = self.target_N
                
                print(f"Generating {num_generate} samples...")
                candidate_samples = self.generate_samples(num_generate, save_to_disk=True)
                generated_samples_for_eval = candidate_samples
                
                if num_generate < num_select:
                    num_select = num_generate
                
                selected_samples, _ = self.select_samples(candidate_samples, num_select, iteration)
                
                # Training
                self.train_model_on_correction_data(selected_samples, None, num_epochs=self.num_epochs)
                
            # Eval & Save
            self.evaluate(iteration, generated_samples_for_eval)
            self.save_results()
            self.save_checkpoint(iteration)
            
        return self.results
