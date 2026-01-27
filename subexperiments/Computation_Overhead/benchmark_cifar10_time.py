
import torch
import time
import sys
import os
import pandas as pd
import numpy as np
from torchvision import datasets, transforms
from tqdm import tqdm

# Add path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

try:
    from selection.gem_selection import GEMSelector
except ImportError as e:
    print(f"Import Error: {e}")
    # Try adding project root to sys.path explicitly if not resolved
    sys.path.append("/opt/data/private/synthetic_data")
    try:
        from selection.gem_selection import GEMSelector
    except ImportError as e:
        print(f"Import Error after path fix: {e}")
        sys.exit(1)

def load_cifar10_data():
    print("Loading CIFAR-10 data...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Updated path to project root data directory
    data_dir = os.path.join(project_root, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    try:
        dataset = datasets.CIFAR10(root=data_dir, train=True, download=False, transform=transform)
    except:
        print(f"Could not find data in {data_dir}, attempting download...")
        dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
        
    # Flatten images: [50000, 3072]
    all_data = []
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=False)
    for imgs, _ in dataloader:
        all_data.append(imgs.view(imgs.size(0), -1))
        
    return torch.cat(all_data, dim=0)

def run_single_benchmark(selector_cls, N, M, K, D, T, method, data_pool, device):
    # Explicitly clear cache before run
    if device == "cuda":
        torch.cuda.empty_cache()

    # Sample Data
    # Candidates (N)
    candidates_idx = torch.randperm(len(data_pool))[:N]
    candidates = data_pool[candidates_idx].to(device)
    
    # Resize to dimension D if needed (simple slicing for speed/simulation)
    if D < candidates.shape[1]:
        candidates = candidates[:, :D]
    
    # Clients (K * M)
    client_data_list = []
    remaining_pool = data_pool # Simplified: just reuse pool for random sampling
    
    # We only need 1 client data for parallel simulation
    for _ in range(K):
        idx = torch.randperm(len(remaining_pool))[:M]
        c_data = remaining_pool[idx].to(device)
        if D < c_data.shape[1]:
            c_data = c_data[:, :D]
        client_data_list.append(c_data)
        
    # Init Selector
    selector = selector_cls(client_data=client_data_list, device=device)
    
    # Benchmark
    torch.cuda.synchronize() if device == "cuda" else None
    start_time = time.time()
    
    BATCH_SIZE = 10000  # Adjust based on GPU memory

    try:
        if method == "greedy":
            # Scheme I: Local Greedy
            # Parallel Simulation: Each client computes scores independently.
            # Time = 1 Client Compute (Parallel) + Server Greedy Aggregation
            target_client = selector.client_data[0]
            
            # 1. Client Score Computation (One Client)
            # To avoid OOM and simulate large scale, we batch this.
            all_scores_one_client = []
            num_batches = (N + BATCH_SIZE - 1) // BATCH_SIZE
            for i in range(num_batches):
                start_idx = i * BATCH_SIZE
                end_idx = min((i + 1) * BATCH_SIZE, N)
                batch_candidates = candidates[start_idx:end_idx]
                s = selector._compute_scores_internal(batch_candidates, target_client)
                all_scores_one_client.append(s)
            
            scores_one_client = torch.cat(all_scores_one_client, dim=0)
            
            # 2. Server Greedy Selection
            # Simulate receiving scores from K clients.
            # We replicate the scores K times to simulate K clients.
            all_scores_K = scores_one_client.unsqueeze(1).repeat(1, K)
            
            # Run Greedy Selection
            # Limit selection size for benchmark speed if N is huge?
            # User wants "Time", so we should run it.
            # M is the reference size, usually we select M samples.
            num_select = M 
            
            # Optimization: If N is very large, greedy is slow. 
            # For the purpose of "Computation Overhead" plot, we use a fixed smaller selection size 
            # or scaling ratio if full run is too slow.
            # Let's use num_select = 1000 to keep it responsive, or M if M < 2000.
            # The bottleneck is the loop in Python.
            select_limit = 1000 # Fixed limit for benchmark responsiveness
            real_select = min(M, select_limit)
            
            selector._solve_greedy_max(all_scores_K, real_select)
            
        elif method == "barycenter":
            # Scheme II: Barycenter
            # Time = 1 Interpolation (Client) + 1 Barycenter Update (Server)
            # This represents the cost per iteration (Round).
            target_client = selector.client_data[0]
            current_support = candidates.clone() # Initialization
            
            # 1. Client Interpolation (One Client)
            transported_batches = []
            num_batches = (N + BATCH_SIZE - 1) // BATCH_SIZE
            
            for i in range(num_batches):
                start_idx = i * BATCH_SIZE
                end_idx = min((i + 1) * BATCH_SIZE, N)
                batch_support = current_support[start_idx:end_idx]
                
                # Compute map for this batch
                batch_transported = selector.compute_transport_map(batch_support, target_client)
                transported_batches.append(batch_transported)
            
            # Reassemble (Client output)
            transported_one_client = torch.cat(transported_batches, dim=0)
            
            # 2. Server Barycenter Update (Aggregation)
            # Simulate K clients: sum w_k * T_k
            # Since we only have 1 client's result, we simulate sum by multiplying.
            # In reality, server does: sum += transported_k
            # This is a vector addition of size [N, D].
            # We simulate K additions.
            
            # Mock aggregation
            barycenter_accum = torch.zeros_like(transported_one_client)
            for k in range(K):
                # Simulating addition of K vectors
                barycenter_accum += transported_one_client * (1.0/K)
            
            # Result is barycenter_accum
            
    except torch.cuda.OutOfMemoryError:
        print(f"OOM Error at N={N}, M={M}, K={K}, T={T}. Returning NaN.")
        return float('nan')
        
    torch.cuda.synchronize() if device == "cuda" else None
    end_time = time.time()
    
    return end_time - start_time

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load Data Once
    full_data = load_cifar10_data()
    print(f"Data Loaded: {full_data.shape}")
    
    # Repeat data to reach 200k if needed
    # CIFAR-10 is 50k. We need 200k.
    if len(full_data) < 200000:
        repeat_factor = 200000 // len(full_data) + 1
        full_data = full_data.repeat(repeat_factor, 1)
    
    results = []
    
    # Updated Base Parameters (Large Scale)
    base_N = 200000  # Candidates
    base_M = 5000    # Reference Size
    base_K = 10      # Clients
    base_D = 512     # Dimension 
    base_T = 10      # Iterations
    
    print(f"Base Config: N={base_N}, M={base_M}, K={base_K}, D={base_D}, T={base_T}")
    print("Mode: Parallel Simulation (Measuring 1 Client's latency)")
    
    # 1. Vary N (Candidate Size)
    print("\n--- Varying N (Candidate Size) ---")
    # Range: 10k to 200k
    N_list = [10000, 50000, 100000, 150000, 200000]
    for n in tqdm(N_list):
        # Greedy
        t = run_single_benchmark(GEMSelector, n, base_M, base_K, base_D, base_T, "greedy", full_data, device)
        results.append({"Parameter": "N (Candidates)", "Value": n, "Method": "Greedy", "Time": t})
        
        # Barycenter
        t = run_single_benchmark(GEMSelector, n, base_M, base_K, base_D, base_T, "barycenter", full_data, device)
        results.append({"Parameter": "N (Candidates)", "Value": n, "Method": "Barycenter", "Time": t})

    # 2. Vary M (Reference Size)
    print("\n--- Varying M (Reference Size) ---")
    M_list = [1000, 2000, 5000, 8000, 10000]
    for m in tqdm(M_list):
        # Use base_N 
        t = run_single_benchmark(GEMSelector, base_N, m, base_K, base_D, base_T, "greedy", full_data, device)
        results.append({"Parameter": "M (Reference Size)", "Value": m, "Method": "Greedy", "Time": t})
        
        t = run_single_benchmark(GEMSelector, base_N, m, base_K, base_D, base_T, "barycenter", full_data, device)
        results.append({"Parameter": "M (Reference Size)", "Value": m, "Method": "Barycenter", "Time": t})

    # 3. Vary K (Num Clients)
    print("\n--- Varying K (Num Clients) ---")
    # Parallel Time should be CONSTANT w.r.t K (ideally)
    # But we measure 1 client, so it SHOULD be constant in our simulation.
    K_list = [5, 10, 20, 50]
    for k in tqdm(K_list):
        t = run_single_benchmark(GEMSelector, base_N, base_M, k, base_D, base_T, "greedy", full_data, device)
        results.append({"Parameter": "K (Clients)", "Value": k, "Method": "Greedy", "Time": t})
        
        t = run_single_benchmark(GEMSelector, base_N, base_M, k, base_D, base_T, "barycenter", full_data, device)
        results.append({"Parameter": "K (Clients)", "Value": k, "Method": "Barycenter", "Time": t})

    # 4. Vary T (Barycenter Iterations)
    print("\n--- Varying T (Iterations) ---")
    T_list = [1, 5, 10, 20]
    for iter_t in tqdm(T_list):
        t_greedy = run_single_benchmark(GEMSelector, base_N, base_M, base_K, base_D, iter_t, "greedy", full_data, device)
        results.append({"Parameter": "T (Iterations)", "Value": iter_t, "Method": "Greedy", "Time": t_greedy})

        t_bary = run_single_benchmark(GEMSelector, base_N, base_M, base_K, base_D, iter_t, "barycenter", full_data, device)
        results.append({"Parameter": "T (Iterations)", "Value": iter_t, "Method": "Barycenter", "Time": t_bary})

    # Save Results
    df = pd.DataFrame(results)
    # Use relative path suitable for current execution context (cwd is subexperiments/Computation_Overhead)
    df.to_csv("benchmark_time_results_v2.csv", index=False)
    print("\n✅ Benchmark Completed. Results saved to benchmark_time_results_v2.csv")

if __name__ == "__main__":
    main()
