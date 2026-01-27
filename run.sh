# ============================================================================
# Experiment 1: Biased Verification Experiment - Core Method Testing (Image & Text)
# Compare selection effects of biased vs unbiased real datasets
# ============================================================================

# Default to sequential run, can be set to parallel via argument
RUN_MODE="${1:-sequential}"

# Set process management
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/experiment_pids.txt"

# Function: Cleanup
cleanup() {
    echo -e "\n=== Received interrupt signal, terminating all experiment processes ==="

    # Read and terminate all recorded processes
    if [ -f "$PID_FILE" ]; then
        while read -r pid; do
            if [ -n "$pid" ] && kill -0 $pid 2>/dev/null; then
                echo "Terminating process $pid..."
                kill -TERM $pid 2>/dev/null || true
            fi
        done < "$PID_FILE"

        # Wait for processes to terminate
        sleep 3

        # Force kill remaining processes
        while read -r pid; do
            if [ -n "$pid" ] && kill -0 $pid 2>/dev/null; then
                echo "Force killing process $pid..."
                kill -KILL $pid 2>/dev/null || true
            fi
        done < "$PID_FILE"

        rm -f "$PID_FILE"
    fi

    echo "All experiment processes terminated"
    exit 1
}

# Function: Run Experiment
run_experiment() {
    local experiment_num="$1"
    local description="$2"
    shift 2

    echo "$experiment_num. $description"

    if [ "$RUN_MODE" = "parallel" ]; then
        # Parallel run: run in background
        "$@" &
        local pid=$!
        # Record PID
        echo $pid >> "$PID_FILE"
        echo "Process ID: $pid"
        # Wait a bit to avoid startup conflicts
        sleep 1
    else
        # Sequential run: run directly
        "$@"
    fi
}

# Set signal handlers
trap cleanup SIGINT SIGTERM

# Initialize files
rm -f "$PID_FILE"
touch "$PID_FILE"
echo "Process ID record file: $PID_FILE"

echo "=== Quick Verification Experiment - Core Method Testing ==="
if [ "$RUN_MODE" = "parallel" ]; then
    echo "Run Mode: Parallel - All experiments run simultaneously"
    echo "PIDs will be recorded to: $PID_FILE"
    echo "Use Ctrl+C to interrupt all experiments and clean up"
else
    echo "Run Mode: Sequential - Experiments run one by one"
    echo "Use Ctrl+C to interrupt current experiment"
fi

# ============================================================================
# Image Generation Model Experiments (CIFAR-10, CelebA, FFHQ)
# Methods: CovMatch, CenterMatch, K-means
# ============================================================================

# ---------------------------
# 1. CovMatch (Covariance Matching)
# ---------------------------
run_experiment "1a-1" "Image - CovMatch (Biased) - CIFAR-10..." \
    python main.py --experiment_type biased_verification \
                   --dataset cifar10 \
                   --selection_method covariance_matching \
                   --feature_extractor dinov2 \
                   --fid_model inceptionv3 \
                   --data_strategy accumulate_subsample \
                   --num_iterations 10 \
                   --from_scratch \
                   --non_iid_alpha 0.1

run_experiment "1a-2" "Image - CovMatch (Biased) - STL-10..." \
    python main.py --experiment_type biased_verification \
                   --dataset stl10 \
                   --selection_method covariance_matching \
                   --feature_extractor dinov2 \
                   --fid_model inceptionv3 \
                   --data_strategy accumulate_subsample \
                   --num_iterations 10 \
                   --from_scratch \
                   --non_iid_alpha 0.1                   

run_experiment "1a-3" "Image - CovMatch (Biased) - CelebA..." \
    python main.py --experiment_type biased_verification \
                   --dataset celeba \
                   --selection_method covariance_matching \
                   --feature_extractor dinov2 \
                   --fid_model inceptionv3 \
                   --data_strategy accumulate_subsample \
                   --num_iterations 10 \
                   --from_scratch \
                   --non_iid_alpha 0.1

run_experiment "1a-4" "Image - CovMatch (Biased) - CIFAR-10 (Alpha 0)..." \
    python main.py --experiment_type biased_verification \
                   --dataset cifar10 \
                   --selection_method covariance_matching \
                   --feature_extractor dinov2 \
                   --fid_model inceptionv3 \
                   --data_strategy accumulate_subsample \
                   --num_iterations 10 \
                   --from_scratch \
                   --non_iid_alpha 0                   

# ---------------------------
# 2. CenterMatch (Center Matching)
# ---------------------------
run_experiment "2a-1" "Image - CenterMatch (Biased) - CIFAR-10..." \
    python main.py --experiment_type biased_verification \
                   --dataset cifar10 \
                   --selection_method center_matching \
                   --feature_extractor dinov2 \
                   --fid_model inceptionv3 \
                   --data_strategy accumulate_subsample \
                   --num_iterations 10 \
                   --from_scratch \
                   --non_iid_alpha 0.0

run_experiment "2a-2" "Image - CenterMatch (Biased) - CIFAR-10 (Alpha 0.1)..." \
    python main.py --experiment_type biased_verification \
                   --dataset cifar10 \
                   --selection_method center_matching \
                   --feature_extractor dinov2 \
                   --fid_model inceptionv3 \
                   --data_strategy accumulate_subsample \
                   --num_iterations 10 \
                   --from_scratch \
                   --non_iid_alpha 0.1

run_experiment "2a-3" "Image - CenterMatch (Biased) - STL-10..." \
    python main.py --experiment_type biased_verification \
                   --dataset stl10 \
                   --selection_method center_matching \
                   --feature_extractor dinov2 \
                   --fid_model inceptionv3 \
                   --data_strategy accumulate_subsample \
                   --num_iterations 10 \
                   --from_scratch \
                   --non_iid_alpha 0.1
    

run_experiment "2a-4" "Image - CenterMatch (Biased) - CelebA..." \
    python main.py --experiment_type biased_verification \
                   --dataset celeba \
                   --selection_method center_matching \
                   --feature_extractor dinov2 \
                   --fid_model inceptionv3 \
                   --data_strategy accumulate_subsample \
                   --num_iterations 10 \
                   --from_scratch \
                   --non_iid_alpha 0.1



# Wait for all experiments to complete
echo ""
if [ "$RUN_MODE" = "parallel" ]; then
    echo "All experiments started in background, waiting for completion..."
    echo "PIDs recorded in: $PID_FILE"
    echo "To terminate all experiments, run: ./kill.sh" 

    # Monitor process status
    while true; do
        running_count=0
        if [ -f "$PID_FILE" ]; then
            while read -r pid; do
                if [ -n "$pid" ] && kill -0 $pid 2>/dev/null; then
                    running_count=$((running_count + 1))
                fi
            done < "$PID_FILE"
        fi

        if [ $running_count -eq 0 ]; then
            break
        fi

        echo "$running_count experiments still running..."
        sleep 10
    done

    # Cleanup files
    rm -f "$PID_FILE"
else
    echo "All experiments completed sequentially"
fi

echo "=== Quick Verification Experiment Completed ==="
echo "Results saved in ./results directory"
