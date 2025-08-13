#!/bin/bash

# Enhanced VAR Evaluation Script with Flexible Caching Control
# This script provides convenient presets for different evaluation scenarios

set -e

# Default paths (modify these according to your setup)
MODEL_PATH=""
VAE_PATH=""
OUTPUT_DIR="./enhanced_var_output"
DEVICE="cuda"
BATCH_SIZE=64
NUM_SAMPLES=50

# Print usage information
print_usage() {
    echo "Enhanced VAR Evaluation Script"
    echo "Usage: $0 [OPTIONS] COMMAND"
    echo ""
    echo "Commands:"
    echo "  no-cache              Run without any caching"
    echo "  original-cache        Use original VAR caching (169,256 skip; 100,169 cache)"
    echo "  aggressive-cache      Use aggressive caching (more stages)"
    echo "  conservative-cache    Use conservative caching (fewer stages)"
    echo "  attn-only-cache      Use attention-only caching"
    echo "  mlp-only-cache       Use MLP-only caching"
    echo "  adaptive-cache       Use adaptive threshold caching"
    echo "  calibrate            Run cache calibration"
    echo "  benchmark            Run speed benchmark"
    echo "  compare              Compare multiple cache configurations"
    echo "  custom               Use custom cache configuration"
    echo ""
    echo "Options:"
    echo "  --model_path PATH     Path to VAR model checkpoint"
    echo "  --vae_path PATH       Path to VAE checkpoint"
    echo "  --output_dir PATH     Output directory (default: ./enhanced_var_output)"
    echo "  --device DEVICE       Device to use (default: cuda)"
    echo "  --batch_size SIZE     Batch size (default: 16)"
    echo "  --num_samples NUM     Number of samples (default: 100)"
    echo "  --skip_stages LIST    Comma-separated skip stages (for custom mode)"
    echo "  --cache_stages LIST   Comma-separated cache stages (for custom mode)"
    echo "  --threshold VALUE     Cache similarity threshold (default: 0.7)"
    echo "  --save_images         Save generated images"
    echo "  --help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --model_path model.pth --vae_path vae.pth original-cache"
    echo "  $0 --model_path model.pth --vae_path vae.pth --save_images benchmark"
    echo "  $0 --model_path model.pth --vae_path vae.pth compare"
    echo "  $0 --model_path model.pth --vae_path vae.pth --skip_stages \"169,256\" --cache_stages \"100,169\" custom"
}

# Parse command line arguments
COMMAND=""
CUSTOM_SKIP_STAGES=""
CUSTOM_CACHE_STAGES=""
THRESHOLD=0.0
SAVE_IMAGES=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH=$2
            shift 2
            ;;
        --vae_path)
            VAE_PATH=$2
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR=$2
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --skip_stages)
            CUSTOM_SKIP_STAGES="$2"
            shift 2
            ;;
        --cache_stages)
            CUSTOM_CACHE_STAGES="$2"
            shift 2
            ;;
        --threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        --save_images)
            SAVE_IMAGES="--save_images"
            shift
            ;;
        --help|-h)
            print_usage
            exit 0
            ;;
        no-cache|original-cache|aggressive-cache|conservative-cache|attn-only-cache|mlp-only-cache|adaptive-cache|calibrate|benchmark|compare|custom)
            COMMAND="$1"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$MODEL_PATH" ]]; then
    echo "Error: --model_path is required"
    print_usage
    exit 1
fi

if [[ -z "$VAE_PATH" ]]; then
    echo "Error: --vae_path is required"
    print_usage
    exit 1
fi

if [[ -z "$COMMAND" ]]; then
    echo "Error: Command is required"
    print_usage
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Common arguments
COMMON_ARGS="--model_path $MODEL_PATH --vae_path $VAE_PATH --output_dir $OUTPUT_DIR --device $DEVICE --batch_size $BATCH_SIZE --num_samples $NUM_SAMPLES --cache_threshold $THRESHOLD $SAVE_IMAGES"

echo "=============================================="
echo "Enhanced VAR Evaluation"
echo "=============================================="
echo "Model: $MODEL_PATH"
echo "VAE: $VAE_PATH"
echo "Command: $COMMAND"
echo "Output: $OUTPUT_DIR"
echo "Device: $DEVICE"
echo "=============================================="

case $COMMAND in
    no-cache)
        echo "Running without caching..."
        python var_evaluate_enhanced.py $COMMON_ARGS \
            --skip_stages "" \
            --cache_stages "" \
            --generate --benchmark
        ;;
    
    original-cache)
        echo "Running with original VAR caching (skip: 169,256; cache: 100,169)..."
        python var_evaluate_enhanced.py $COMMON_ARGS \
            --skip_stages "169,256" \
            --cache_stages "100,169" \
            --enable_attn_cache --enable_mlp_cache \
            --generate --benchmark
        ;;
    
    aggressive-cache)
        echo "Running with aggressive caching (skip: 100,169,256; cache: 64,100,169)..."
        python var_evaluate_enhanced.py $COMMON_ARGS \
            --skip_stages "100,169,256" \
            --cache_stages "64,100,169" \
            --enable_attn_cache --enable_mlp_cache \
            --generate --benchmark
        ;;
    
    conservative-cache)
        echo "Running with conservative caching (skip: 256; cache: 169)..."
        python var_evaluate_enhanced.py $COMMON_ARGS \
            --skip_stages "256" \
            --cache_stages "169" \
            --enable_attn_cache --enable_mlp_cache \
            --generate --benchmark
        ;;
    
    attn-only-cache)
        echo "Running with attention-only caching..."
        python var_evaluate_enhanced.py $COMMON_ARGS \
            --skip_stages "169,256" \
            --cache_stages "100,169" \
            --enable_attn_cache --disable_mlp_cache \
            --generate --benchmark
        ;;
    
    mlp-only-cache)
        echo "Running with MLP-only caching..."
        python var_evaluate_enhanced.py $COMMON_ARGS \
            --skip_stages "169,256" \
            --cache_stages "100,169" \
            --disable_attn_cache --enable_mlp_cache \
            --generate --benchmark
        ;;
    
    adaptive-cache)
        echo "Running with adaptive threshold caching..."
        python var_evaluate_enhanced.py $COMMON_ARGS \
            --skip_stages "169,256" \
            --cache_stages "100,169" \
            --enable_attn_cache --enable_mlp_cache \
            --adaptive_threshold \
            --calibrate --generate --benchmark
        ;;
    
    calibrate)
        echo "Running cache calibration..."
        python var_evaluate_enhanced.py $COMMON_ARGS \
            --skip_stages "169,256" \
            --cache_stages "100,169" \
            --enable_attn_cache --enable_mlp_cache \
            --calibrate \
            --similarity_data_path "$OUTPUT_DIR/similarity_data.pth"
        ;;
    
    benchmark)
        echo "Running speed benchmark..."
        python var_evaluate_enhanced.py $COMMON_ARGS \
            --skip_stages "169,256" \
            --cache_stages "100,169" \
            --enable_attn_cache --enable_mlp_cache \
            --benchmark
        ;;
    
    compare)
        echo "Comparing multiple cache configurations..."
        python var_evaluate_enhanced.py $COMMON_ARGS \
            --compare_configs
        ;;
    
    custom)
        if [[ -z "$CUSTOM_SKIP_STAGES" ]] && [[ -z "$CUSTOM_CACHE_STAGES" ]]; then
            echo "Error: For custom mode, specify --skip_stages and/or --cache_stages"
            exit 1
        fi
        
        echo "Running with custom configuration..."
        echo "Skip stages: $CUSTOM_SKIP_STAGES"
        echo "Cache stages: $CUSTOM_CACHE_STAGES"
        
        python var_evaluate_enhanced.py $COMMON_ARGS \
            --skip_stages "$CUSTOM_SKIP_STAGES" \
            --cache_stages "$CUSTOM_CACHE_STAGES" \
            --enable_attn_cache --enable_mlp_cache \
            --generate --generate_fid
        ;;
    
    *)
        echo "Unknown command: $COMMAND"
        print_usage
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo "Evaluation completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================="

# Display quick summary if results file exists
RESULTS_FILE="$OUTPUT_DIR/enhanced_var_results.json"
if [[ -f "$RESULTS_FILE" ]] && command -v jq &> /dev/null; then
    echo ""
    echo "Quick Summary:"
    echo "=============================================="
    
    # Extract benchmark results if available
    if jq -e '.benchmark' "$RESULTS_FILE" > /dev/null 2>&1; then
        MEAN_TIME=$(jq -r '.benchmark.mean_time // "N/A"' "$RESULTS_FILE")
        THROUGHPUT=$(jq -r '.benchmark.throughput_img_per_sec // "N/A"' "$RESULTS_FILE")
        MEMORY=$(jq -r '.benchmark.mean_memory_mb // "N/A"' "$RESULTS_FILE")
        
        echo "Generation Time: ${MEAN_TIME}s"
        echo "Throughput: ${THROUGHPUT} images/sec"
        echo "Memory Usage: ${MEMORY}MB"
    fi
    
    # Extract comparison results if available
    if jq -e '.comparison' "$RESULTS_FILE" > /dev/null 2>&1; then
        echo ""
        echo "Configuration Comparison:"
        echo "$(jq -r '.comparison | to_entries[] | "\(.key): \(.value.performance.mean_time)s (\(.value.performance.throughput_img_per_sec) img/s)"' "$RESULTS_FILE")"
    fi
    
    echo "=============================================="
fi