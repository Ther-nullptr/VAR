#!/bin/bash

# VAR Layer Skipping Evaluation Script
# This script provides convenient presets for different layer skipping scenarios

set -e

# Default paths (modify these according to your setup)
MODEL_PATH=""
VAE_PATH=""
OUTPUT_DIR="./layer_skip_results"
DEVICE="cuda"
BATCH_SIZE=16
NUM_SAMPLES=100

# Print usage information
print_usage() {
    echo "VAR Layer Skipping Evaluation Script"
    echo "Usage: $0 [OPTIONS] COMMAND"
    echo ""
    echo "Commands:"
    echo "  no-skip               Run without any layer skipping (baseline)"
    echo "  skip-last-stage-low   Skip layers 0,1,2 in the last stage"
    echo "  skip-last-stage-high  Skip layers 13,14,15 in the last stage"
    echo "  skip-early-layers     Skip layer 0 in all later stages (6-9)"
    echo "  skip-late-layers      Skip layers 14,15 in all later stages (7-9)"
    echo "  skip-middle-layers    Skip layers 7,8 in middle stages (4-6)"
    echo "  custom                Use custom layer skipping configuration"
    echo "  benchmark             Run speed benchmark with layer skipping"
    echo ""
    echo "Options:"
    echo "  --model_path PATH     Path to VAR model checkpoint"
    echo "  --vae_path PATH       Path to VAE checkpoint"
    echo "  --output_dir PATH     Output directory (default: ./layer_skip_results)"
    echo "  --device DEVICE       Device to use (default: cuda)"
    echo "  --batch_size SIZE     Batch size (default: 16)"
    echo "  --num_samples NUM     Number of samples (default: 100)"
    echo "  --model_depth DEPTH   Model depth: 16, 20, 24, or 30 (default: 16)"
    echo "  --stage_layer_skip    Custom stage-layer skip (format: \"stage1:layer1,layer2;stage2:layer3,layer4\")"
    echo "  --skip_layers_below   Skip layers below threshold (format: \"stage1:threshold1;stage2:threshold2\")"
    echo "  --skip_layers_above   Skip layers above threshold (format: \"stage1:threshold1;stage2:threshold2\")"
    echo "  --save_images         Save generated images"
    echo "  --help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Skip layers 0,1,2 in the last stage (stage 9)"
    echo "  $0 --model_path model.pth --vae_path vae.pth skip-last-stage-low"
    echo ""
    echo "  # Custom: Skip layers 0,1 in stage 8 and layers 0,1,2 in stage 9"
    echo "  $0 --model_path model.pth --vae_path vae.pth --stage_layer_skip \"8:0,1;9:0,1,2\" custom"
    echo ""
    echo "  # Skip all layers with index < 3 in the last stage"
    echo "  $0 --model_path model.pth --vae_path vae.pth --skip_layers_below \"9:3\" custom"
    echo ""
    echo "  # Benchmark different configurations"
    echo "  $0 --model_path model.pth --vae_path vae.pth benchmark"
}

# Parse command line arguments
COMMAND=""
MODEL_DEPTH="16"
CUSTOM_STAGE_LAYER_SKIP=""
CUSTOM_SKIP_LAYERS_BELOW=""
CUSTOM_SKIP_LAYERS_ABOVE=""
SAVE_IMAGES=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --vae_path)
            VAE_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
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
        --model_depth)
            MODEL_DEPTH="$2"
            shift 2
            ;;
        --stage_layer_skip)
            CUSTOM_STAGE_LAYER_SKIP="$2"
            shift 2
            ;;
        --skip_layers_below)
            CUSTOM_SKIP_LAYERS_BELOW="$2"
            shift 2
            ;;
        --skip_layers_above)
            CUSTOM_SKIP_LAYERS_ABOVE="$2"
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
        no-skip|skip-last-stage-low|skip-last-stage-high|skip-early-layers|skip-late-layers|skip-middle-layers|custom|benchmark)
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
COMMON_ARGS="--model_path \"$MODEL_PATH\" --vae_path \"$VAE_PATH\" --output_dir \"$OUTPUT_DIR\" --device $DEVICE --batch_size $BATCH_SIZE --num_samples $NUM_SAMPLES --model-depth $MODEL_DEPTH $SAVE_IMAGES"

echo "=============================================="
echo "VAR Layer Skipping Evaluation"
echo "=============================================="
echo "Model: $MODEL_PATH"
echo "VAE: $VAE_PATH"
echo "Command: $COMMAND"
echo "Model Depth: $MODEL_DEPTH"
echo "Output: $OUTPUT_DIR"
echo "Device: $DEVICE"
echo "=============================================="

case $COMMAND in
    no-skip)
        echo "Running without layer skipping (baseline)..."
        python var_evaluate_layer_skip.py $COMMON_ARGS \
            --disable_layer_skip \
            --generate --benchmark
        ;;
    
    skip-last-stage-low)
        echo "Skipping layers 0,1,2 in the last stage (stage 9)..."
        python var_evaluate_layer_skip.py $COMMON_ARGS \
            --stage_layer_skip "9:0,1,2" \
            --generate --benchmark
        ;;
    
    skip-last-stage-high)
        echo "Skipping layers 13,14,15 in the last stage (stage 9)..."
        python var_evaluate_layer_skip.py $COMMON_ARGS \
            --stage_layer_skip "9:13,14,15" \
            --generate --benchmark
        ;;
    
    skip-early-layers)
        echo "Skipping layer 0 in later stages (6,7,8,9)..."
        python var_evaluate_layer_skip.py $COMMON_ARGS \
            --stage_layer_skip "6:0;7:0;8:0;9:0" \
            --generate --benchmark
        ;;
    
    skip-late-layers)
        echo "Skipping layers 14,15 in later stages (7,8,9)..."
        python var_evaluate_layer_skip.py $COMMON_ARGS \
            --stage_layer_skip "7:14,15;8:14,15;9:14,15" \
            --generate --benchmark
        ;;
    
    skip-middle-layers)
        echo "Skipping layers 7,8 in middle stages (4,5,6)..."
        python var_evaluate_layer_skip.py $COMMON_ARGS \
            --stage_layer_skip "4:7,8;5:7,8;6:7,8" \
            --generate --benchmark
        ;;
    
    custom)
        echo "Running with custom layer skipping configuration..."
        CUSTOM_ARGS=""
        
        if [[ -n "$CUSTOM_STAGE_LAYER_SKIP" ]]; then
            CUSTOM_ARGS="$CUSTOM_ARGS --stage_layer_skip \"$CUSTOM_STAGE_LAYER_SKIP\""
            echo "Stage-layer skip: $CUSTOM_STAGE_LAYER_SKIP"
        fi
        
        if [[ -n "$CUSTOM_SKIP_LAYERS_BELOW" ]]; then
            CUSTOM_ARGS="$CUSTOM_ARGS --skip_layers_below \"$CUSTOM_SKIP_LAYERS_BELOW\""
            echo "Skip layers below: $CUSTOM_SKIP_LAYERS_BELOW"
        fi
        
        if [[ -n "$CUSTOM_SKIP_LAYERS_ABOVE" ]]; then
            CUSTOM_ARGS="$CUSTOM_ARGS --skip_layers_above \"$CUSTOM_SKIP_LAYERS_ABOVE\""
            echo "Skip layers above: $CUSTOM_SKIP_LAYERS_ABOVE"
        fi
        
        if [[ -z "$CUSTOM_ARGS" ]]; then
            echo "Error: For custom mode, specify at least one of --stage_layer_skip, --skip_layers_below, or --skip_layers_above"
            exit 1
        fi
        
        eval "python var_evaluate_layer_skip.py $COMMON_ARGS $CUSTOM_ARGS --generate --benchmark"
        ;;
    
    benchmark)
        echo "Running comprehensive benchmark of different layer skipping strategies..."
        
        # Benchmark baseline (no skipping)
        echo "=== Benchmarking: No layer skipping ==="
        python var_evaluate_layer_skip.py $COMMON_ARGS \
            --disable_layer_skip \
            --benchmark
        
        # Benchmark skip early layers in last stage
        echo "=== Benchmarking: Skip layers 0,1,2 in last stage ==="
        python var_evaluate_layer_skip.py $COMMON_ARGS \
            --stage_layer_skip "9:0,1,2" \
            --benchmark
        
        # Benchmark skip late layers in last stage
        echo "=== Benchmarking: Skip layers 13,14,15 in last stage ==="
        python var_evaluate_layer_skip.py $COMMON_ARGS \
            --stage_layer_skip "9:13,14,15" \
            --benchmark
        
        # Benchmark skip early layers in multiple stages
        echo "=== Benchmarking: Skip layer 0 in later stages ==="
        python var_evaluate_layer_skip.py $COMMON_ARGS \
            --stage_layer_skip "6:0;7:0;8:0;9:0" \
            --benchmark
        
        echo "=== Benchmark completed! ==="
        ;;
    
    *)
        echo "Unknown command: $COMMAND"
        print_usage
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo "Layer skipping evaluation completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================="

# Display quick summary if results files exist
echo ""
echo "Quick Summary:"
echo "=============================================="

for results_file in "$OUTPUT_DIR"/*/layer_skip_results.json; do
    if [[ -f "$results_file" ]] && command -v jq &> /dev/null; then
        config_name=$(basename $(dirname "$results_file"))
        echo "Configuration: $config_name"
        
        # Extract benchmark results if available
        if jq -e '.benchmark' "$results_file" > /dev/null 2>&1; then
            MEAN_TIME=$(jq -r '.benchmark.mean_time // "N/A"' "$results_file")
            THROUGHPUT=$(jq -r '.benchmark.throughput_img_per_sec // "N/A"' "$results_file")
            MEMORY=$(jq -r '.benchmark.mean_memory_mb // "N/A"' "$results_file")
            
            echo "  Generation Time: ${MEAN_TIME}s"
            echo "  Throughput: ${THROUGHPUT} images/sec"
            echo "  Memory Usage: ${MEMORY}MB"
        fi
        
        # Extract layer skip summary
        if jq -e '.parameters.layer_skip_config.skip_summary' "$results_file" > /dev/null 2>&1; then
            SKIP_SUMMARY=$(jq -r '.parameters.layer_skip_config.skip_summary // "N/A"' "$results_file")
            echo "  Layer Skip: $SKIP_SUMMARY"
        fi
        
        echo ""
    fi
done

echo "=============================================="