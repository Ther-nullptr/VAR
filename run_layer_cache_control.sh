#!/bin/bash

# VAR Fine-Grained Layer Cache Control Evaluation Script
# This script provides comprehensive control over layer-level caching in VAR model generation

set -e  # Exit on any error

# Default configuration
MODEL_PATH=""
VAE_PATH=""
MODEL_DEPTH=16
NUM_SAMPLES=50000
BATCH_SIZE=100
CLASS_NUM=1000
CFG=4.0
SEED=0
TOP_K=900
TOP_P=0.96
TEMPERATURE=1.0
MORE_SMOOTH=false

# Fine-grained layer cache control parameters
LAYER_CACHE_SPEC=""
CACHE_ATTN_LAYERS=""
CACHE_MLP_LAYERS=""
LAYER_BLEND_RATIOS=""
INTERPOLATION_MODE="bilinear"

# Legacy cache parameters
SKIP_STAGES=""
CACHE_STAGES=""
ENABLE_ATTN_CACHE=true
ENABLE_MLP_CACHE=true
THRESHOLD=0.7
MAX_SKIP_STAGES=9
ADAPTIVE_THRESHOLD=false

# Output parameters
OUTPUT_DIR="generated_images"
SAVE_IMAGES=true
SAVE_NPY=false

# Evaluation parameters
COMPUTE_FID=false
REAL_IMG_DIR=""
FID_CACHE_FILE=""

# Performance parameters
DEVICE="cuda"
NUM_WORKERS=8

# Display help function
show_help() {
    cat << EOF
VAR Fine-Grained Layer Cache Control Evaluation Script

Usage: $0 [OPTIONS]

Required Arguments:
  --model_path PATH         Path to VAR checkpoint
  --vae_path PATH          Path to VAE checkpoint

Model Architecture Arguments:
  --model_depth N          Model depth: 16, 20, 24, or 30 (default: 16)

Basic Generation Arguments:
  --num_samples N          Number of samples to generate (default: 50000)
  --batch_size N           Batch size for generation (default: 100)
  --class_num N            Number of classes (default: 1000)
  --cfg FLOAT              Classifier-free guidance scale (default: 4.0)
  --seed N                 Random seed (default: 0)
  --top_k N                Top-k sampling (default: 900)
  --top_p FLOAT            Top-p sampling (default: 0.96)
  --temperature FLOAT      Sampling temperature (default: 1.0)
  --more_smooth BOOL       Enable more smooth sampling (default: false)

Fine-Grained Layer Cache Control (NEW):
  --layer_cache_spec SPEC  Fine-grained layer cache specification
                          Format: "stage1:layer1:mode1,layer2:mode2;stage2:layer3:mode3"
                          Example: "9:0:cache,1:cache,2:cache;8:0:cache,15:cache"
                          Modes: cache, skip, normal
                          
  --cache_attn_layers SPEC Attention layer cache specification
                          Format: "stage1:layer1,layer2;stage2:layer3,layer4"
                          Example: "9:0,1,2;8:0,15"
                          
  --cache_mlp_layers SPEC  MLP layer cache specification
                          Format: "stage1:layer1,layer2;stage2:layer3,layer4"
                          Example: "9:0,1,2;8:0,15"
                          
  --layer_blend_ratios SPEC Layer blend ratio specification
                          Format: "stage1:layer1:ratio1,layer2:ratio2;stage2:layer3:ratio3"
                          Example: "9:0:0.8,1:0.6;8:0:0.7"
                          Ratios: 0.0 (pure computation) to 1.0 (pure cache)
                          
  --interpolation_mode MODE Interpolation mode for feature upsampling
                          Options: linear, bilinear, bicubic (default: bilinear)

Legacy Cache Control:
  --skip_stages LIST       Comma-separated list of stages to skip (e.g., "169,256")
  --cache_stages LIST      Comma-separated list of stages to cache (e.g., "100,169")
  --enable_attn_cache BOOL Enable attention layer caching (default: true)
  --enable_mlp_cache BOOL  Enable MLP layer caching (default: true)
  --threshold FLOAT        Similarity threshold for cache reuse (default: 0.7)
  --max_skip_stages N      Maximum number of stages to skip (default: 9)
  --adaptive_threshold BOOL Use adaptive thresholding (default: false)

Output Arguments:
  --output_dir DIR         Directory to save generated images (default: generated_images)
  --save_images BOOL       Whether to save generated images (default: true)
  --save_npy BOOL          Whether to save as numpy arrays (default: false)

Evaluation Arguments:
  --compute_fid BOOL       Compute FID score (default: false)
  --real_img_dir DIR       Path to real images for FID computation
  --fid_cache_file PATH    Path to FID cache file

Performance Arguments:
  --device DEVICE          Device to use (default: cuda)
  --num_workers N          Number of dataloader workers (default: 8)

Examples:

1. Basic generation with fine-grained layer control:
   $0 --model_path /path/to/var.pth --vae_path /path/to/vae.pth \\
      --layer_cache_spec "9:0:cache,1:cache;8:15:cache" \\
      --num_samples 1000 --batch_size 50

2. Fine-grained control with blend ratios:
   $0 --model_path /path/to/var.pth --vae_path /path/to/vae.pth \\
      --layer_cache_spec "9:0:cache,1:cache,2:cache" \\
      --layer_blend_ratios "9:0:0.8,1:0.6,2:0.9" \\
      --interpolation_mode bicubic

3. Mixed fine-grained and legacy cache control:
   $0 --model_path /path/to/var.pth --vae_path /path/to/vae.pth \\
      --layer_cache_spec "9:0:cache;8:15:cache" \\
      --skip_stages "169,256" --cache_stages "100,169"

4. Generate images and compute FID:
   $0 --model_path /path/to/var.pth --vae_path /path/to/vae.pth \\
      --layer_cache_spec "9:0:cache,1:cache,2:cache" \\
      --compute_fid true --real_img_dir /path/to/real/images \\
      --num_samples 10000

Stage Information:
  VAR stages correspond to different resolution levels:
  Stage 0: 1x1     (L=1)      Stage 5: 6x6     (L=36)
  Stage 1: 2x2     (L=4)      Stage 6: 8x8     (L=64)  
  Stage 2: 3x3     (L=9)      Stage 7: 10x10   (L=100)
  Stage 3: 4x4     (L=16)     Stage 8: 13x13   (L=169)
  Stage 4: 5x5     (L=25)     Stage 9: 16x16   (L=256)

Layer Information:
  VAR typically has 16 transformer layers (0-15)
  Each layer has both attention and MLP components
  Fine-grained control allows specifying both stage AND layer

EOF
}

# Parse command line arguments
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
        --model_depth)
            MODEL_DEPTH="$2"
            shift 2
            ;;
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --class_num)
            CLASS_NUM="$2"
            shift 2
            ;;
        --cfg)
            CFG="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --top_k)
            TOP_K="$2"
            shift 2
            ;;
        --top_p)
            TOP_P="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --more_smooth)
            MORE_SMOOTH="$2"
            shift 2
            ;;
        --layer_cache_spec)
            LAYER_CACHE_SPEC="$2"
            shift 2
            ;;
        --cache_attn_layers)
            CACHE_ATTN_LAYERS="$2"
            shift 2
            ;;
        --cache_mlp_layers)
            CACHE_MLP_LAYERS="$2"
            shift 2
            ;;
        --layer_blend_ratios)
            LAYER_BLEND_RATIOS="$2"
            shift 2
            ;;
        --interpolation_mode)
            INTERPOLATION_MODE="$2"
            shift 2
            ;;
        --skip_stages)
            SKIP_STAGES="$2"
            shift 2
            ;;
        --cache_stages)
            CACHE_STAGES="$2"
            shift 2
            ;;
        --enable_attn_cache)
            ENABLE_ATTN_CACHE="$2"
            shift 2
            ;;
        --enable_mlp_cache)
            ENABLE_MLP_CACHE="$2"
            shift 2
            ;;
        --threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        --max_skip_stages)
            MAX_SKIP_STAGES="$2"
            shift 2
            ;;
        --adaptive_threshold)
            ADAPTIVE_THRESHOLD="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --save_images)
            SAVE_IMAGES="$2"
            shift 2
            ;;
        --save_npy)
            SAVE_NPY="$2"
            shift 2
            ;;
        --compute_fid)
            COMPUTE_FID="$2"
            shift 2
            ;;
        --real_img_dir)
            REAL_IMG_DIR="$2"
            shift 2
            ;;
        --fid_cache_file)
            FID_CACHE_FILE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --num_workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$MODEL_PATH" ]]; then
    echo "Error: --model_path is required"
    echo "Use --help for usage information"
    exit 1
fi

if [[ -z "$VAE_PATH" ]]; then
    echo "Error: --vae_path is required"
    echo "Use --help for usage information"
    exit 1
fi

# Validate file paths
if [[ ! -f "$MODEL_PATH" ]]; then
    echo "Error: Model file not found: $MODEL_PATH"
    exit 1
fi

if [[ ! -f "$VAE_PATH" ]]; then
    echo "Error: VAE file not found: $VAE_PATH"
    exit 1
fi

# Validate FID settings
if [[ "$COMPUTE_FID" == "true" ]] && [[ -z "$REAL_IMG_DIR" ]]; then
    echo "Error: --real_img_dir is required when --compute_fid is true"
    exit 1
fi

if [[ "$COMPUTE_FID" == "true" ]] && [[ ! -d "$REAL_IMG_DIR" ]]; then
    echo "Error: Real images directory not found: $REAL_IMG_DIR"
    exit 1
fi

# Print configuration
echo "============================================"
echo "VAR Fine-Grained Layer Cache Control"
echo "============================================"
echo "Model path: $MODEL_PATH"
echo "VAE path: $VAE_PATH"
echo "Model depth: $MODEL_DEPTH"
echo "Number of samples: $NUM_SAMPLES"
echo "Batch size: $BATCH_SIZE"
echo "CFG scale: $CFG"
echo "Top-k: $TOP_K, Top-p: $TOP_P"
echo "Temperature: $TEMPERATURE"
echo "Seed: $SEED"
echo "============================================"
echo "Fine-Grained Cache Configuration:"
echo "Layer cache spec: $LAYER_CACHE_SPEC"
echo "Cache attention layers: $CACHE_ATTN_LAYERS"
echo "Cache MLP layers: $CACHE_MLP_LAYERS"
echo "Layer blend ratios: $LAYER_BLEND_RATIOS"
echo "Interpolation mode: $INTERPOLATION_MODE"
echo "============================================"
echo "Legacy Cache Configuration:"
echo "Skip stages: $SKIP_STAGES"
echo "Cache stages: $CACHE_STAGES"
echo "Enable attention cache: $ENABLE_ATTN_CACHE"
echo "Enable MLP cache: $ENABLE_MLP_CACHE"
echo "Threshold: $THRESHOLD"
echo "============================================"
echo "Output configuration:"
echo "Output directory: $OUTPUT_DIR"
echo "Save images: $SAVE_IMAGES"
echo "Save numpy: $SAVE_NPY"
echo "Compute FID: $COMPUTE_FID"
if [[ "$COMPUTE_FID" == "true" ]]; then
    echo "Real images directory: $REAL_IMG_DIR"
fi
echo "============================================"

# Build Python command
PYTHON_CMD="python var_evaluate_layer_cache_control.py"
PYTHON_CMD="$PYTHON_CMD --model_path $MODEL_PATH"
PYTHON_CMD="$PYTHON_CMD --vae_path $VAE_PATH"
PYTHON_CMD="$PYTHON_CMD --model_depth $MODEL_DEPTH"
PYTHON_CMD="$PYTHON_CMD --num_samples $NUM_SAMPLES"
PYTHON_CMD="$PYTHON_CMD --batch_size $BATCH_SIZE"
PYTHON_CMD="$PYTHON_CMD --class_num $CLASS_NUM"
PYTHON_CMD="$PYTHON_CMD --cfg $CFG"
PYTHON_CMD="$PYTHON_CMD --seed $SEED"
PYTHON_CMD="$PYTHON_CMD --top_k $TOP_K"
PYTHON_CMD="$PYTHON_CMD --top_p $TOP_P"
PYTHON_CMD="$PYTHON_CMD --temperature $TEMPERATURE"
PYTHON_CMD="$PYTHON_CMD --more_smooth $MORE_SMOOTH"

# Fine-grained cache control parameters
if [[ -n "$LAYER_CACHE_SPEC" ]]; then
    PYTHON_CMD="$PYTHON_CMD --layer_cache_spec '$LAYER_CACHE_SPEC'"
fi
if [[ -n "$CACHE_ATTN_LAYERS" ]]; then
    PYTHON_CMD="$PYTHON_CMD --cache_attn_layers '$CACHE_ATTN_LAYERS'"
fi
if [[ -n "$CACHE_MLP_LAYERS" ]]; then
    PYTHON_CMD="$PYTHON_CMD --cache_mlp_layers '$CACHE_MLP_LAYERS'"
fi
if [[ -n "$LAYER_BLEND_RATIOS" ]]; then
    PYTHON_CMD="$PYTHON_CMD --layer_blend_ratios '$LAYER_BLEND_RATIOS'"
fi
PYTHON_CMD="$PYTHON_CMD --interpolation_mode $INTERPOLATION_MODE"

# Legacy cache control parameters
if [[ -n "$SKIP_STAGES" ]]; then
    PYTHON_CMD="$PYTHON_CMD --skip_stages $SKIP_STAGES"
fi
if [[ -n "$CACHE_STAGES" ]]; then
    PYTHON_CMD="$PYTHON_CMD --cache_stages $CACHE_STAGES"
fi
PYTHON_CMD="$PYTHON_CMD --enable_attn_cache $ENABLE_ATTN_CACHE"
PYTHON_CMD="$PYTHON_CMD --enable_mlp_cache $ENABLE_MLP_CACHE"
PYTHON_CMD="$PYTHON_CMD --threshold $THRESHOLD"
PYTHON_CMD="$PYTHON_CMD --max_skip_stages $MAX_SKIP_STAGES"
PYTHON_CMD="$PYTHON_CMD --adaptive_threshold $ADAPTIVE_THRESHOLD"

# Output parameters
PYTHON_CMD="$PYTHON_CMD --output_dir $OUTPUT_DIR"
PYTHON_CMD="$PYTHON_CMD --save_images $SAVE_IMAGES"
PYTHON_CMD="$PYTHON_CMD --save_npy $SAVE_NPY"

# Evaluation parameters
PYTHON_CMD="$PYTHON_CMD --compute_fid $COMPUTE_FID"
if [[ -n "$REAL_IMG_DIR" ]]; then
    PYTHON_CMD="$PYTHON_CMD --real_img_dir $REAL_IMG_DIR"
fi
if [[ -n "$FID_CACHE_FILE" ]]; then
    PYTHON_CMD="$PYTHON_CMD --fid_cache_file $FID_CACHE_FILE"
fi

# Performance parameters
PYTHON_CMD="$PYTHON_CMD --device $DEVICE"
PYTHON_CMD="$PYTHON_CMD --num_workers $NUM_WORKERS"

# Execute the command
echo "Executing: $PYTHON_CMD"
echo "============================================"

# Check if Python script exists
if [[ ! -f "var_evaluate_layer_cache_control.py" ]]; then
    echo "Error: var_evaluate_layer_cache_control.py not found in current directory"
    exit 1
fi

# Run the Python script
eval $PYTHON_CMD

echo "============================================"
echo "Generation completed!"
echo "============================================"