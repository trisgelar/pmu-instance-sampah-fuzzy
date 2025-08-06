#!/bin/bash

# =============================================================================
# YOLOv10 PIPELINE EXECUTION SCRIPT
# =============================================================================
# This script provides various scenarios for running YOLOv10 trash image segmentation
# 
# USAGE:
#   ./run_yolov10_pipeline.sh [scenario]
#
# SCENARIOS:
#   1. train_new          - Train a new YOLOv10 model from scratch
#   2. use_existing       - Use existing training results (no retraining)
#   3. complete_pipeline  - Complete pipeline with ONNX export and inference
#   4. safe_pipeline      - Safe pipeline (skip problematic inference)
#   5. onnx_export        - Export ONNX from existing model
#   6. analysis_only      - Only run analysis on existing results
#   7. inference_only     - Only run inference and visualization
#   8. rknn_only          - Only convert to RKNN
#   9. all_models         - Run all YOLOv10 variants (v10n, v10s, v10m, v10l, v10x)
#   10. custom            - Custom configuration (edit script)
#
# EXAMPLES:
#   ./run_yolov10_pipeline.sh train_new
#   ./run_yolov10_pipeline.sh use_existing
#   ./run_yolov10_pipeline.sh complete_pipeline
# =============================================================================

# Configuration
MODEL_VERSION="v10n"  # Change to v10s, v10m, v10l, v10x as needed
EPOCHS=50
BATCH_SIZE=16
IMAGE_SIZE=640
DATASET_PATH="datasets"
SAVE_TO_DRIVE=true

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${PURPLE}=============================================================================${NC}"
    echo -e "${PURPLE}$1${NC}"
    echo -e "${PURPLE}=============================================================================${NC}"
}

# Function to check if virtual environment is activated
check_venv() {
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        print_warning "Virtual environment not detected. Activating .venv..."
        source .venv/bin/activate
    else
        print_success "Virtual environment detected: $VIRTUAL_ENV"
    fi
}

# Function to check if required files exist
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check if main_colab.py exists
    if [[ ! -f "main_colab.py" ]]; then
        print_error "main_colab.py not found!"
        exit 1
    fi
    
    # Check if config.yaml exists
    if [[ ! -f "config.yaml" ]]; then
        print_warning "config.yaml not found. Using default configuration."
    fi
    
    # Check if datasets directory exists
    if [[ ! -d "$DATASET_PATH" ]]; then
        print_warning "Datasets directory not found. Will download from Roboflow."
    fi
    
    print_success "Prerequisites check completed."
}

# Function to run training scenario
run_train_new() {
    print_header "YOLOv10 TRAINING NEW MODEL"
    print_status "Model: $MODEL_VERSION"
    print_status "Epochs: $EPOCHS"
    print_status "Batch Size: $BATCH_SIZE"
    print_status "Image Size: $IMAGE_SIZE"
    
    python main_colab.py \
        --model $MODEL_VERSION \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --image-size $IMAGE_SIZE \
        --save-to-drive $SAVE_TO_DRIVE
}

# Function to use existing training results
run_use_existing() {
    print_header "YOLOv10 USING EXISTING RESULTS"
    print_status "Model: $MODEL_VERSION"
    print_status "Looking for existing training results..."
    
    python main_colab.py \
        --model $MODEL_VERSION \
        --use-existing-results \
        --save-to-drive $SAVE_TO_DRIVE
}

# Function to run complete pipeline
run_complete_pipeline() {
    print_header "YOLOv10 COMPLETE PIPELINE"
    print_status "Model: $MODEL_VERSION"
    print_status "This will train (if needed), analyze, infer, and convert to RKNN"
    
    python main_colab.py \
        --model $MODEL_VERSION \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --image-size $IMAGE_SIZE \
        --complete-pipeline \
        --save-to-drive $SAVE_TO_DRIVE
}

# Function to run safe pipeline
run_safe_pipeline() {
    print_header "YOLOv10 SAFE PIPELINE"
    print_status "Model: $MODEL_VERSION"
    print_status "This skips problematic inference visualization"
    
    python main_colab.py \
        --model $MODEL_VERSION \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --image-size $IMAGE_SIZE \
        --safe-pipeline \
        --save-to-drive $SAVE_TO_DRIVE
}

# Function to export ONNX from existing model
run_onnx_export() {
    print_header "YOLOv10 ONNX EXPORT"
    print_status "Model: $MODEL_VERSION"
    print_status "Exporting ONNX from existing PyTorch model..."
    
    python main_colab.py \
        --model $MODEL_VERSION \
        --onnx-export \
        --save-to-drive $SAVE_TO_DRIVE
}

# Function to run analysis only
run_analysis_only() {
    print_header "YOLOv10 ANALYSIS ONLY"
    print_status "Model: $MODEL_VERSION"
    print_status "Running analysis on existing training results..."
    
    python main_colab.py \
        --model $MODEL_VERSION \
        --analysis-only \
        --save-to-drive $SAVE_TO_DRIVE
}

# Function to run inference only
run_inference_only() {
    print_header "YOLOv10 INFERENCE ONLY"
    print_status "Model: $MODEL_VERSION"
    print_status "Running inference and visualization..."
    
    python main_colab.py \
        --model $MODEL_VERSION \
        --inference-only \
        --save-to-drive $SAVE_TO_DRIVE
}

# Function to run RKNN conversion only
run_rknn_only() {
    print_header "YOLOv10 RKNN CONVERSION ONLY"
    print_status "Model: $MODEL_VERSION"
    print_status "Converting to RKNN format..."
    
    python main_colab.py \
        --model $MODEL_VERSION \
        --rknn-only \
        --save-to-drive $SAVE_TO_DRIVE
}

# Function to run all YOLOv10 variants
run_all_models() {
    print_header "YOLOv10 ALL VARIANTS"
    print_status "Running all YOLOv10 variants: v10n, v10s, v10m, v10l, v10x"
    
    models=("v10n" "v10s" "v10m" "v10l" "v10x")
    
    for model in "${models[@]}"; do
        print_status "Processing YOLO$model..."
        python main_colab.py \
            --model $model \
            --epochs $EPOCHS \
            --batch-size $BATCH_SIZE \
            --image-size $IMAGE_SIZE \
            --complete-pipeline \
            --save-to-drive $SAVE_TO_DRIVE
    done
}

# Function to run custom configuration
run_custom() {
    print_header "YOLOv10 CUSTOM CONFIGURATION"
    print_status "Model: $MODEL_VERSION"
    print_status "Running custom configuration..."
    
    # EDIT THESE PARAMETERS AS NEEDED
    CUSTOM_EPOCHS=100
    CUSTOM_BATCH_SIZE=8
    CUSTOM_IMAGE_SIZE=512
    
    print_status "Custom parameters:"
    print_status "  Epochs: $CUSTOM_EPOCHS"
    print_status "  Batch Size: $CUSTOM_BATCH_SIZE"
    print_status "  Image Size: $CUSTOM_IMAGE_SIZE"
    
    python main_colab.py \
        --model $MODEL_VERSION \
        --epochs $CUSTOM_EPOCHS \
        --batch-size $CUSTOM_BATCH_SIZE \
        --image-size $CUSTOM_IMAGE_SIZE \
        --complete-pipeline \
        --save-to-drive $SAVE_TO_DRIVE
}

# Function to show help
show_help() {
    print_header "YOLOv10 PIPELINE HELP"
    echo "Available scenarios:"
    echo "  1. train_new          - Train a new YOLOv10 model from scratch"
    echo "  2. use_existing       - Use existing training results (no retraining)"
    echo "  3. complete_pipeline  - Complete pipeline with ONNX export and inference"
    echo "  4. safe_pipeline      - Safe pipeline (skip problematic inference)"
    echo "  5. onnx_export        - Export ONNX from existing model"
    echo "  6. analysis_only      - Only run analysis on existing results"
    echo "  7. inference_only     - Only run inference and visualization"
    echo "  8. rknn_only          - Only convert to RKNN"
    echo "  9. all_models         - Run all YOLOv10 variants"
    echo "  10. custom            - Custom configuration (edit script)"
    echo ""
    echo "Examples:"
    echo "  ./run_yolov10_pipeline.sh train_new"
    echo "  ./run_yolov10_pipeline.sh use_existing"
    echo "  ./run_yolov10_pipeline.sh complete_pipeline"
    echo ""
    echo "Configuration (edit script to change):"
    echo "  MODEL_VERSION: $MODEL_VERSION"
    echo "  EPOCHS: $EPOCHS"
    echo "  BATCH_SIZE: $BATCH_SIZE"
    echo "  IMAGE_SIZE: $IMAGE_SIZE"
    echo "  DATASET_PATH: $DATASET_PATH"
    echo "  SAVE_TO_DRIVE: $SAVE_TO_DRIVE"
}

# Main execution
main() {
    print_header "YOLOv10 TRASH IMAGE SEGMENTATION PIPELINE"
    
    # Check virtual environment
    check_venv
    
    # Check prerequisites
    check_prerequisites
    
    # Get scenario from command line argument
    SCENARIO=${1:-"help"}
    
    case $SCENARIO in
        "train_new")
            run_train_new
            ;;
        "use_existing")
            run_use_existing
            ;;
        "complete_pipeline")
            run_complete_pipeline
            ;;
        "safe_pipeline")
            run_safe_pipeline
            ;;
        "onnx_export")
            run_onnx_export
            ;;
        "analysis_only")
            run_analysis_only
            ;;
        "inference_only")
            run_inference_only
            ;;
        "rknn_only")
            run_rknn_only
            ;;
        "all_models")
            run_all_models
            ;;
        "custom")
            run_custom
            ;;
        "help"|*)
            show_help
            ;;
    esac
    
    print_success "YOLOv10 pipeline execution completed!"
}

# Run main function
main "$@" 