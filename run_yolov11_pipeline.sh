#!/bin/bash

# =============================================================================
# YOLOv11 PIPELINE EXECUTION SCRIPT
# =============================================================================
# This script provides various scenarios for running YOLOv11 trash image segmentation
# 
# USAGE:
#   ./run_yolov11_pipeline.sh [scenario]
#
# SCENARIOS:
#   1. train_new          - Train a new YOLOv11 model from scratch
#   2. use_existing       - Use existing training results (no retraining)
#   3. complete_pipeline  - Complete pipeline with ONNX export and inference
#   4. safe_pipeline      - Safe pipeline (skip problematic inference)
#   5. onnx_export        - Export ONNX from existing model
#   6. analysis_only      - Only run analysis on existing results
#   7. inference_only     - Only run inference and visualization
#   8. rknn_only          - Only convert to RKNN
#   9. all_models         - Run all YOLOv11 variants (v11n, v11s, v11m, v11l, v11x)
#   10. custom            - Custom configuration (edit script)
#
# EXAMPLES:
#   ./run_yolov11_pipeline.sh train_new
#   ./run_yolov11_pipeline.sh use_existing
#   ./run_yolov11_pipeline.sh complete_pipeline
# =============================================================================

# =============================================================================
# EXECUTION MODE CONFIGURATION
# =============================================================================
# Detect execution environment and set appropriate parameters
# LOCAL MODE  (RTX 3050): Light training with 10 epochs for quick testing
# COLAB MODE  (High GPU): Full training with 200 epochs for production

# Function to detect execution environment
detect_environment() {
    if [[ -n "$COLAB_GPU" ]] || [[ -n "$COLAB_TPU_ADDR" ]] || [[ "$(hostname)" =~ colab ]] || [[ -f "/content" ]]; then
        echo "colab"
    else
        echo "local"
    fi
}

# Set environment-specific configuration
EXEC_MODE=$(detect_environment)

if [[ "$EXEC_MODE" == "colab" ]]; then
    # COLAB CONFIGURATION - High GPU resources
    DEFAULT_EPOCHS=200
    DEFAULT_BATCH_SIZE=32
    DEFAULT_IMAGE_SIZE=640
    SAVE_TO_DRIVE=true
    print_header "🚀 GOOGLE COLAB MODE DETECTED - HIGH PERFORMANCE TRAINING"
    print_status "GPU: Tesla T4/V100/A100 - Training with $DEFAULT_EPOCHS epochs"
else
    # LOCAL CONFIGURATION - RTX 3050 laptop
    DEFAULT_EPOCHS=10
    DEFAULT_BATCH_SIZE=8
    DEFAULT_IMAGE_SIZE=512
    SAVE_TO_DRIVE=false
    print_header "💻 LOCAL DEVELOPMENT MODE DETECTED - RTX 3050 OPTIMIZATION"
    print_status "GPU: RTX 3050 - Quick training with $DEFAULT_EPOCHS epochs"
fi

# Configuration (can be overridden by command line or manual edit)
MODEL_VERSION="v11n"  # Change to v11s, v11m, v11l, v11x as needed
EPOCHS=${EPOCHS:-$DEFAULT_EPOCHS}  # Use env var or default
BATCH_SIZE=${BATCH_SIZE:-$DEFAULT_BATCH_SIZE}  # Use env var or default
IMAGE_SIZE=${IMAGE_SIZE:-$DEFAULT_IMAGE_SIZE}  # Use env var or default
DATASET_PATH="datasets"

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

print_mode() {
    if [[ "$EXEC_MODE" == "colab" ]]; then
        echo -e "${CYAN}[COLAB]${NC} $1"
    else
        echo -e "${YELLOW}[LOCAL]${NC} $1"
    fi
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
    print_header "YOLOv11 TRAINING NEW MODEL"
    print_mode "Execution Mode: $EXEC_MODE"
    print_status "Model: $MODEL_VERSION"
    print_status "Epochs: $EPOCHS (${EXEC_MODE} optimized)"
    print_status "Batch Size: $BATCH_SIZE"
    print_status "Image Size: $IMAGE_SIZE"
    
    if [[ "$EXEC_MODE" == "local" ]]; then
        print_warning "LOCAL MODE: Using reduced epochs ($EPOCHS) for RTX 3050"
        print_warning "For full training, use Google Colab with 200 epochs"
    else
        print_success "COLAB MODE: Using full epochs ($EPOCHS) for production training"
    fi
    
    python main_colab.py \
        --models $MODEL_VERSION \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        $([ "$SAVE_TO_DRIVE" = "true" ] && echo "--save-to-drive")
}

# Function to use existing training results
run_use_existing() {
    print_header "YOLOv11 USING EXISTING RESULTS"
    print_status "Model: $MODEL_VERSION"
    print_status "Looking for existing training results..."
    
    python main_colab.py \
        --models $MODEL_VERSION \
        $([ "$SAVE_TO_DRIVE" = "true" ] && echo "--save-to-drive")
}

# Function to run complete pipeline
run_complete_pipeline() {
    print_header "YOLOv11 COMPLETE PIPELINE"
    print_status "Model: $MODEL_VERSION"
    print_status "This will train (if needed), analyze, infer, and convert to RKNN"
    
    python main_colab.py \
        --models $MODEL_VERSION \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        \
        --complete-pipeline \
        $([ "$SAVE_TO_DRIVE" = "true" ] && echo "--save-to-drive")
}

# Function to run safe pipeline
run_safe_pipeline() {
    print_header "YOLOv11 SAFE PIPELINE"
    print_status "Model: $MODEL_VERSION"
    print_status "This skips problematic inference visualization"
    
    python main_colab.py \
        --models $MODEL_VERSION \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        \
        --safe-pipeline \
        $([ "$SAVE_TO_DRIVE" = "true" ] && echo "--save-to-drive")
}

# Function to export ONNX from existing model
run_onnx_export() {
    print_header "YOLOv11 ONNX EXPORT"
    print_status "Model: $MODEL_VERSION"
    print_status "Exporting ONNX from existing PyTorch model..."
    
    python main_colab.py \
        --models $MODEL_VERSION \
        --onnx-export \
        $([ "$SAVE_TO_DRIVE" = "true" ] && echo "--save-to-drive")
}

# Function to run analysis only
run_analysis_only() {
    print_header "YOLOv11 ANALYSIS ONLY"
    print_status "Model: $MODEL_VERSION"
    print_status "Running analysis on existing training results..."
    
    python main_colab.py \
        --models $MODEL_VERSION \
print_status "Use complete_pipeline or onnx_export instead"
print_status "Use complete_pipeline or onnx_export instead"
}

# Function to run inference only
run_inference_only() {
    print_header "YOLOv11 INFERENCE ONLY"
    print_status "Model: $MODEL_VERSION"
    print_status "Running inference and visualization..."
    
    print_warning "Inference-only mode not supported in main_colab.py"
    print_status "Use complete_pipeline instead"
    return 1
}

# Function to run RKNN conversion only
run_rknn_only() {
    print_header "YOLOv11 RKNN CONVERSION ONLY"
    print_status "Model: $MODEL_VERSION"
    print_status "Converting to RKNN format..."
    
    print_warning "RKNN-only mode not supported in main_colab.py"
    print_status "Use complete_pipeline instead"
    return 1
}

# Function to run all YOLOv11 variants
run_all_models() {
    print_header "YOLOv11 ALL VARIANTS"
    print_status "Running all YOLOv11 variants: v11n, v11s, v11m, v11l, v11x"
    
    models=("v11n" "v11s" "v11m" "v11l" "v11x")
    
    for model in "${models[@]}"; do
        print_status "Processing YOLO$model..."
        python main_colab.py \
            --model $model \
            --epochs $EPOCHS \
            --batch-size $BATCH_SIZE \
            \
            --complete-pipeline \
            $([ "$SAVE_TO_DRIVE" = "true" ] && echo "--save-to-drive")
    done
}

# Function to run custom configuration
run_custom() {
    print_header "YOLOv11 CUSTOM CONFIGURATION"
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
        --models $MODEL_VERSION \
        --epochs $CUSTOM_EPOCHS \
        --batch-size $CUSTOM_BATCH_SIZE \
        --image-size $CUSTOM_IMAGE_SIZE \
        --complete-pipeline \
        $([ "$SAVE_TO_DRIVE" = "true" ] && echo "--save-to-drive")
}

# Function to show help
show_help() {
    print_header "YOLOv11 PIPELINE HELP"
    print_mode "Current execution mode: $EXEC_MODE"
    echo ""
    echo "Available scenarios:"
    echo "  1. train_new          - Train a new YOLOv11 model from scratch"
    echo "  2. use_existing       - Use existing training results (no retraining)"
    echo "  3. complete_pipeline  - Complete pipeline with ONNX export and inference"
    echo "  4. safe_pipeline      - Safe pipeline (skip problematic inference)"
    echo "  5. onnx_export        - Export ONNX from existing model"
    echo "  6. analysis_only      - Only run analysis on existing results"
    echo "  7. inference_only     - Only run inference and visualization"
    echo "  8. rknn_only          - Only convert to RKNN"
    echo "  9. all_models         - Run all YOLOv11 variants"
    echo "  10. custom            - Custom configuration (edit script)"
    echo "  11. local_mode        - Force local mode (10 epochs)"
    echo "  12. colab_mode        - Force colab mode (200 epochs)"
    echo ""
    echo "Examples:"
    echo "  ./run_yolov11_pipeline.sh train_new"
    echo "  ./run_yolov11_pipeline.sh use_existing"
    echo "  ./run_yolov11_pipeline.sh complete_pipeline"
    echo "  ./run_yolov11_pipeline.sh local_mode train_new"
    echo "  ./run_yolov11_pipeline.sh colab_mode train_new"
    echo ""
    echo "Current Configuration:"
    echo "  EXECUTION_MODE: $EXEC_MODE"
    echo "  MODEL_VERSION: $MODEL_VERSION"
    echo "  EPOCHS: $EPOCHS (${EXEC_MODE} optimized)"
    echo "  BATCH_SIZE: $BATCH_SIZE"
    echo "  IMAGE_SIZE: $IMAGE_SIZE"
    echo "  DATASET_PATH: $DATASET_PATH"
    echo "  SAVE_TO_DRIVE: $SAVE_TO_DRIVE"
    echo ""
    if [[ "$EXEC_MODE" == "local" ]]; then
        echo "🖥️  LOCAL MODE (RTX 3050):"
        echo "  • 10 epochs for quick testing"
        echo "  • Smaller batch size (8)"
        echo "  • Optimized for laptop GPU"
        echo "  • Results saved locally"
    else
        echo "☁️  COLAB MODE (High GPU):"
        echo "  • 200 epochs for production training"
        echo "  • Larger batch size (32)"
        echo "  • Optimized for cloud GPU"
        echo "  • Results saved to Google Drive"
    fi
    echo ""
    echo "Environment Variables (override defaults):"
    echo "  EPOCHS=50 ./run_yolov11_pipeline.sh train_new"
    echo "  BATCH_SIZE=16 ./run_yolov11_pipeline.sh train_new"
    echo "  IMAGE_SIZE=640 ./run_yolov11_pipeline.sh train_new"
}

# Main execution
main() {
    print_header "YOLOv11 TRASH IMAGE SEGMENTATION PIPELINE"
    
    # Check virtual environment
    check_venv
    
    # Check prerequisites
    check_prerequisites
    
    # Handle mode override arguments
    if [[ "$1" == "local_mode" ]]; then
        EXEC_MODE="local"
        EPOCHS=10
        BATCH_SIZE=8
        IMAGE_SIZE=512
        SAVE_TO_DRIVE=false
        print_warning "FORCED LOCAL MODE: RTX 3050 optimization"
        shift  # Remove first argument
    elif [[ "$1" == "colab_mode" ]]; then
        EXEC_MODE="colab"
        EPOCHS=200
        BATCH_SIZE=32
        IMAGE_SIZE=640
        SAVE_TO_DRIVE=true
        print_success "FORCED COLAB MODE: High GPU optimization"
        shift  # Remove first argument
    fi
    
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
    
    print_success "YOLOv11 pipeline execution completed!"
}

# Run main function
main "$@" 