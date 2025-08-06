#!/bin/bash

# =============================================================================
# MASTER PIPELINE EXECUTION SCRIPT
# =============================================================================
# This script provides a unified interface for running all YOLO versions
# (YOLOv8, YOLOv10, YOLOv11) with various scenarios
# 
# USAGE:
#   ./run_all_pipelines.sh [yolo_version] [scenario]
#
# YOLO VERSIONS:
#   v8  - YOLOv8 (v8n, v8s, v8m, v8l, v8x)
#   v10 - YOLOv10 (v10n, v10s, v10m, v10l, v10x)
#   v11 - YOLOv11 (v11n, v11s, v11m, v11l, v11x)
#   all - Run all YOLO versions
#
# SCENARIOS:
#   1. train_new          - Train a new model from scratch
#   2. use_existing       - Use existing training results (no retraining)
#   3. complete_pipeline  - Complete pipeline with ONNX export and inference
#   4. safe_pipeline      - Safe pipeline (skip problematic inference)
#   5. onnx_export        - Export ONNX from existing model
#   6. analysis_only      - Only run analysis on existing results
#   7. inference_only     - Only run inference and visualization
#   8. rknn_only          - Only convert to RKNN
#   9. all_models         - Run all variants of the specified YOLO version
#   10. custom            - Custom configuration (edit individual scripts)
#
# EXAMPLES:
#   ./run_all_pipelines.sh v8 train_new
#   ./run_all_pipelines.sh v10 use_existing
#   ./run_all_pipelines.sh v11 complete_pipeline
#   ./run_all_pipelines.sh all safe_pipeline
# =============================================================================

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
    
    # Check if individual scripts exist
    if [[ ! -f "run_yolov8_pipeline.sh" ]]; then
        print_error "run_yolov8_pipeline.sh not found!"
        exit 1
    fi
    
    if [[ ! -f "run_yolov10_pipeline.sh" ]]; then
        print_error "run_yolov10_pipeline.sh not found!"
        exit 1
    fi
    
    if [[ ! -f "run_yolov11_pipeline.sh" ]]; then
        print_error "run_yolov11_pipeline.sh not found!"
        exit 1
    fi
    
    # Make scripts executable
    chmod +x run_yolov8_pipeline.sh
    chmod +x run_yolov10_pipeline.sh
    chmod +x run_yolov11_pipeline.sh
    
    print_success "Prerequisites check completed."
}

# Function to run YOLOv8 pipeline
run_yolov8() {
    local scenario=$1
    print_header "RUNNING YOLOv8 PIPELINE"
    print_status "Scenario: $scenario"
    
    ./run_yolov8_pipeline.sh "$scenario"
}

# Function to run YOLOv10 pipeline
run_yolov10() {
    local scenario=$1
    print_header "RUNNING YOLOv10 PIPELINE"
    print_status "Scenario: $scenario"
    
    ./run_yolov10_pipeline.sh "$scenario"
}

# Function to run YOLOv11 pipeline
run_yolov11() {
    local scenario=$1
    print_header "RUNNING YOLOv11 PIPELINE"
    print_status "Scenario: $scenario"
    
    ./run_yolov11_pipeline.sh "$scenario"
}

# Function to run all YOLO versions
run_all_versions() {
    local scenario=$1
    print_header "RUNNING ALL YOLO VERSIONS"
    print_status "Scenario: $scenario"
    
    # Run YOLOv8
    print_status "Starting YOLOv8..."
    run_yolov8 "$scenario"
    
    # Run YOLOv10
    print_status "Starting YOLOv10..."
    run_yolov10 "$scenario"
    
    # Run YOLOv11
    print_status "Starting YOLOv11..."
    run_yolov11 "$scenario"
}

# Function to show help
show_help() {
    print_header "MASTER PIPELINE HELP"
    echo "Usage: ./run_all_pipelines.sh [yolo_version] [scenario]"
    echo ""
    echo "YOLO VERSIONS:"
    echo "  v8  - YOLOv8 (v8n, v8s, v8m, v8l, v8x)"
    echo "  v10 - YOLOv10 (v10n, v10s, v10m, v10l, v10x)"
    echo "  v11 - YOLOv11 (v11n, v11s, v11m, v11l, v11x)"
    echo "  all - Run all YOLO versions"
    echo ""
    echo "SCENARIOS:"
    echo "  1. train_new          - Train a new model from scratch"
    echo "  2. use_existing       - Use existing training results (no retraining)"
    echo "  3. complete_pipeline  - Complete pipeline with ONNX export and inference"
    echo "  4. safe_pipeline      - Safe pipeline (skip problematic inference)"
    echo "  5. onnx_export        - Export ONNX from existing model"
    echo "  6. analysis_only      - Only run analysis on existing results"
    echo "  7. inference_only     - Only run inference and visualization"
    echo "  8. rknn_only          - Only convert to RKNN"
    echo "  9. all_models         - Run all variants of the specified YOLO version"
    echo "  10. custom            - Custom configuration (edit individual scripts)"
    echo ""
    echo "EXAMPLES:"
    echo "  ./run_all_pipelines.sh v8 train_new"
    echo "  ./run_all_pipelines.sh v10 use_existing"
    echo "  ./run_all_pipelines.sh v11 complete_pipeline"
    echo "  ./run_all_pipelines.sh all safe_pipeline"
    echo ""
    echo "INDIVIDUAL SCRIPTS:"
    echo "  ./run_yolov8_pipeline.sh [scenario]"
    echo "  ./run_yolov10_pipeline.sh [scenario]"
    echo "  ./run_yolov11_pipeline.sh [scenario]"
}

# Function to validate inputs
validate_inputs() {
    local yolo_version=$1
    local scenario=$2
    
    # Validate YOLO version
    case $yolo_version in
        "v8"|"v10"|"v11"|"all")
            ;;
        *)
            print_error "Invalid YOLO version: $yolo_version"
            print_error "Valid versions: v8, v10, v11, all"
            return 1
            ;;
    esac
    
    # Validate scenario
    case $scenario in
        "train_new"|"use_existing"|"complete_pipeline"|"safe_pipeline"|"onnx_export"|"analysis_only"|"inference_only"|"rknn_only"|"all_models"|"custom")
            ;;
        *)
            print_error "Invalid scenario: $scenario"
            print_error "Valid scenarios: train_new, use_existing, complete_pipeline, safe_pipeline, onnx_export, analysis_only, inference_only, rknn_only, all_models, custom"
            return 1
            ;;
    esac
    
    return 0
}

# Main execution
main() {
    print_header "MASTER PIPELINE EXECUTION"
    
    # Check virtual environment
    check_venv
    
    # Check prerequisites
    check_prerequisites
    
    # Get arguments
    YOLO_VERSION=${1:-"help"}
    SCENARIO=${2:-"help"}
    
    # Show help if no arguments or help requested
    if [[ "$YOLO_VERSION" == "help" ]] || [[ "$SCENARIO" == "help" ]]; then
        show_help
        exit 0
    fi
    
    # Validate inputs
    if ! validate_inputs "$YOLO_VERSION" "$SCENARIO"; then
        show_help
        exit 1
    fi
    
    # Run appropriate pipeline
    case $YOLO_VERSION in
        "v8")
            run_yolov8 "$SCENARIO"
            ;;
        "v10")
            run_yolov10 "$SCENARIO"
            ;;
        "v11")
            run_yolov11 "$SCENARIO"
            ;;
        "all")
            run_all_versions "$SCENARIO"
            ;;
        *)
            print_error "Unknown YOLO version: $YOLO_VERSION"
            show_help
            exit 1
            ;;
    esac
    
    print_success "Master pipeline execution completed!"
}

# Run main function
main "$@" 