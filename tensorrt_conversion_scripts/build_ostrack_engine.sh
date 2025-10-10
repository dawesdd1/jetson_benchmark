#!/bin/bash
# build_ostrack_engine.sh
# Builds TensorRT engine from ONNX model (FP32) for OSTrack tracker on Jetson Orin

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

ONNX_MODEL="/home/copter/EdgeTAM/notebooks/onnx_models/vitb_256_mae_ce_32x4_ep300.onnx"
OUTPUT_DIR="/home/copter/EdgeTAM/notebooks/engine_files"
ENGINE_NAME="ostrack_vitb256_fp16_t128_s256.engine"
TRTEXEC="/usr/src/tensorrt/bin/trtexec"

# Input shapes for OSTrack
TEMPLATE_SHAPE="template:1x3x128x128"
SEARCH_SHAPE="search:1x3x256x256"

# ============================================================================
# Colors for output
# ============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# Validation
# ============================================================================

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}TensorRT Engine Builder for OSTrack${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if ONNX model exists
if [ ! -f "$ONNX_MODEL" ]; then
    echo -e "${RED}Error: ONNX model not found at: $ONNX_MODEL${NC}"
    exit 1
fi

# Check if trtexec exists
if [ ! -f "$TRTEXEC" ]; then
    echo -e "${RED}Error: trtexec not found at: $TRTEXEC${NC}"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Get ONNX model size
ONNX_SIZE=$(du -h "$ONNX_MODEL" | cut -f1)
echo -e "${GREEN}✓${NC} Found ONNX model: $ONNX_MODEL (${ONNX_SIZE})"
echo -e "${GREEN}✓${NC} Output directory: $OUTPUT_DIR"
echo ""

# ============================================================================
# Build TensorRT Engine
# ============================================================================

ENGINE_PATH="${OUTPUT_DIR}/${ENGINE_NAME}"

echo -e "${YELLOW}Building TensorRT engine...${NC}"
echo -e "  Input shapes:"
echo -e "    - Template: 1x3x128x128"
echo -e "    - Search:   1x3x256x256"
echo -e "  Precision: FP16"
echo -e "  Target: Jetson Orin"
echo ""

# Build engine with trtexec
TIMESTAMP=$(date +"%m%d%Y_%H%M")

$TRTEXEC \
    --onnx="$ONNX_MODEL" \
    --saveEngine="$ENGINE_PATH" \
    --fp16 \
    --memPoolSize=workspace:2048M \
    --verbose \
    --dumpProfile \
    --separateProfileRun \
    --warmUp=1000 \
    --iterations=100 \
    --avgRuns=10 \
    --duration=0 \
    --device=0 \
    --builderOptimizationLevel=5 \
    --tacticSources=+CUDNN,+CUBLAS,+CUBLAS_LT \
    --timingCacheFile="${OUTPUT_DIR}/timing_cache.bin" \
    --profilingVerbosity=detailed \
    2>&1 | tee "${OUTPUT_DIR}/build_log_${TIMESTAMP}.txt"

# ============================================================================
# Validation and Summary
# ============================================================================

if [ -f "$ENGINE_PATH" ]; then
    ENGINE_SIZE=$(du -h "$ENGINE_PATH" | cut -f1)
    
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN} ✅ Engine Built Successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "  Engine file:  $ENGINE_PATH"
    echo -e "  Engine size:  ${ENGINE_SIZE}"
    echo -e "  ONNX size:    ${ONNX_SIZE}"
    echo ""
    echo -e "${BLUE}Build log saved to:${NC} ${OUTPUT_DIR}/build_log.txt"
    echo -e "${BLUE}Timing cache saved to:${NC} ${OUTPUT_DIR}/timing_cache.bin"
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo -e "  1a. Test the engine with trtexec:"
    echo -e "     $TRTEXEC --loadEngine=$ENGINE_PATH --shapes=${TEMPLATE_SHAPE},${SEARCH_SHAPE}"
    echo ""
    echo -e "${YELLOW}or:${NC}"
    echo -e "  1b. Test the engine with trtexec and --useSpinWait to improve the latency spike stability.:"
    echo -e "     $TRTEXEC --loadEngine=$ENGINE_PATH --shapes=${TEMPLATE_SHAPE},${SEARCH_SHAPE} --useSpinWait"
    echo ""
    echo -e "  2. Update your Python code to use the engine:"
    echo -e "     onnx_model_path='$ENGINE_PATH'"
    echo ""
    
    exit 0
else
    # Get current timestamp in MMDDYYYY_HHMM format
    TIMESTAMP=$(date +"%m%d%Y_%H%M")
    
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED} ❌ Engine Build Failed!${NC}"
    echo -e "${RED}========================================${NC}"
    echo ""
    echo -e "${YELLOW}Check the build log for details:${NC}"
    echo -e "  ${OUTPUT_DIR}/build_log_${TIMESTAMP}.txt"
    echo ""
    
    exit 1
fi

