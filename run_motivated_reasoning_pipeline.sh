#!/bin/bash

# ======================================================
# Motivated Reasoning Simulation - Complete Pipeline
# ======================================================
# This script runs the complete motivated reasoning simulation
# pipeline from data generation to analysis and visualization.

set -e  # Exit on error

# Directory structure
BASE_DIR="data/motivated_reasoning"
PLOTS_DIR="${BASE_DIR}/plots"
MAIN_SCRIPT="motivated_reasoning_simulation.py"
MAIN_PLOT_SCRIPT="plots_motivated_reasoning_main.py"
SI_PLOT_SCRIPT="plots_motivated_reasoning_SI.py"

# Default parameters - can be changed via command line
NUM_AGENTS=20
NUM_SIMULATIONS=5
MODEL="gpt-4o-mini-2024-07-18"
TEMPERATURE=0.7
GROUP_OPINION_ALIGNMENT_PROB=0.7
CLEAN_PREVIOUS=false

# Color codes for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print header
echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE}     Motivated Reasoning Simulation Pipeline          ${NC}"
echo -e "${BLUE}======================================================${NC}"

# Function to display usage
usage() {
    echo -e "Usage: $0 [options]"
    echo -e "Options:"
    echo -e "  -n, --num-agents NUM      Number of agents (default: ${NUM_AGENTS})"
    echo -e "  -s, --simulations NUM     Number of simulation runs (default: ${NUM_SIMULATIONS})"
    echo -e "  -m, --model MODEL         LLM model to use (default: ${MODEL})"
    echo -e "  -t, --temperature TEMP    Temperature parameter (default: ${TEMPERATURE})"
    echo -e "  -a, --group-alignment-prob PROB Group-opinion alignment probability (default: ${GROUP_OPINION_ALIGNMENT_PROB})"
    echo -e "  -c, --clean               Clean previous data before running"
    echo -e "  -h, --help                Display this help message"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--num-agents)
            NUM_AGENTS="$2"
            shift 2
            ;;
        -s|--simulations)
            NUM_SIMULATIONS="$2"
            shift 2
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -t|--temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        -a|--group-alignment-prob)
            GROUP_OPINION_ALIGNMENT_PROB="$2"
            shift 2
            ;;
        -c|--clean)
            CLEAN_PREVIOUS=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python is not installed or not in PATH${NC}"
    exit 1
fi

# Function to check if a Python package is installed
check_package() {
    python -c "import $1" 2>/dev/null
    return $?
}

# Check required Python packages
echo -e "${BLUE}Checking dependencies...${NC}"
MISSING_PACKAGES=()

for package in openai numpy pandas matplotlib scipy; do
    if ! check_package $package; then
        MISSING_PACKAGES+=($package)
    fi
done

# Install missing packages if any
if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo -e "${YELLOW}Missing required packages: ${MISSING_PACKAGES[*]}${NC}"
    read -p "Do you want to install them now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pip install ${MISSING_PACKAGES[*]}
    else
        echo -e "${RED}Required packages missing. Cannot continue.${NC}"
        exit 1
    fi
fi

# Clean previous data if requested
if [ "$CLEAN_PREVIOUS" = true ]; then
    echo -e "${YELLOW}This will delete all previous simulation data in ${BASE_DIR}${NC}"
    read -p "Are you sure you want to continue? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}Cleaning previous data...${NC}"
        rm -rf ${BASE_DIR}
        echo -e "${GREEN}Previous data removed${NC}"
    else
        echo -e "${BLUE}Keeping previous data${NC}"
    fi
fi

# Create necessary directories
echo -e "${BLUE}Setting up directories...${NC}"
mkdir -p ${PLOTS_DIR}
mkdir -p ${PLOTS_DIR}/SI

# Check if simulation script exists
if [ ! -f "$MAIN_SCRIPT" ]; then
    echo -e "${RED}Error: Simulation script ${MAIN_SCRIPT} not found${NC}"
    exit 1
fi

# Check if plotting scripts exist
if [ ! -f "$MAIN_PLOT_SCRIPT" ]; then
    echo -e "${RED}Error: Main plotting script ${MAIN_PLOT_SCRIPT} not found${NC}"
    exit 1
fi

if [ ! -f "$SI_PLOT_SCRIPT" ]; then
    echo -e "${RED}Error: SI plotting script ${SI_PLOT_SCRIPT} not found${NC}"
    exit 1
fi

# Display configuration
echo -e "${BLUE}Configuration:${NC}"
echo -e "  Number of agents:      ${GREEN}${NUM_AGENTS}${NC}"
echo -e "  Number of simulations: ${GREEN}${NUM_SIMULATIONS}${NC}"
echo -e "  Model:                 ${GREEN}${MODEL}${NC}"
echo -e "  Temperature:           ${GREEN}${TEMPERATURE}${NC}"
echo -e "  Group-opinion alignment probability:  ${GREEN}${GROUP_OPINION_ALIGNMENT_PROB}${NC}"

# Confirm before proceeding
read -p "Press Enter to start the simulation pipeline or Ctrl+C to cancel..."

# Run the simulation
echo -e "${BLUE}Starting motivated reasoning simulation...${NC}"
echo -e "${YELLOW}This may take a while depending on the number of agents and simulations.${NC}"
echo -e "${YELLOW}Check the API key in the script if you encounter authentication errors.${NC}"

# Start timer
start_time=$(date +%s)

# Run the simulation script with parameters passed directly
python ${MAIN_SCRIPT} --num-agents ${NUM_AGENTS} --simulations ${NUM_SIMULATIONS} \
    --model "${MODEL}" --temperature ${TEMPERATURE} --group-alignment-prob ${GROUP_OPINION_ALIGNMENT_PROB}

# Check if simulation was successful
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Simulation failed${NC}"
    exit 1
fi

# Run the plotting scripts
echo -e "${BLUE}Generating main plots...${NC}"
python ${MAIN_PLOT_SCRIPT}

echo -e "${BLUE}Generating supplementary plots...${NC}"
python ${SI_PLOT_SCRIPT}

# Calculate total execution time
end_time=$(date +%s)
duration=$((end_time - start_time))
minutes=$((duration / 60))
seconds=$((duration % 60))

# Print completion message
echo -e "${GREEN}======================================================${NC}"
echo -e "${GREEN}     Motivated Reasoning Analysis Complete            ${NC}"
echo -e "${GREEN}======================================================${NC}"
echo -e "Total execution time: ${minutes} minutes and ${seconds} seconds"
echo -e ""
echo -e "Results saved to:"
echo -e "  - Simulation data:    ${BLUE}${BASE_DIR}${NC}"
echo -e "  - Main plots:         ${BLUE}${PLOTS_DIR}${NC}"
echo -e "  - Supplementary plots: ${BLUE}${PLOTS_DIR}/SI${NC}"
echo -e ""
echo -e "To view the results, check the plot files in the plots directory"

exit 0 