#!/bin/bash

# ======================================================
# Motivated Reasoning - Group Alignment Analysis
# ======================================================
# This script runs only the group alignment impact analysis
# and generates the corresponding visualization in the tests folder.

set -e  # Exit on error

# Directory structure - updated to use tests folder
TESTS_DIR="tests"
BASE_DIR="${TESTS_DIR}/motivated_reasoning"
PLOTS_DIR="${BASE_DIR}/plots"
MAIN_SCRIPT="motivated_reasoning_simulation.py"
PLOT_SCRIPT="plots_motivated_reasoning_main.py"

# Default parameters - can be changed via command line
NUM_AGENTS=20
MODEL="gpt-4o-mini-2024-07-18"
TEMPERATURE=0.7
GROUP_ALIGNMENT_PROB=0.95
CLEAN_PREVIOUS=false

# Color codes for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print header
echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE}     Group Alignment Impact Analysis             ${NC}"
echo -e "${BLUE}======================================================${NC}"

# Function to display usage
usage() {
    echo -e "Usage: $0 [options]"
    echo -e "Options:"
    echo -e "  -n, --num-agents NUM      Number of agents (default: ${NUM_AGENTS})"
    echo -e "  -m, --model MODEL         LLM model to use (default: ${MODEL})"
    echo -e "  -t, --temperature TEMP    Temperature parameter (default: ${TEMPERATURE})"
    echo -e "  -g, --group-alignment PROB  Group-opinion alignment probability (default: ${GROUP_ALIGNMENT_PROB})"
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
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -t|--temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        -g|--group-alignment)
            GROUP_ALIGNMENT_PROB="$2"
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

# Clean previous data if requested
if [ "$CLEAN_PREVIOUS" = true ]; then
    echo -e "${YELLOW}This will delete all previous group alignment data in ${TESTS_DIR}${NC}"
    read -p "Are you sure you want to continue? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}Cleaning previous data...${NC}"
        # Clean both old and new naming patterns
        rm -rf "${BASE_DIR}"/with_identity/correlation_impact_*
        rm -rf "${BASE_DIR}"/without_identity/correlation_impact_*
        rm -rf "${PLOTS_DIR}"/correlation_impact_*
        rm -rf "${PLOTS_DIR}"/correlation_impact_comparison_*
        
        rm -rf "${BASE_DIR}"/with_identity/group_alignment_impact_*
        rm -rf "${BASE_DIR}"/without_identity/group_alignment_impact_*
        rm -rf "${PLOTS_DIR}"/group_alignment_impact_*
        rm -rf "${PLOTS_DIR}"/group_alignment_impact_comparison_*
        
        echo -e "${GREEN}Previous group alignment data removed${NC}"
    else
        echo -e "${BLUE}Keeping previous data${NC}"
    fi
fi

# Create necessary directories in tests folder
echo -e "${BLUE}Setting up directories in tests folder...${NC}"
mkdir -p ${BASE_DIR}/with_identity
mkdir -p ${BASE_DIR}/without_identity
mkdir -p ${PLOTS_DIR}

# Check if required files exist
if [ ! -f "$MAIN_SCRIPT" ]; then
    echo -e "${RED}Error: Simulation script ${MAIN_SCRIPT} not found${NC}"
    exit 1
fi

if [ ! -f "$PLOT_SCRIPT" ]; then
    echo -e "${RED}Error: Plotting script ${PLOT_SCRIPT} not found${NC}"
    exit 1
fi

# Display configuration
echo -e "${BLUE}Configuration:${NC}"
echo -e "  Number of agents:      ${GREEN}${NUM_AGENTS}${NC}"
echo -e "  Model:                 ${GREEN}${MODEL}${NC}"
echo -e "  Temperature:           ${GREEN}${TEMPERATURE}${NC}"
echo -e "  Group alignment:       ${GREEN}${GROUP_ALIGNMENT_PROB}${NC}"
echo -e "  Output directory:      ${GREEN}${BASE_DIR}${NC}"

# Confirm before proceeding
read -p "Press Enter to start the analysis or Ctrl+C to cancel..."

# Run only Part 1B of the simulation using Python to import and call the function directly
echo -e "${BLUE}Running group alignment impact analysis...${NC}"
echo -e "${YELLOW}This will take some time as it queries the LLM multiple times.${NC}"

# Start timer
start_time=$(date +%s)

# Run Part 1B for both conditions using Python one-liner with custom data directory
python -c "
import os
from motivated_reasoning_simulation import measure_identity_alignment_impact

# Override the data directories to use tests folder
data_dirs = [
    '${BASE_DIR}/with_identity', 
    '${BASE_DIR}/without_identity'
]
for dir_path in data_dirs:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# Set constants within this script
MODEL = '${MODEL}'
T = ${TEMPERATURE}

# Define a wrapper function that overrides the output path
def run_with_custom_path(n, group_alignment):
    folder = 'with_identity' if group_alignment else 'without_identity'
    output_file = f'${BASE_DIR}/{folder}/group_alignment_impact_n{n}_t{T}_{MODEL}_k_z.txt'
    
    # Original function uses a hardcoded path pattern, so we need to monkey patch
    import builtins
    original_open = builtins.open
    
    def custom_open(file, mode='r', *args, **kwargs):
        if 'group_alignment_impact' in file and mode == 'w':
            return original_open(output_file, mode, *args, **kwargs)
        elif 'group_alignment_impact' in file and mode == 'a':
            return original_open(output_file, mode, *args, **kwargs)
        return original_open(file, mode, *args, **kwargs)
    
    # Apply the monkey patch
    builtins.open = custom_open
    
    try:
        print(f'Running with custom path: {output_file}')
        measure_identity_alignment_impact(n, group_alignment)
    finally:
        # Restore original open
        builtins.open = original_open

# Run the functions with custom paths
print('Running with identity:')
run_with_custom_path(${NUM_AGENTS}, True)
print('Running without identity:')
run_with_custom_path(${NUM_AGENTS}, False)
"

# Check if simulation was successful
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Group alignment analysis failed${NC}"
    exit 1
fi

# Generate group alignment impact plot with custom paths - UPDATED TO DROP THE THIRD GRAPH
echo -e "${BLUE}Generating group alignment impact plot...${NC}"
python -c "
import os
import glob
import matplotlib.pyplot as plt

# Override the plots directory
plots_dir = '${PLOTS_DIR}'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Define a custom function that uses the tests paths
def custom_plot_group_alignment_impact():
    import pandas as pd
    import numpy as np
    import re
    
    # Find group alignment impact files for both conditions in tests folder
    with_id_pattern = '${BASE_DIR}/with_identity/group_alignment_impact_n*_t*_*_k_z.txt'
    without_id_pattern = '${BASE_DIR}/without_identity/group_alignment_impact_n*_t*_*_k_z.txt'
    
    with_id_files = glob.glob(with_id_pattern)
    without_id_files = glob.glob(without_id_pattern)
    
    if not with_id_files:
        print('No group alignment impact files found for with_identity condition.')
        return
    
    if not without_id_files:
        print('No group alignment impact files found for without_identity condition.')
        return
    
    # Use the most recent files for both conditions
    with_id_file = max(with_id_files, key=os.path.getmtime)
    without_id_file = max(without_id_files, key=os.path.getmtime)
    
    print(f'Using files:\\n- {with_id_file}\\n- {without_id_file}')
    
    # Extract simulation parameters
    filename = os.path.basename(with_id_file)
    n_match = re.search(r'_n(\d+)_', filename)
    t_match = re.search(r'_t([\d\.]+)_', filename)
    model_match = re.search(r'_t[\d\.]+_(.+?)_k_z\.txt', filename)
    
    N = n_match.group(1) if n_match else 'unknown'
    T = t_match.group(1) if t_match else 'unknown'
    model = model_match.group(1) if model_match else 'unknown_model'
    
    # Load data for both conditions
    with_id_data = pd.read_csv(with_id_file)
    without_id_data = pd.read_csv(without_id_file)
    
    # Create a figure with TWO subplots instead of three
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # PLOT 1: Group-Specific Aligned Opinion Adoption (WITH identity)
    ax1.plot(with_id_data['group_alignment_prob'], with_id_data['group_a_aligned_adoption'], 
            marker='o', linestyle='-', linewidth=2, markersize=8, 
            label='Group A', color='#1f77b4')
    
    ax1.plot(with_id_data['group_alignment_prob'], with_id_data['group_b_aligned_adoption'], 
            marker='s', linestyle='-', linewidth=2, markersize=8, 
            label='Group B', color='#ff7f0e')
    
    # Reference line at 0.5 (random choice)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random Choice (0.5)')
    
    ax1.set_xlabel('Group-Opinion Alignment Probability')
    ax1.set_ylabel('Probability of Adopting Aligned Opinion')
    ax1.set_title('Group-Specific Aligned Opinion Adoption\\n(WITH Identity Visible)')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])
    ax1.set_xlim([0.45, 1.05])
    ax1.legend(loc='best')
    
    # PLOT 2: Group-Specific Aligned Opinion Adoption (WITHOUT identity)
    ax2.plot(without_id_data['group_alignment_prob'], without_id_data['group_a_aligned_adoption'], 
            marker='o', linestyle='-', linewidth=2, markersize=8, 
            label='Group A', color='#1f77b4')
    
    ax2.plot(without_id_data['group_alignment_prob'], without_id_data['group_b_aligned_adoption'], 
            marker='s', linestyle='-', linewidth=2, markersize=8, 
            label='Group B', color='#ff7f0e')
    
    # Reference line at 0.5 (random choice)
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random Choice (0.5)')
    
    ax2.set_xlabel('Group-Opinion Alignment Probability')
    ax2.set_ylabel('Probability of Adopting Aligned Opinion')
    ax2.set_title('Group-Specific Aligned Opinion Adoption\\n(WITHOUT Identity Visible)')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    ax2.set_xlim([0.45, 1.05])
    ax2.legend(loc='best')
    
    # Add an overall title
    plt.suptitle(f'Effect of Group-Opinion Alignment on Decision Making\\n{model}, N={N}, T={T}', fontsize=16)
    
    # Save the plot to tests folder
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
    output_file = f'${PLOTS_DIR}/group_alignment_impact_{N}_{model}.png'
    plt.savefig(output_file, dpi=300)
    print(f'Plot saved to: {output_file}')
    plt.close()
    
    # Keep the second comparison plot unchanged
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate the difference in aligned adoption between conditions
    with_aligned_avg = (with_id_data['group_a_aligned_adoption'] + with_id_data['group_b_aligned_adoption'])/2
    without_aligned_avg = (without_id_data['group_a_aligned_adoption'] + without_id_data['group_b_aligned_adoption'])/2
    
    # Plot both average aligned adoption rates
    ax.plot(with_id_data['group_alignment_prob'], with_aligned_avg, 
            marker='o', linestyle='-', linewidth=2, markersize=10, 
            label='With Identity', color='#1f77b4')
    
    ax.plot(without_id_data['group_alignment_prob'], without_aligned_avg, 
            marker='s', linestyle='-', linewidth=2, markersize=10, 
            label='Without Identity', color='#ff7f0e')
    
    # Plot the difference
    ax.plot(with_id_data['group_alignment_prob'], with_aligned_avg - without_aligned_avg, 
            marker='d', linestyle='--', linewidth=2, markersize=8, 
            label='Difference (With - Without)', color='#2ca02c')
    
    # Reference line at 0 (no difference)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    # Reference line at 0.5 (random choice)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Random Choice (0.5)')
    
    ax.set_xlabel('Group-Opinion Alignment Probability')
    ax.set_ylabel('Average Aligned Opinion Adoption')
    ax.set_title(f'Impact of Identity Visibility on Aligned Opinion Adoption\\n{model}, N={N}, T={T}')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    ax.set_xlim([0.45, 1.05])
    
    # Save the comparison plot
    comparison_file = f'${PLOTS_DIR}/group_alignment_impact_comparison_{N}_{model}.png'
    plt.tight_layout()
    plt.savefig(comparison_file, dpi=300)
    print(f'Comparison plot saved to: {comparison_file}')
    plt.close()

# Run the custom plotting function
custom_plot_group_alignment_impact()
"

# Calculate total execution time
end_time=$(date +%s)
duration=$((end_time - start_time))
minutes=$((duration / 60))
seconds=$((duration % 60))

# Print completion message
echo -e "${GREEN}======================================================${NC}"
echo -e "${GREEN}     Group Alignment Analysis Complete           ${NC}"
echo -e "${GREEN}======================================================${NC}"
echo -e "Total execution time: ${minutes} minutes and ${seconds} seconds"
echo -e ""
echo -e "Results saved to:"
echo -e "  - Group alignment data:    ${BLUE}${BASE_DIR}/with_identity/group_alignment_impact_*.txt${NC}"
echo -e "                         ${BLUE}${BASE_DIR}/without_identity/group_alignment_impact_*.txt${NC}"
echo -e "  - Group alignment plot:    ${BLUE}${PLOTS_DIR}/group_alignment_impact_*.png${NC}"

exit 0 