import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import json
from scipy.optimize import curve_fit
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
import re

# Create output directory for plots
plots_dir = 'data/motivated_reasoning/plots'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Define the tanh fitting function for transition probability
def tanh_fit(x, beta):
    return 0.5 * (np.tanh(beta * x) + 1)

# Increase font size for all plots
plt.rcParams.update({'font.size': 18})

#----------------------------------------
# PLOT 1: Transition Probability Curves
#----------------------------------------
def plot_transition_probabilities():
    # Paths to the transition probability files - Updated patterns
    with_id_pattern = 'data/motivated_reasoning/with_identity/transition_prob_n*_t*_*_k_z.txt'
    without_id_pattern = 'data/motivated_reasoning/without_identity/transition_prob_n*_t*_*_k_z.txt'
    
    # Find all matching files
    with_id_files = glob.glob(with_id_pattern)
    without_id_files = glob.glob(without_id_pattern)
    
    # If no files found, return
    if not with_id_files or not without_id_files:
        print("No transition probability files found.")
        return
    
    # Use the most recent files
    with_id_file = max(with_id_files, key=os.path.getmtime)
    without_id_file = max(without_id_files, key=os.path.getmtime)
    
    # Extract model and N info from filenames - Updated extraction
    filename = os.path.basename(with_id_file)
    
    # Extract N and T using regex to avoid confusion with other 't's in the filename
    n_match = re.search(r'_n(\d+)_', filename)
    t_match = re.search(r'_t([\d\.]+)_', filename)
    
    if n_match:
        N = n_match.group(1)
    else:
        N = "unknown"
        print(f"WARNING: Could not extract N from filename: {filename}")
    
    if t_match:
        T = t_match.group(1)
    else:
        T = "unknown"
        print(f"WARNING: Could not extract T from filename: {filename}")
    
    # Extract model name - everything between T value and _k_z.txt
    model_match = re.search(r'_t[\d\.]+_(.+?)_k_z\.txt', filename)
    if model_match:
        model = model_match.group(1)
    else:
        model = "unknown_model"
        print(f"WARNING: Could not extract model from filename: {filename}")
    
    # Load data - specify header=0 to recognize the file's header row
    with_id_data = pd.read_csv(with_id_file, header=0)
    without_id_data = pd.read_csv(without_id_file, header=0)
    
    # Fit the tanh function to both datasets
    try:
        with_id_popt, _ = curve_fit(tanh_fit, with_id_data['magnetization'], with_id_data['k_probability'])
        without_id_popt, _ = curve_fit(tanh_fit, without_id_data['magnetization'], without_id_data['k_probability'])
        
        beta_with_id = with_id_popt[0]
        beta_without_id = without_id_popt[0]
    except RuntimeError:
        print("Curve fitting failed. Using placeholder values.")
        beta_with_id = 1.0
        beta_without_id = 1.0
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot original data points with reduced alpha for visibility when overlapping
    ax.scatter(with_id_data['magnetization'], with_id_data['k_probability'], 
               label=f'With Identity (β={beta_with_id:.2f})', 
               marker='o', s=80, color='#1f77b4', alpha=0.7, zorder=3)
    ax.scatter(without_id_data['magnetization'], without_id_data['k_probability'], 
               label=f'Without Identity (β={beta_without_id:.2f})', 
               marker='s', s=80, color='#ff7f0e', alpha=0.7, zorder=2)
    
    # Plot fitted curves
    x_fit = np.linspace(-1, 1, 100)
    ax.plot(x_fit, tanh_fit(x_fit, beta_with_id), color='#1f77b4')
    ax.plot(x_fit, tanh_fit(x_fit, beta_without_id), color='#ff7f0e')
    
    # Add identity line for reference
    ax.plot([-1, 1], [0, 1], 'k--', alpha=0.3, label='Linear Response')
    
    # Annotations and styling
    ax.set_xlabel('Collective Opinion (m)')
    ax.set_ylabel('Adoption Probability P(m)')
    ax.set_title(f'Effect of Group Identity on Opinion Adoption\n{model}, N={N}, T={T}')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-0.05, 1.05])
    ax.legend(loc='best')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/transition_probabilities_{N}_{model}.png', dpi=300)
    plt.close()
    
    # Return the betas for further analysis
    return beta_with_id, beta_without_id

#----------------------------------------
# PLOT 2: Magnetization Time Series
#----------------------------------------
def plot_magnetization_time_series():
    # Find all magnetization files - Updated patterns
    with_id_pattern = 'data/motivated_reasoning/with_identity/magnetization_*_n*_t*_*.txt'
    without_id_pattern = 'data/motivated_reasoning/without_identity/magnetization_*_n*_t*_*.txt'
    
    with_id_files = glob.glob(with_id_pattern)
    without_id_files = glob.glob(without_id_pattern)
    
    if not with_id_files or not without_id_files:
        print("No magnetization time series files found.")
        return
    
    # Extract model and N info - Updated extraction
    filename = os.path.basename(with_id_files[0])
    
    # Extract N and T using regex
    n_match = re.search(r'_n(\d+)_', filename)
    t_match = re.search(r'_t([\d\.]+)_', filename)
    
    if n_match:
        N = n_match.group(1)
    else:
        N = "unknown"
        print(f"WARNING: Could not extract N from filename: {filename}")
    
    if t_match:
        T = t_match.group(1)
    else:
        T = "unknown"
        print(f"WARNING: Could not extract T from filename: {filename}")
    
    # Extract model name - everything between T value and .txt
    model_match = re.search(r'_t[\d\.]+_(.+?)\.txt', filename)
    if model_match:
        model = model_match.group(1)
    else:
        model = "unknown_model"
        print(f"WARNING: Could not extract model from filename: {filename}")
    
    # Group files by simulation number - Updated to match file pattern
    sim_groups = {}
    for with_file in with_id_files:
        sim_num = os.path.basename(with_file).split('_')[1]
        matching_without = [f for f in without_id_files if f'_{sim_num}_' in f]
        if matching_without:
            sim_groups[sim_num] = (with_file, matching_without[0])
    
    # If no matching pairs, return
    if not sim_groups:
        print("No matching pairs of magnetization files found.")
        return
    
    # Determine grid size based on number of simulations
    num_sims = len(sim_groups)
    cols = min(3, num_sims)  # Maximum 3 columns
    rows = (num_sims + cols - 1) // cols  # Ceiling division for number of rows
    
    # Create a grid of plots for all simulation runs
    fig = plt.figure(figsize=(6*cols, 5*rows))
    gs = GridSpec(rows, cols, figure=fig, hspace=0.3, wspace=0.3)
    
    for i, (sim_num, (with_file, without_file)) in enumerate(sorted(sim_groups.items())):
        # Load data
        with_data = pd.read_csv(with_file, header=None, names=['t', 'm'])
        without_data = pd.read_csv(without_file, header=None, names=['t', 'm'])
        
        # Rescale time by N for better comparability
        try:
            n_value = int(N)
        except ValueError:
            print(f"Warning: Could not convert N='{N}' to integer, using N=10 instead")
            n_value = 10  # Default if conversion fails
            
        with_data['t_scaled'] = with_data['t'] / n_value
        without_data['t_scaled'] = without_data['t'] / n_value
        
        # Create subplot
        row, col = i // cols, i % cols
        ax = fig.add_subplot(gs[row, col])
        
        # Plot both time series
        ax.plot(with_data['t_scaled'], with_data['m'], label='With Identity', color='#1f77b4')
        ax.plot(without_data['t_scaled'], without_data['m'], label='Without Identity', color='#ff7f0e', linestyle='--')
        
        # Annotations
        ax.set_xlabel('Scaled Time (t/N)')
        ax.set_ylabel('Magnetization m(t)')
        ax.set_title(f'Simulation Run {int(sim_num)+1}')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-1.1, 1.1])
        
        # Only add legend to the first subplot
        if i == 0:
            ax.legend(loc='best')
    
    # Add an overall title
    plt.suptitle(f'Opinion Evolution With and Without Group Identity\n{model}, N={N}, T={T}', fontsize=20)
    
    # Save the plot
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
    plt.savefig(f'{plots_dir}/magnetization_time_series_{N}_{model}.png', dpi=300)
    plt.close()

#----------------------------------------
# PLOT 3: Group Opinion Analysis
#----------------------------------------
def plot_group_opinion_dynamics():
    """
    Analyze group-level opinion dynamics from simulations with identity.
    """
    # Find all group opinion files
    group_pattern = 'data/motivated_reasoning/with_identity/group_opinions_*_n*_t*_*.txt'
    group_files = glob.glob(group_pattern)
    
    if not group_files:
        print("No group opinion files found.")
        return
    
    # More robust parameter extraction with error handling
    try:
        filename = os.path.basename(group_files[0])
        print(f"Processing file: {filename}")
        
        # First, try to extract N using regex pattern
        n_match = re.search(r'n(\d+)', filename)
        if n_match:
            N = n_match.group(1)
            print(f"Extracted N = {N}")
        else:
            # Fallback to split method with more careful handling
            filename_parts = filename.split('_')
            # Look for the part that starts with 'n' and extract the number
            for part in filename_parts:
                if part.startswith('n') and len(part) > 1:
                    N = part[1:]
                    print(f"Extracted N = {N} (fallback method)")
                    break
            else:
                # If we can't find N, use a default value and log a warning
                N = "10"  # Default to 10 as a reasonable fallback
                print(f"WARNING: Could not extract N from filename, using default N = {N}")
                
        # Extract model name - everything after t{T}_ until .txt
        model_match = re.search(r't[\d\.]+_(.+?)\.txt', filename)
        if model_match:
            model = model_match.group(1)
        else:
            model = "unknown_model"
            print("WARNING: Could not extract model name from filename")
            
    except Exception as e:
        print(f"Error parsing filename: {e}")
        print(f"Using default values instead")
        N = "10"  # Default value
        model = "unknown_model"
    
    # Load the data
    data = pd.read_csv(group_files[0], header=None, names=['t', 'A_k', 'A_z', 'B_k', 'B_z'], comment='#')
    
    # Calculate proportions within each group
    data['A_prop_k'] = data['A_k'] / (data['A_k'] + data['A_z'])
    data['B_prop_k'] = data['B_k'] / (data['B_k'] + data['B_z'])
    
    # Calculate rescaled time - with safer int conversion
    try:
        n_value = int(N)
    except ValueError:
        print(f"Warning: Could not convert N='{N}' to integer, using N=10 instead")
        n_value = 10  # Default if conversion fails
        
    data['t_scaled'] = data['t'] / n_value
    
    # Calculate group-level magnetization
    data['m_A'] = (data['A_k'] - data['A_z']) / (data['A_k'] + data['A_z'])
    data['m_B'] = (data['B_k'] - data['B_z']) / (data['B_k'] + data['B_z'])
    
    # Create a figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot 1: Proportion of opinion k in each group
    axes[0].plot(data['t_scaled'], data['A_prop_k'], label='Group A', color='#1f77b4')
    axes[0].plot(data['t_scaled'], data['B_prop_k'], label='Group B', color='#ff7f0e')
    axes[0].set_xlabel('Scaled Time (t/N)')
    axes[0].set_ylabel('Proportion Supporting Opinion k')
    axes[0].set_title('Opinion Distribution by Group')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([-0.05, 1.05])
    axes[0].legend()
    
    # Plot 2: Group-level magnetization
    axes[1].plot(data['t_scaled'], data['m_A'], label='Group A', color='#1f77b4')
    axes[1].plot(data['t_scaled'], data['m_B'], label='Group B', color='#ff7f0e')
    axes[1].plot(data['t_scaled'], data['m_A'] - data['m_B'], label='Difference', color='#2ca02c', linestyle='--')
    axes[1].set_xlabel('Scaled Time (t/N)')
    axes[1].set_ylabel('Group Magnetization')
    axes[1].set_title('Group-Level Opinion Polarity')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([-1.1, 1.1])
    axes[1].legend()
    
    # Add an overall title
    plt.suptitle(f'Group Opinion Dynamics in Motivated Reasoning\n{model}, N={N}')
    
    # Save the plot
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
    plt.savefig(f'{plots_dir}/group_opinion_dynamics_{N}_{model}.png', dpi=300)
    plt.close()

#----------------------------------------
# PLOT 4: Comparative Metrics
#----------------------------------------
def plot_comparative_metrics():
    # Look for summary file - Updated pattern
    summary_pattern = 'data/motivated_reasoning/comparative/summary_n*_t*_*.json'
    summary_files = glob.glob(summary_pattern)
    
    if not summary_files:
        print("No comparative metrics summary files found.")
        return
    
    # Use the most recent summary file
    summary_file = max(summary_files, key=os.path.getmtime)
    
    # Extract N and model info - Updated extraction
    filename_parts = os.path.basename(summary_file).split('_')
    N = filename_parts[1][1:]  # Remove the 'n' prefix
    T = filename_parts[2][1:]  # Remove the 't' prefix
    model = '_'.join(filename_parts[3:]).replace('.json', '')
    
    # Load the summary data
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    # Load individual simulation data - Updated pattern
    sim_pattern = f'data/motivated_reasoning/comparative/sim_*_n{N}_t{T}_{model}.json'
    sim_files = sorted(glob.glob(sim_pattern))
    
    # Extract individual simulation data
    consensus_times_with = []
    consensus_times_without = []
    opinion_changes_with = []
    opinion_changes_without = []
    
    for sim_file in sim_files:
        with open(sim_file, 'r') as f:
            sim_data = json.load(f)
        consensus_times_with.append(sim_data['with_identity']['time'])
        consensus_times_without.append(sim_data['without_identity']['time'])
        opinion_changes_with.append(sim_data['with_identity']['changes'])
        opinion_changes_without.append(sim_data['without_identity']['changes'])
    
    # Create a figure with bar charts
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # Simulation numbers for x-axis
    x = np.arange(len(sim_files))
    width = 0.35
    
    # Plot 1: Consensus Times
    axes[0].bar(x - width/2, consensus_times_with, width, label='With Identity', color='#1f77b4')
    axes[0].bar(x + width/2, consensus_times_without, width, label='Without Identity', color='#ff7f0e')
    axes[0].set_xlabel('Simulation Run')
    axes[0].set_ylabel('Consensus Time (iterations)')
    axes[0].set_title('Consensus Formation Times')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(i+1) for i in range(len(sim_files))])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Opinion Changes
    axes[1].bar(x - width/2, opinion_changes_with, width, label='With Identity', color='#1f77b4')
    axes[1].bar(x + width/2, opinion_changes_without, width, label='Without Identity', color='#ff7f0e')
    axes[1].set_xlabel('Simulation Run')
    axes[1].set_ylabel('Number of Opinion Changes')
    axes[1].set_title('Opinion Stability')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(i+1) for i in range(len(sim_files))])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add an overall title
    plt.suptitle(f'Comparative Metrics: Impact of Group Identity\n{model}, N={N}')
    
    # Add summary statistics to the figure
    summary_text = (
        f"Mean Consensus Time (With Identity): {summary['mean_consensus_time_with_identity']:.2f}\n"
        f"Mean Consensus Time (Without Identity): {summary['mean_consensus_time_without_identity']:.2f}\n"
        f"Mean Opinion Changes (With Identity): {summary['mean_opinion_changes_with_identity']:.2f}\n"
        f"Mean Opinion Changes (Without Identity): {summary['mean_opinion_changes_without_identity']:.2f}"
    )
    
    plt.figtext(0.5, 0.01, summary_text, ha='center', fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Save the plot
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Make room for suptitle and summary text
    plt.savefig(f'{plots_dir}/comparative_metrics_{N}_{model}.png', dpi=300)
    plt.close()

#----------------------------------------
# PLOT 5: Group-Opinion Alignment Impact
#----------------------------------------
def plot_alignment_impact():
    """
    Plot how varying group-opinion alignment probability affects decision-making,
    comparing between with and without identity conditions.
    """
    # Find alignment impact files for both conditions
    with_id_pattern = 'data/motivated_reasoning/with_identity/alignment_impact_n*_t*_*_k_z.txt'
    without_id_pattern = 'data/motivated_reasoning/without_identity/alignment_impact_n*_t*_*_k_z.txt'
    
    with_id_files = glob.glob(with_id_pattern)
    without_id_files = glob.glob(without_id_pattern)
    
    if not with_id_files:
        print("No alignment impact files found for with_identity condition.")
        return
    
    if not without_id_files:
        print("No alignment impact files found for without_identity condition.")
        return
    
    # Use the most recent files for both conditions
    with_id_file = max(with_id_files, key=os.path.getmtime)
    without_id_file = max(without_id_files, key=os.path.getmtime)
    
    # Extract simulation parameters
    filename = os.path.basename(with_id_file)
    n_match = re.search(r'_n(\d+)_', filename)
    t_match = re.search(r'_t([\d\.]+)_', filename)
    model_match = re.search(r'_t[\d\.]+_(.+?)_k_z\.txt', filename)
    
    N = n_match.group(1) if n_match else "unknown"
    T = t_match.group(1) if t_match else "unknown"
    model = model_match.group(1) if model_match else "unknown_model"
    
    # Load data for both conditions
    with_id_data = pd.read_csv(with_id_file)
    without_id_data = pd.read_csv(without_id_file)
    
    # Create a figure with TWO subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # PLOT 1: Group-Specific Aligned Opinion Adoption (WITH identity)
    ax1.plot(with_id_data['alignment_probability'], with_id_data['group_a_aligned_adoption'], 
            marker='o', linestyle='-', linewidth=2, markersize=8, 
            label='Group A', color='#1f77b4')
    
    ax1.plot(with_id_data['alignment_probability'], with_id_data['group_b_aligned_adoption'], 
            marker='s', linestyle='-', linewidth=2, markersize=8, 
            label='Group B', color='#ff7f0e')
    
    # Reference line at 0.5 (random choice)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random Choice (0.5)')
    
    ax1.set_xlabel('Group-Opinion Alignment Probability')
    ax1.set_ylabel('Probability of Adopting Aligned Opinion')
    ax1.set_title('Group-Specific Aligned Opinion Adoption\n(WITH Identity Visible)')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])
    ax1.set_xlim([0.45, 1.05])
    ax1.legend(loc='best')
    
    # PLOT 2: Compare k_probability between with and without identity
    ax2.plot(with_id_data['alignment_probability'], with_id_data['k_probability'], 
            marker='o', linestyle='-', linewidth=2, markersize=8, 
            label='With Identity', color='#1f77b4')
    
    ax2.plot(without_id_data['alignment_probability'], without_id_data['k_probability'], 
            marker='s', linestyle='-', linewidth=2, markersize=8, 
            label='Without Identity', color='#ff7f0e')
    
    # Reference line at 0.5 (random choice)
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random Choice (0.5)')
    
    ax2.set_xlabel('Group-Opinion Alignment Probability')
    ax2.set_ylabel('Probability of Adopting Opinion k')
    ax2.set_title('Opinion Adoption Probability\n(With vs. Without Identity)')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    ax2.set_xlim([0.45, 1.05])
    ax2.legend(loc='best')
    
    # Add an overall title
    plt.suptitle(f'Effect of Group-Opinion Alignment Probability on Decision Making\n{model}, N={N}, T={T}', fontsize=16)
    
    # Save the plot
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
    plt.savefig(f'{plots_dir}/alignment_impact_{N}_{model}.png', dpi=300)
    plt.close()

#----------------------------------------
# MAIN EXECUTION
#----------------------------------------
if __name__ == "__main__":
    print("Generating plots for motivated reasoning simulation...")
    
    # Generate all plots
    print("Plotting transition probabilities...")
    beta_with, beta_without = plot_transition_probabilities()
    print(f"Beta values - With Identity: {beta_with:.2f}, Without Identity: {beta_without:.2f}")
    
    print("Plotting magnetization time series...")
    plot_magnetization_time_series()
    
    print("Plotting group opinion dynamics...")
    plot_group_opinion_dynamics()
    
    print("Plotting comparative metrics...")
    plot_comparative_metrics()
    
    print("Plotting group-opinion alignment impact...")
    plot_alignment_impact()
    
    print("All plots generated and saved to:", plots_dir) 