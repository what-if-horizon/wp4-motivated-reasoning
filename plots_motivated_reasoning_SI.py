import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import json
from scipy.optimize import curve_fit
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
from scipy.stats import ttest_ind, pearsonr
import re

# Create output directory for plots
plots_dir = 'data/motivated_reasoning/plots/SI'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Define the tanh fitting function for transition probability
def tanh_fit(x, beta):
    return 0.5 * (np.tanh(beta * x) + 1)

# Increase font size for all plots
plt.rcParams.update({'font.size': 18})

#----------------------------------------
# PLOT 1: Effect of Group Balance
#----------------------------------------
def plot_group_balance_effect():
    """
    Compare transition probability curves for different group balance settings.
    This is a supplementary analysis that would require running the simulation
    with different group balance parameters.
    """
    # This plot would need data from simulations with different group balances
    # For now, we'll create a placeholder that explains how to use this function
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Create a placeholder message
    ax.text(0.5, 0.5, 
            "This plot requires data from multiple runs with different group balance settings.\n\n"
            "To generate this plot:\n"
            "1. Run the simulation with different group_balance values (e.g., 0.3, 0.5, 0.7)\n"
            "2. Update the group_balance parameter in motivated_reasoning_simulation.py\n"
            "3. Run the simulation for each setting\n"
            "4. Re-run this plotting script", 
            horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes, fontsize=14,
            bbox=dict(facecolor='#f9f9f9', alpha=0.8, boxstyle='round,pad=0.5'))
    
    ax.set_title('Effect of Group Balance on Opinion Dynamics')
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.savefig(f'{plots_dir}/group_balance_effect_placeholder.png', dpi=300)
    plt.close()
    
    # Note: When actual data becomes available, implement the real plot
    print("Created placeholder for group balance effect plot.")

#----------------------------------------
# PLOT 2: Individual-Level Opinion Flips
#----------------------------------------
def plot_opinion_flip_analysis():
    """
    Analyze opinion flips at the individual level to understand when and why agents change opinions.
    """
    # Find all magnetization files from identity condition - Updated pattern
    mag_pattern = 'data/motivated_reasoning/with_identity/magnetization_*_n*_t*_*.txt'
    group_pattern = 'data/motivated_reasoning/with_identity/group_opinions_*_n*_t*_*.txt'
    
    mag_files = glob.glob(mag_pattern)
    group_files = glob.glob(group_pattern)
    
    if not mag_files or not group_files:
        print("Insufficient data for opinion flip analysis.")
        return
    
    # Match files from the same simulation run - Updated extraction
    sim_data = {}
    for mag_file in mag_files:
        filename_parts = os.path.basename(mag_file).split('_')
        sim_num = filename_parts[1]
        N = filename_parts[2][1:]  # Remove the 'n' prefix
        T = filename_parts[3][1:]  # Remove the 't' prefix
        model = filename_parts[4].replace('.txt', '')
        
        matching_group = [f for f in group_files if f'_{sim_num}_n{N}_t{T}_' in f]
        if matching_group:
            sim_data[sim_num] = {
                'mag_file': mag_file,
                'group_file': matching_group[0],
                'N': N,
                'model': model
            }
    
    if not sim_data:
        print("No matching simulation files found.")
        return
    
    # Use the first available simulation
    sim_num = list(sim_data.keys())[0]
    sim_info = sim_data[sim_num]
    
    # Load the data
    mag_data = pd.read_csv(sim_info['mag_file'], header=None, names=['t', 'm'])
    group_data = pd.read_csv(sim_info['group_file'], header=None, names=['t', 'A_k', 'A_z', 'B_k', 'B_z'])
    
    # Calculate opinion changes by looking at consecutive differences in magnetization
    mag_data['m_diff'] = mag_data['m'].diff()
    
    # Calculate the absolute change in each group
    group_data['A_k_diff'] = group_data['A_k'].diff()
    group_data['A_z_diff'] = group_data['A_z'].diff()
    group_data['B_k_diff'] = group_data['B_k'].diff()
    group_data['B_z_diff'] = group_data['B_z'].diff()
    
    # Identify flips (where magnetization changes)
    flip_points = mag_data[mag_data['m_diff'].abs() > 0.01].copy()
    flip_points['scaled_t'] = flip_points['t'] / int(sim_info['N'])
    
    # Create the plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [2, 1]})
    
    # Plot 1: Magnetization with flip points highlighted
    axes[0].plot(mag_data['t'] / int(sim_info['N']), mag_data['m'], color='#1f77b4', label='Magnetization')
    axes[0].scatter(flip_points['scaled_t'], flip_points['m'], color='red', s=80, marker='o', label='Opinion Flips')
    
    # Add horizontal lines at ±0.5 where majority opinion shifts
    axes[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    axes[0].axhline(y=-0.5, color='gray', linestyle='--', alpha=0.5)
    axes[0].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    
    axes[0].set_xlabel('Scaled Time (t/N)')
    axes[0].set_ylabel('Magnetization m(t)')
    axes[0].set_title('Opinion Flips in Identity Condition')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Histogram of flip timings
    num_bins = min(20, len(flip_points))
    if num_bins > 0:
        axes[1].hist(flip_points['scaled_t'], bins=num_bins, color='#ff7f0e', alpha=0.7)
        axes[1].set_xlabel('Scaled Time (t/N)')
        axes[1].set_ylabel('Number of Opinion Flips')
        axes[1].set_title('Distribution of Opinion Flip Times')
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "Insufficient opinion flips for histogram", 
                     ha='center', va='center', transform=axes[1].transAxes)
    
    # Add overall title
    plt.suptitle(f'Detailed Analysis of Opinion Flips\n{sim_info["model"]}, N={sim_info["N"]}, Simulation #{int(sim_num)+1}', 
                 fontsize=20)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
    plt.savefig(f'{plots_dir}/opinion_flip_analysis_{sim_info["N"]}_{sim_info["model"]}.png', dpi=300)
    plt.close()
    
    print(f"Created opinion flip analysis plot for simulation {int(sim_num)+1}.")

#----------------------------------------
# PLOT 3: Statistical Comparison
#----------------------------------------
def plot_statistical_comparison():
    """
    Create statistical comparisons between with-identity and without-identity conditions.
    """
    # Find all comparative metrics - Updated pattern
    summary_pattern = 'data/motivated_reasoning/comparative/summary_n*_t*_*.json'
    summary_files = glob.glob(summary_pattern)
    
    if not summary_files:
        print("No summary files found for statistical comparison.")
        return
    
    # Load all summaries - Updated extraction
    summaries = []
    for summary_file in summary_files:
        filename_parts = os.path.basename(summary_file).split('_')
        N = filename_parts[1][1:]  # Remove the 'n' prefix
        T = filename_parts[2][1:]  # Remove the 't' prefix
        model = '_'.join(filename_parts[3:]).replace('.json', '')  # Join in case model name has underscores
        
        with open(summary_file, 'r') as f:
            data = json.load(f)
            data['N'] = N
            data['T'] = T
            data['model'] = model
            summaries.append(data)
    
    # If only one summary is available, we'll need to load the individual simulation data
    if len(summaries) == 1:
        summary = summaries[0]
        
        # Load individual simulation data for statistical testing - Updated pattern
        sim_pattern = f'data/motivated_reasoning/comparative/sim_*_n{summary["N"]}_t{summary["T"]}_{summary["model"]}.json'
        sim_files = sorted(glob.glob(sim_pattern))
        
        # Extract data for statistical tests
        times_with = []
        times_without = []
        changes_with = []
        changes_without = []
        
        for sim_file in sim_files:
            with open(sim_file, 'r') as f:
                sim_data = json.load(f)
            times_with.append(sim_data['with_identity']['time'])
            times_without.append(sim_data['without_identity']['time'])
            changes_with.append(sim_data['with_identity']['changes'])
            changes_without.append(sim_data['without_identity']['changes'])
        
        # Calculate statistics
        mean_time_with = np.mean(times_with)
        mean_time_without = np.mean(times_without)
        mean_changes_with = np.mean(changes_with)
        mean_changes_without = np.mean(changes_without)
        
        # Perform t-tests
        time_ttest = ttest_ind(times_with, times_without)
        changes_ttest = ttest_ind(changes_with, changes_without)
        
        # Create statistical comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        
        # Plot means with error bars
        axes[0].bar([0, 1], [mean_time_with, mean_time_without], 
                   yerr=[np.std(times_with), np.std(times_without)],
                   color=['#1f77b4', '#ff7f0e'])
        axes[0].set_xticks([0, 1])
        axes[0].set_xticklabels(['With Identity', 'Without Identity'])
        axes[0].set_ylabel('Mean Consensus Time (iterations)')
        axes[0].set_title(f'Consensus Time Comparison\np={time_ttest.pvalue:.4f}')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        axes[1].bar([0, 1], [mean_changes_with, mean_changes_without], 
                   yerr=[np.std(changes_with), np.std(changes_without)],
                   color=['#1f77b4', '#ff7f0e'])
        axes[1].set_xticks([0, 1])
        axes[1].set_xticklabels(['With Identity', 'Without Identity'])
        axes[1].set_ylabel('Mean Number of Opinion Changes')
        axes[1].set_title(f'Opinion Stability Comparison\np={changes_ttest.pvalue:.4f}')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Add overall title
        plt.suptitle(f'Statistical Comparison of Motivated Reasoning Effects\n{summary["model"]}, N={summary["N"]}', 
                     fontsize=20)
        
        # Add significance stars if applicable
        for i, pvalue in enumerate([time_ttest.pvalue, changes_ttest.pvalue]):
            if pvalue < 0.001:
                axes[i].text(0.5, 0.9, '***', ha='center', va='center', transform=axes[i].transAxes, fontsize=24)
            elif pvalue < 0.01:
                axes[i].text(0.5, 0.9, '**', ha='center', va='center', transform=axes[i].transAxes, fontsize=24)
            elif pvalue < 0.05:
                axes[i].text(0.5, 0.9, '*', ha='center', va='center', transform=axes[i].transAxes, fontsize=24)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
        plt.savefig(f'{plots_dir}/statistical_comparison_{summary["N"]}_{summary["model"]}.png', dpi=300)
        plt.close()
        
        print(f"Created statistical comparison plot for {summary['model']}, N={summary['N']}.")
    
    # If multiple summaries are available, we can compare across different parameters
    else:
        # To be implemented when more data becomes available
        print("Multiple simulation summaries found. Extended comparison not yet implemented.")

#----------------------------------------
# PLOT 4: In-Group vs. Out-Group Influence
#----------------------------------------
def plot_group_influence_analysis():
    """
    Analyze how in-group vs. out-group opinions influence decision-making.
    """
    # Find group opinion files - Updated pattern
    group_pattern = 'data/motivated_reasoning/with_identity/group_opinions_*_n*_t*_*.txt'
    group_files = glob.glob(group_pattern)
    
    if not group_files:
        print("No group opinion files found.")
        return
    
    # Select the first file for analysis
    group_file = group_files[0]
    
    # More robust parameter extraction with error handling
    try:
        filename = os.path.basename(group_file)
        print(f"Processing file: {filename}")
        
        # Extract simulation number
        sim_num_match = re.search(r'opinions_(\d+)_', filename)
        if sim_num_match:
            sim_num = sim_num_match.group(1)
        else:
            sim_num = "0"
            print(f"WARNING: Could not extract simulation number from filename: {filename}")
            
        # Extract N using regex
        n_match = re.search(r'_n(\d+)_', filename)
        if n_match:
            N = n_match.group(1)
            print(f"Extracted N = {N}")
        else:
            N = "10"  # Default to 10 as a reasonable fallback
            print(f"WARNING: Could not extract N from filename, using default N = {N}")
            
        # Extract T using regex
        t_match = re.search(r'_t([\d\.]+)_', filename)
        if t_match:
            T = t_match.group(1)
        else:
            T = "unknown"
            print(f"WARNING: Could not extract T from filename: {filename}")
        
        # Extract model name
        model_match = re.search(r'_t[\d\.]+_(.+?)\.txt', filename)
        if model_match:
            model = model_match.group(1)
        else:
            model = "unknown_model"
            print(f"WARNING: Could not extract model from filename: {filename}")
            
    except Exception as e:
        print(f"Error parsing filename: {e}")
        print(f"Using default values instead")
        sim_num = "0"
        N = "10"  # Default value
        model = "unknown_model"
    
    # Load the data
    group_data = pd.read_csv(group_file, header=None, names=['t', 'A_k', 'A_z', 'B_k', 'B_z'])
    
    # Calculate scaled time with safer int conversion
    try:
        n_value = int(N)
    except ValueError:
        print(f"Warning: Could not convert N='{N}' to integer, using N=10 instead")
        n_value = 10  # Default if conversion fails
        
    group_data['t_scaled'] = group_data['t'] / n_value
    
    # Calculate group proportions and magnetizations
    group_data['A_total'] = group_data['A_k'] + group_data['A_z']
    group_data['B_total'] = group_data['B_k'] + group_data['B_z']
    group_data['A_prop_k'] = group_data['A_k'] / group_data['A_total']
    group_data['B_prop_k'] = group_data['B_k'] / group_data['B_total']
    group_data['total_prop_k'] = (group_data['A_k'] + group_data['B_k']) / (group_data['A_total'] + group_data['B_total'])
    
    # Calculate differences between groups
    group_data['prop_diff'] = group_data['A_prop_k'] - group_data['B_prop_k']
    
    # Create the plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))
    
    # Plot 1: Opinion proportions in each group and overall
    axes[0].plot(group_data['t_scaled'], group_data['A_prop_k'], label='Group A', color='#1f77b4')
    axes[0].plot(group_data['t_scaled'], group_data['B_prop_k'], label='Group B', color='#ff7f0e')
    axes[0].plot(group_data['t_scaled'], group_data['total_prop_k'], label='Overall', color='#2ca02c', linewidth=2)
    
    axes[0].set_xlabel('Scaled Time (t/N)')
    axes[0].set_ylabel('Proportion Supporting Opinion k')
    axes[0].set_title('Group Opinion Evolution')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([-0.05, 1.05])
    axes[0].legend()
    
    # Plot 2: Difference between group opinion proportions
    axes[1].plot(group_data['t_scaled'], group_data['prop_diff'], color='#d62728', linewidth=2)
    axes[1].fill_between(group_data['t_scaled'], 0, group_data['prop_diff'], 
                        where=(group_data['prop_diff'] > 0), color='#d62728', alpha=0.3)
    axes[1].fill_between(group_data['t_scaled'], 0, group_data['prop_diff'], 
                        where=(group_data['prop_diff'] < 0), color='#d62728', alpha=0.3)
    
    axes[1].set_xlabel('Scaled Time (t/N)')
    axes[1].set_ylabel('Difference in Opinion Proportion (A - B)')
    axes[1].set_title('Group Opinion Divergence')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([-1.05, 1.05])
    axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Add a horizontal line at 0
    axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Add overall title
    plt.suptitle(f'In-Group vs. Out-Group Influence Analysis\n{model}, N={N}, Simulation #{int(sim_num)+1}', 
                 fontsize=20)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
    plt.savefig(f'{plots_dir}/group_influence_analysis_{N}_{model}.png', dpi=300)
    plt.close()
    
    print(f"Created group influence analysis plot for {model}, N={N}.")

#----------------------------------------
# MAIN EXECUTION
#----------------------------------------
if __name__ == "__main__":
    print("Generating supplementary plots for motivated reasoning simulation...")
    
    # Generate all SI plots
    print("Plotting group balance effect placeholder...")
    plot_group_balance_effect()
    
    print("Plotting opinion flip analysis...")
    plot_opinion_flip_analysis()
    
    print("Plotting statistical comparison...")
    plot_statistical_comparison()
    
    print("Plotting group influence analysis...")
    plot_group_influence_analysis()
    
    print("All supplementary plots generated and saved to:", plots_dir) 