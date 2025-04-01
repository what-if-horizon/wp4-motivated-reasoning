import openai
from openai import OpenAI
import random
import time
import threading
import numpy as np
import string
import os
import traceback
import json
import argparse
import dotenv  # Add this import for loading .env file

# Load environment variables from .env file
dotenv.load_dotenv()

# Constants
#MODEL = "gpt-4o-mini-2024-07-18"  # Using GPT-4o-mini specifically
MODEL = "gpt-4o-2024-11-20"
T = 0.7  # Temperature
GROUP_ALIGNMENT_PROB = 0.7  # Probability of initial opinion aligning with group identity

# Initialize the client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Create data directories with a dedicated folder for motivated reasoning
data_dirs = [
    'data/motivated_reasoning/with_identity', 
    'data/motivated_reasoning/without_identity', 
    'data/motivated_reasoning/comparative'
]
for dir_path in data_dirs:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# Utility Functions
def run_with_timeout(func, args=(), timeout_duration=25, default=None):
    """Runs the given function with the given args.
       If the function takes more than timeout_duration seconds, 
       default is returned instead."""
    
    class InterruptableThread(threading.Thread):
        def __init__(self, func, args):
            threading.Thread.__init__(self)
            self.func = func
            self.args = args
            self.result = default

        def run(self):
            try:
                self.result = self.func(*self.args)
            except Exception as e:
                self.result = default

    it = InterruptableThread(func, args)
    it.start()
    it.join(timeout_duration)
    
    # If thread is still alive after join(timeout), it's running for too long
    if it.is_alive():
        print('timeout')
        return default  # Return default value
    
    return it.result

def get_llm_response(prompt, model=MODEL, temperature=T):
    try:
        # New API call format
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        # Extract the content from the response
        return response.choices[0].message.content
    except Exception as e:
        print(f"API Error: {e}")
        return None

def generate_unique_random_strings(length, num_strings):
    """Generate a list of unique random strings."""
    generated_strings = set()
    chars = string.ascii_letters + string.digits
    
    while len(generated_strings) < num_strings:
        random_str = ''.join(random.choice(chars) for _ in range(length))
        generated_strings.add(random_str)

    return list(generated_strings)

def synchronized_shuffle(a, b):
    """Shuffle two lists in the same way."""
    assert len(a) == len(b)
    indices = list(range(len(a)))
    random.shuffle(indices)
    a_shuffled = [a[i] for i in indices]
    b_shuffled = [b[i] for i in indices]
    return a_shuffled, b_shuffled

def assign_identities(N, group_balance=0.5):
    """
    Assign agents to groups A and B.
    
    Parameters:
    N (int): Total number of agents
    group_balance (float): Proportion of agents in group A (0.5 means equal groups)
    
    Returns:
    list: Group assignments ('A' or 'B')
    """
    n_group_a = int(N * group_balance)
    n_group_b = N - n_group_a
    identities = ['A'] * n_group_a + ['B'] * n_group_b
    random.shuffle(identities)
    return identities

def initialize_opinions(N, identities, group_opinion_alignment_prob):
    """
    Initialize agent opinions with a specific level of alignment with group identity.
    
    Parameters:
    N (int): Number of agents
    identities (list): Group identities ('A' or 'B') for each agent
    group_opinion_alignment_prob (float): Probability that an agent's opinion aligns with their group's preferred opinion
    
    Returns:
    tuple: (opinions, identity_opinion_map)
    """
    opinions = []
    
    # Randomly choose which opinion is correlated with which identity
    if random.random() < 0.5:
        # Group A tends to opinion 'k', Group B tends to opinion 'z'
        identity_opinion_map = {'A': 'k', 'B': 'z'}
    else:
        # Group A tends to opinion 'z', Group B tends to opinion 'k'
        identity_opinion_map = {'A': 'z', 'B': 'k'}
    
    for identity in identities:
        aligned_opinion = identity_opinion_map[identity]
        opposed_opinion = 'z' if aligned_opinion == 'k' else 'k'
        
        # Assign opinions with alignment to identity
        if random.random() < group_opinion_alignment_prob:
            opinions.append(aligned_opinion)
        else:
            opinions.append(opposed_opinion)
    
    return opinions, identity_opinion_map

def create_identity_prompt(agent_names, agent_opinions, agent_groups, idx):
    """
    Create a prompt that includes group identity information in a natural way.
    
    Parameters:
    agent_names (list): Names of all agents
    agent_opinions (list): Opinions of all agents ('k' or 'z')
    agent_groups (list): Group assignments ('A' or 'B') for all agents
    idx (int): Index of the agent being updated
    
    Returns:
    str: The prompt with group identity information
    """
    # Get the current agent's group
    my_group = agent_groups[idx]
    
    # Base prompt with group information
    base_query = f"""Below you can see the list of all people in the community together with the opinion they support.
    You are a member of Group {my_group}.
    You must reply with the opinion you want to support.
    The opinion must be reported between square brackets."""
    
    # Create the list of people with their groups and opinions
    other_agents = []
    for i, (name, opinion, group) in enumerate(zip(agent_names, agent_opinions, agent_groups)):
        if i != idx:  # Skip the current agent
            other_agents.append(f"{name} (Group {group}): {opinion}")
    
    # Shuffle the list to avoid order bias
    random.shuffle(other_agents)
    
    # Combine into final prompt
    agent_list = "\n".join(other_agents)
    final_prompt = f"{base_query}\n\n{agent_list}\n\nReply only with the opinion you want to support, between square brackets."
    
    return final_prompt

def create_standard_prompt(agent_names, agent_opinions, idx):
    """
    Create a prompt without identity information (standard condition).
    
    Parameters:
    agent_names (list): Names of all agents
    agent_opinions (list): Opinions of all agents ('k' or 'z')
    idx (int): Index of the agent being updated
    
    Returns:
    str: The prompt without group identity information
    """
    base_query = """Below you can see the list of all people in the community together with the opinion they support.
    You must reply with the opinion you want to support.
    The opinion must be reported between square brackets."""
    
    # Create the list of people with their opinions
    other_agents = []
    for i, (name, opinion) in enumerate(zip(agent_names, agent_opinions)):
        if i != idx:  # Skip the current agent
            other_agents.append(f"{name}: {opinion}")
    
    # Shuffle the list to avoid order bias
    random.shuffle(other_agents)
    
    # Combine into final prompt
    agent_list = "\n".join(other_agents)
    final_prompt = f"{base_query}\n\n{agent_list}\n\nReply only with the opinion you want to support, between square brackets."
    
    return final_prompt

# =====================================================================
# PART 1: SINGLE AGENT TRANSITION PROBABILITY MEASUREMENT
# =====================================================================
# This section measures how individual agents respond to different distributions
# of opinions at various levels of overall agreement (magnetization).
# It creates probability curves showing the likelihood of adopting opinion 'k'
# at different magnetization values, both with and without group identity.
# =====================================================================

def measure_adoption_probability(prompt, model=MODEL, temperature=T):
    """
    Measure the probability of an agent adopting a particular opinion
    based on a single prompt.
    """
    response = get_llm_response(prompt, model, temperature)
    
    # Handle None responses
    if response is None:
        print("Received None response from API")
        return None
    
    # Extract opinion from response
    if "[" in response and "]" in response:
        # Extract content between brackets
        opinion = response.partition("[")[2].partition("]")[0]
        if opinion in ['k', 'z']:
            return opinion
    
    # If we get here, no valid opinion was found
    print("Could not extract valid opinion from response")
    return None

def generate_transition_curves(N, group_identity=False, group_alignment_prob=GROUP_ALIGNMENT_PROB):
    """
    Generate transition probability curves with or without group identity.
    
    This function tests how a single agent responds to different distributions
    of opinions in the population. For each magnetization value (representing
    different levels of majority opinion), it measures the probability of
    the agent adopting opinion 'k'.
    
    Parameters:
    N (int): Number of agents
    group_identity (bool): Whether to include group identity in the prompt
    group_alignment_prob (float): Probability of initial opinion aligning with group identity
    
    Returns:
    None (saves results to file)
    """
    # Custom magnetization values:  
    magnetization_values = [-0.95, -0.8, -0.6, -0.4, -0.2, 0.0, 0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 0.95]
    repetitions = 20  # Number of measurements for each magnetization value
    
    id_tag = "with_identity" if group_identity else "without_identity"
    filename = f"data/motivated_reasoning/{id_tag}/transition_prob_n{N}_t{T}_{MODEL}_k_z.txt"
    
    # Create a clean file
    with open(filename, 'w') as file:
        file.write("magnetization,k_probability\n")
    
    # Generate random agent names
    list_names = generate_unique_random_strings(5, N)
    
    for m0 in magnetization_values:
        print(f"Processing magnetization m0 = {m0}")
        
        # Calculate number of agents with each opinion based on magnetization
        num_k = int((m0 + 1) * N / 2)  # Convert magnetization to count
        num_z = N - num_k
        
        # Track transitions
        k_count = 0
        z_count = 0
        
        for _ in range(repetitions):
            # Create opinions list
            opinions = ['k'] * num_k + ['z'] * num_z
            random.shuffle(opinions)
            
            # Assign group identities if needed
            identities = None
            if group_identity:
                identities = assign_identities(N)
            
            # Select a random agent to measure transition
            idx = random.randint(0, N-1)
            current_opinion = opinions[idx]
            
            # Create appropriate prompt
            if group_identity:
                query = create_identity_prompt(list_names, opinions, identities, idx)
            else:
                query = create_standard_prompt(list_names, opinions, idx)
            
            # Get new opinion
            new_opinion = measure_adoption_probability(query)
            
            # Track transitions
            if new_opinion is not None:
                if new_opinion == 'k':
                    k_count += 1
                else:
                    z_count += 1
        
        # Calculate transition probability to opinion 'k'
        total = k_count + z_count
        k_probability = k_count / total if total > 0 else 0.0
        
        # Record the results
        with open(filename, 'a') as file:
            file.write(f"{m0},{k_probability}\n")
        
        print(f"Completed m0={m0}: k_probability={k_probability} ({k_count}/{total} measurements)")

# =====================================================================
# PART 1B: GROUP ALIGNMENT IMPACT MEASUREMENT
# =====================================================================
# This section examines how the alignment between group identity and
# opinions affects agent decision-making behavior. Unlike Part 1, which 
# varies the overall opinion distribution (magnetization), this part varies 
# the strength of group-opinion alignment from 0 to 1 to examine how 
# the strength of group alignment affects agent choices.
#
# For each alignment level, a population is created where group identity
# and opinions are correlated at the specified strength. Then a random agent
# is selected, and the LLM is queried to measure how it would update its
# opinion given the surrounding population distribution.
#
# Two conditions are compared:
# 1. With identity information: Agents can see group affiliations
# 2. Without identity information: Agents see only opinions
#
# The function tracks:
# - Raw opinion adoption (k vs z)
# - Group-specific opinion patterns
# - Aligned opinion adoption (how often agents adopt opinions aligned with their group)
# - Which opinion was randomly assigned to each group
#
# =====================================================================

def measure_identity_alignment_impact(N, group_identity=True, repetitions=20):
    """
    Measure how group-opinion alignment strength affects opinion dynamics.
    """
    print(f"\nMeasuring group-opinion alignment impact {'with' if group_identity else 'without'} group identity...")
    
    # Choose alignment probability values to test, from 0.5 to 1.0 at increments of 0.05
    group_alignment_values = [0.5 + 0.05 * i for i in range(11)]
    
    # Folder based on with/without identity
    folder = "with_identity" if group_identity else "without_identity"
    
    # Initialize output file with new columns for aligned opinion adoption
    output_file = f"data/motivated_reasoning/{folder}/group_alignment_impact_n{N}_t{T}_{MODEL}_k_z.txt"
    with open(output_file, 'w') as file:
        # Use the same column structure for both conditions for better comparison
        file.write("group_alignment_prob,k_probability,group_a_adoption,group_b_adoption,magnetization,"
                  "group_a_aligned_adoption,group_b_aligned_adoption,a_favors_k\n")
    
    # Test each alignment probability value
    for alignment_prob in group_alignment_values:
        print(f"Testing group alignment probability: {alignment_prob}")
        
        # Results for this alignment level - track all metrics in both conditions
        results = {
            'k_transitions': 0,
            'z_transitions': 0,
            'total_responses': 0,
            'group_a_k': 0,
            'group_a_z': 0,
            'group_b_k': 0,
            'group_b_z': 0,
            'group_a_aligned': 0,
            'group_b_aligned': 0,
            'a_favors_k_count': 0
        }
        
        # Run multiple repetitions at this alignment level
        for rep in range(repetitions):
            if rep % 5 == 0:
                print(f"  Progress: {rep}/{repetitions}")
                
            # Assign identities
            identities = assign_identities(N)
            
            # Initialize opinions with the specified alignment probability
            opinions, identity_opinion_map = initialize_opinions(N, identities, alignment_prob)
            
            # Track which opinion Group A favors
            a_favors_k = (identity_opinion_map['A'] == 'k')
            results['a_favors_k_count'] += 1 if a_favors_k else 0
            
            # Generate random agent names
            agent_names = generate_unique_random_strings(5, N)
            
            # Select a random agent to query
            target_idx = random.randint(0, N-1)
            
            # Create the appropriate prompt - with or without showing group identity
            if group_identity:
                query = create_identity_prompt(agent_names, opinions, identities, target_idx)
            else:
                query = create_standard_prompt(agent_names, opinions, target_idx)
            
            # Query the LLM to get a response
            response = measure_adoption_probability(query)
            
            # Skip if no valid response was received
            if response is None:
                continue
            
            # Track the response
            results['total_responses'] += 1
            
            # Agent's identity - tracked in both conditions for comparison
            agent_identity = identities[target_idx]
            aligned_opinion = identity_opinion_map[agent_identity]
            
            if response == 'k':
                results['k_transitions'] += 1
                if agent_identity == 'A':
                    results['group_a_k'] += 1
                else:
                    results['group_b_k'] += 1
            else:  # response == 'z'
                results['z_transitions'] += 1
                if agent_identity == 'A':
                    results['group_a_z'] += 1
                else:
                    results['group_b_z'] += 1
            
            # Track if the agent adopted its aligned opinion
            if response == aligned_opinion:
                if agent_identity == 'A':
                    results['group_a_aligned'] += 1
                else:
                    results['group_b_aligned'] += 1
        
        # Calculate average metrics
        total_responses = results['total_responses']
        if total_responses == 0:
            print(f"  Warning: No valid responses received for alignment level {alignment_prob}")
            with open(output_file, 'a') as file:
                file.write(f"{alignment_prob},0.5,0,0,0,0,0,0.5\n")
            continue
            
        k_probability = results['k_transitions'] / total_responses
        
        # Calculate magnetization
        magnetization = (results['k_transitions'] - results['z_transitions']) / total_responses
        
        # Write results to file - same format for both conditions
        with open(output_file, 'a') as file:
            # Calculate group-specific metrics
            group_a_total = results['group_a_k'] + results['group_a_z']
            group_b_total = results['group_b_k'] + results['group_b_z']
            
            group_a_k_adoption = results['group_a_k'] / group_a_total if group_a_total > 0 else 0
            group_b_k_adoption = results['group_b_k'] / group_b_total if group_b_total > 0 else 0
            
            group_a_aligned_adoption = results['group_a_aligned'] / group_a_total if group_a_total > 0 else 0
            group_b_aligned_adoption = results['group_b_aligned'] / group_b_total if group_b_total > 0 else 0
            
            a_favors_k_ratio = results['a_favors_k_count'] / repetitions
            
            file.write(f"{alignment_prob},{k_probability},{group_a_k_adoption},{group_b_k_adoption},"
                      f"{magnetization},{group_a_aligned_adoption},{group_b_aligned_adoption},{a_favors_k_ratio}\n")
    
    print(f"Group-opinion alignment impact measurements saved to: {output_file}")

# =====================================================================
# PART 2: MULTI-AGENT DYNAMIC SIMULATIONS
# =====================================================================
# This section simulates how opinions evolve over time in a population of
# interacting agents. The system starts with balanced opinions and agents
# update their opinions sequentially until either consensus is reached or
# a maximum number of iterations is hit. The simulation compares dynamics
# with and without group identity information.
# =====================================================================

def simulation_with_identity(N, n_sim, t_max, group_alignment_prob=GROUP_ALIGNMENT_PROB):
    """
    Run a dynamic multi-agent simulation with group identity information.
    
    This function tracks how opinions evolve over time in a population where
    agents are aware of group memberships. It updates agents sequentially and
    records the system state (magnetization, group opinions) over time.
    
    Parameters:
    N (int): Number of agents
    n_sim (int): Simulation number (for file naming)
    t_max (int): Maximum number of iterations
    group_alignment_prob (float): Probability of initial opinion aligning with group identity
    
    Returns:
    tuple: (final_magnetization, iterations, opinion_changes)
    """
    # Initialize with balanced opinions
    Na = int(N/2)
    Nb = N - Na
    
    # Assign group identities
    identities = assign_identities(N)
    
    # Initialize opinions with correlation to identities
    opinions, identity_opinion_map = initialize_opinions(N, identities, group_alignment_prob)
    
    # Count initial opinions
    k_count = opinions.count('k')
    z_count = opinions.count('z')
    
    # Calculate initial magnetization
    # Here we define k as +1 and z as -1
    m = (k_count - z_count) / N
    
    # Initialize counters
    i = 0
    opinion_changes = 0
    
    # Group metrics
    group_A_opinions = {'k': 0, 'z': 0}
    group_B_opinions = {'k': 0, 'z': 0}
    
    # Calculate initial group opinions
    for idx, identity in enumerate(identities):
        if identity == 'A':
            group_A_opinions[opinions[idx]] += 1
        else:
            group_B_opinions[opinions[idx]] += 1
    
    # Record initial state
    mag_filename = f"data/motivated_reasoning/with_identity/magnetization_{n_sim}_n{N}_t{T}_{MODEL}.txt"
    with open(mag_filename, 'a') as file:
        file.write(f"{i},{m}\n")
    
    # Also record group opinions and identity-opinion mapping
    group_filename = f"data/motivated_reasoning/with_identity/group_opinions_{n_sim}_n{N}_t{T}_{MODEL}.txt"
    with open(group_filename, 'a') as file:
        # Add header with identity-opinion mapping
        file.write(f"# identity_correlation: {group_alignment_prob}, Group A aligned with: {identity_opinion_map['A']}, Group B aligned with: {identity_opinion_map['B']}\n")
        file.write(f"{i},{group_A_opinions['k']},{group_A_opinions['z']},{group_B_opinions['k']},{group_B_opinions['z']}\n")
    
    # Generate random agent names
    list_names = generate_unique_random_strings(5, N)
    
    # Run the simulation
    while abs(m) < 1 and i < t_max:
        print(f"Iteration {i}, m = {m}")
        
        # Randomly select an agent to update
        idx = random.randint(0, N-1)
        selected_identity = identities[idx]
        selected_opinion = opinions[idx]
        
        # Create prompt with identity information
        query = create_identity_prompt(list_names, opinions, identities, idx)
        
        # Get response from the model
        check = 0
        attempt_count = 0
        new_opinion = None
        
        while check == 0 and attempt_count < 5:  # Try up to 5 times
            attempt_count += 1
            try:
                response = run_with_timeout(get_llm_response, (query, MODEL, T))
                if "[" in response and "]" in response:
                    y = response.partition("[")[2].partition("]")[0]
                    if y == 'k' or y == 'z':
                        new_opinion = y
                        check = 1
            except Exception as e:
                print("An error occurred:")
                traceback.print_exc()
                pass
        
        # If we got a valid response, update the agent's opinion
        if new_opinion is not None:
            # Check if opinion changed
            if new_opinion != selected_opinion:
                # Update counts
                if selected_opinion == 'k':
                    k_count -= 1
                    z_count += 1
                else:
                    k_count += 1
                    z_count -= 1
                
                # Update selected agent's opinion
                opinions[idx] = new_opinion
                
                # Track opinion change
                opinion_changes += 1
                
                # Update group metrics
                if selected_identity == 'A':
                    group_A_opinions[selected_opinion] -= 1
                    group_A_opinions[new_opinion] += 1
                else:
                    group_B_opinions[selected_opinion] -= 1
                    group_B_opinions[new_opinion] += 1
            
            # Calculate new magnetization
            m = (k_count - z_count) / N
            
            # Increment iteration counter
            i += 1
            
            # Record the state
            with open(mag_filename, 'a') as file:
                file.write(f"{i},{m}\n")
            
            # Record group opinions
            with open(group_filename, 'a') as file:
                file.write(f"{i},{group_A_opinions['k']},{group_A_opinions['z']},{group_B_opinions['k']},{group_B_opinions['z']}\n")
            
            # Check for consensus
            if abs(m) == 1:
                print('Consensus reached')
                break
    
    # Return final state and iterations (consensus time)
    return m, i, opinion_changes

def simulation_without_identity(N, n_sim, t_max):
    """
    Run a dynamic multi-agent simulation without group identity information.
    
    This function is the control condition for the identity simulation,
    where agents update opinions based only on the distribution of opinions
    in the population, without any group membership information.
    
    Parameters:
    N (int): Number of agents
    n_sim (int): Simulation number (for file naming)
    t_max (int): Maximum number of iterations
    
    Returns:
    tuple: Final magnetization, consensus time
    """
    # Initialize with balanced opinions
    Na = int(N/2)
    Nb = N - Na
    
    # Create opinions list
    opinions = ['k'] * Na + ['z'] * Nb
    random.shuffle(opinions)
    
    # Calculate initial magnetization
    k_count = opinions.count('k')
    z_count = opinions.count('z')
    m = (k_count - z_count) / N
    
    # Initialize counters
    i = 0
    opinion_changes = 0
    
    # Record initial state
    mag_filename = f"data/motivated_reasoning/without_identity/magnetization_{n_sim}_n{N}_t{T}_{MODEL}.txt"
    with open(mag_filename, 'a') as file:
        file.write(f"{i},{m}\n")
    
    # Generate random agent names
    list_names = generate_unique_random_strings(5, N)
    
    # Run the simulation
    while abs(m) < 1 and i < t_max:
        print(f"Iteration {i}, m = {m}")
        
        # Randomly select an agent to update
        idx = random.randint(0, N-1)
        selected_opinion = opinions[idx]
        
        # Formulate the prompt without group identity
        string_query = """Below you can see the list of all your friends together with the opinion they support.
        You must reply with the opinion you want to support.
        The opinion must be reported between square brackets."""
        
        # Prepare data for the prompt, excluding the selected agent
        temp_opinions = opinions[:]
        temp_names = list_names[:]
        
        # Remove the selected agent
        del temp_opinions[idx]
        del temp_names[idx]
        
        # Format the opinions
        temp_traits = [": " + str(x) + "\n" for x in temp_opinions]
        
        # Shuffle presentation order
        temp_names, temp_traits = synchronized_shuffle(temp_names, temp_traits)
        
        # Build the final query
        string_personalized = f"""{"".join([x + temp_traits[i] for i, x in enumerate(temp_names)])}
        Reply only with the opinion you want to support, between square brackets."""
        query = string_query + '\n' + string_personalized
        
        # Get response from the model
        check = 0
        attempt_count = 0
        new_opinion = None
        
        while check == 0 and attempt_count < 5:  # Try up to 5 times
            attempt_count += 1
            try:
                response = run_with_timeout(get_llm_response, (query, MODEL, T))
                if "[" in response and "]" in response:
                    y = response.partition("[")[2].partition("]")[0]
                    if y == 'k' or y == 'z':
                        new_opinion = y
                        check = 1
            except Exception as e:
                print("An error occurred:")
                traceback.print_exc()
                pass
        
        # If we got a valid response, update the agent's opinion
        if new_opinion is not None:
            # Check if opinion changed
            if new_opinion != selected_opinion:
                # Update counts
                if selected_opinion == 'k':
                    k_count -= 1
                    z_count += 1
                else:
                    k_count += 1
                    z_count -= 1
                
                # Update selected agent's opinion
                opinions[idx] = new_opinion
                
                # Track opinion change
                opinion_changes += 1
            
            # Calculate new magnetization
            m = (k_count - z_count) / N
            
            # Increment iteration counter
            i += 1
            
            # Record the state
            with open(mag_filename, 'a') as file:
                file.write(f"{i},{m}\n")
            
            # Check for consensus
            if abs(m) == 1:
                print('Consensus reached')
                break
    
    # Return final state and iterations (consensus time)
    return m, i, opinion_changes

def run_comparative_simulations(N, num_simulations=10, group_alignment_prob=GROUP_ALIGNMENT_PROB):
    """
    Run a set of comparative dynamic simulations with and without group identity.
    
    This function runs multiple simulations of both types (with and without identity)
    and compares metrics like consensus time, final opinion distribution, and
    opinion stability between the two conditions.
    
    Parameters:
    N (int): Number of agents
    num_simulations (int): Number of simulation runs
    group_alignment_prob (float): Probability of initial opinion aligning with group identity
    
    Returns:
    None (saves results to files)
    """
    t_max = 10 * N  # Maximum iterations set to 10*N
    
    # Results storage
    with_identity_results = {'consensus_time': [], 'final_magnetization': [], 'opinion_changes': []}
    without_identity_results = {'consensus_time': [], 'final_magnetization': [], 'opinion_changes': []}
    
    # Run simulations
    for i in range(num_simulations):
        print(f"\nRunning simulation {i+1}/{num_simulations}")
        
        # Run with identity
        print("\nRunning WITH identity:")
        m_with, t_with, changes_with = simulation_with_identity(N, i, t_max, group_alignment_prob)
        with_identity_results['consensus_time'].append(t_with)
        with_identity_results['final_magnetization'].append(abs(m_with))
        with_identity_results['opinion_changes'].append(changes_with)
        
        # Run without identity
        print("\nRunning WITHOUT identity:")
        m_without, t_without, changes_without = simulation_without_identity(N, i, t_max)
        without_identity_results['consensus_time'].append(t_without)
        without_identity_results['final_magnetization'].append(abs(m_without))
        without_identity_results['opinion_changes'].append(changes_without)
        
        # Calculate and save comparative metrics for this simulation
        comparative_metrics = {
            'consensus_time_difference': t_with - t_without,
            'magnetization_difference': abs(m_with) - abs(m_without),
            'stability_difference': changes_with - changes_without,
            'with_identity': {'time': t_with, 'magnetization': m_with, 'changes': changes_with},
            'without_identity': {'time': t_without, 'magnetization': m_without, 'changes': changes_without}
        }
        
        # Save comparative metrics
        with open(f"data/motivated_reasoning/comparative/sim_{i}_n{N}_t{T}_{MODEL}.json", 'w') as file:
            json.dump(comparative_metrics, file, indent=2)
    
    # Calculate summary statistics
    comparative_summary = {
        'mean_consensus_time_with_identity': np.mean(with_identity_results['consensus_time']),
        'mean_consensus_time_without_identity': np.mean(without_identity_results['consensus_time']),
        'mean_opinion_changes_with_identity': np.mean(with_identity_results['opinion_changes']),
        'mean_opinion_changes_without_identity': np.mean(without_identity_results['opinion_changes']),
        'mean_final_magnetization_with_identity': np.mean(with_identity_results['final_magnetization']),
        'mean_final_magnetization_without_identity': np.mean(without_identity_results['final_magnetization']),
        'num_simulations': num_simulations,
        'N': N,
        'T': T,
        'model': MODEL,
        'group_alignment_prob': group_alignment_prob
    }
    
    # Save summary
    with open(f"data/motivated_reasoning/comparative/summary_n{N}_t{T}_{MODEL}.json", 'w') as file:
        json.dump(comparative_summary, file, indent=2)
    
    print("\nComparative simulations complete!")
    print(f"Summary saved to data/motivated_reasoning/comparative/summary_n{N}_t{T}_{MODEL}.json")

def generate_parameter_report(N, num_simulations, group_alignment_prob):
    """
    Generate a detailed report of all parameter settings used in the simulation.
    
    Parameters:
    N (int): Number of agents
    num_simulations (int): Number of simulation runs
    group_alignment_prob (float): Probability of initial opinion aligning with group identity
    
    Returns:
    None (saves report to file)
    """
    # Create the report directory if it doesn't exist
    report_dir = "data/motivated_reasoning/reports"
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
    
    # Generate timestamp for the report
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    
    # Create filename
    report_filename = f"{report_dir}/simulation_report_{timestamp}.txt"
    html_report_filename = f"{report_dir}/simulation_report_{timestamp}.html"
    
    # Gather all parameter information
    params = {
        "Basic Configuration": {
            "Number of Agents": N,
            "Number of Simulations": num_simulations,
            "LLM Model": MODEL,
            "Temperature": T,
            "Group-Opinion Alignment": group_alignment_prob,
            "Random Seed": "Not fixed (uses system randomness)"
        },
        "Simulation Settings": {
            "Maximum Iterations": f"10*N = {10*N}",
            "Opinion Labels": "k and z (arbitrary labels to avoid bias)",
            "Data Storage Location": "data/motivated_reasoning/",
            "API Timeout": "25 seconds"
        },
        "Advanced Parameters": {
            "Group Balance": "0.5 (equal sized groups)",
            "Maximum API Retry Attempts": "5 per opinion update",
            "Magnetization Resolution": "11 points from -1.0 to 1.0",
            "Repetitions per Magnetization Point": "10"
        }
    }
    
    # Generate text report
    with open(report_filename, 'w') as file:
        file.write("=======================================================\n")
        file.write("          MOTIVATED REASONING SIMULATION REPORT         \n")
        file.write("=======================================================\n")
        file.write(f"Generated: {timestamp.replace('_', ' ')}\n\n")
        
        for section, section_params in params.items():
            file.write(f"\n{section}\n")
            file.write("-" * len(section) + "\n")
            for param, value in section_params.items():
                file.write(f"{param}: {value}\n")
        
        file.write("\n=======================================================\n")
        file.write("                  SIMULATION PURPOSE                    \n")
        file.write("=======================================================\n")
        file.write("This simulation investigates how group identity affects opinion\n")
        file.write("dynamics in LLM agents. It compares standard opinion dynamics\n")
        file.write("(without identity information) to a scenario where agents are\n")
        file.write("aware of group membership, testing for motivated reasoning effects.\n\n")
        
        file.write("The simulation measures:\n")
        file.write("1. Transition probability curves with/without identity\n")
        file.write("2. Consensus time and opinion stability\n")
        file.write("3. Group opinion polarization\n")
        file.write("4. Magnetization (opinion balance) over time\n")
    
    # Generate HTML report (more visually appealing)
    with open(html_report_filename, 'w') as file:
        file.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Motivated Reasoning Simulation Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }
        h1, h2 {
            color: #2c3e50;
        }
        .header {
            background-color: #f5f5f5;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            text-align: center;
        }
        .section {
            margin-bottom: 30px;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }
        th, td {
            padding: 10px;
            border-bottom: 1px solid #ddd;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
        }
        .footer {
            text-align: center;
            margin-top: 30px;
            font-style: italic;
            color: #777;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Motivated Reasoning Simulation Report</h1>
        <p>Generated: """ + timestamp.replace('_', ' ') + """</p>
    </div>
""")

        for section, section_params in params.items():
            file.write(f"""
    <div class="section">
        <h2>{section}</h2>
        <table>
            <tr>
                <th>Parameter</th>
                <th>Value</th>
            </tr>
""")
            for param, value in section_params.items():
                file.write(f"""
            <tr>
                <td>{param}</td>
                <td>{value}</td>
            </tr>""")
            
            file.write("""
        </table>
    </div>""")
        
        file.write("""
    <div class="section">
        <h2>Simulation Purpose</h2>
        <p>This simulation investigates how group identity affects opinion dynamics in LLM agents. It compares standard opinion dynamics (without identity information) to a scenario where agents are aware of group membership, testing for motivated reasoning effects.</p>
        
        <h3>The simulation measures:</h3>
        <ol>
            <li>Transition probability curves with/without identity</li>
            <li>Consensus time and opinion stability</li>
            <li>Group opinion polarization</li>
            <li>Magnetization (opinion balance) over time</li>
        </ol>
    </div>
    
    <div class="footer">
        <p>Part of the Motivated Reasoning in LLMs research project</p>
    </div>
</body>
</html>""")
    
    print(f"\nParameter report generated:")
    print(f"- Text report: {report_filename}")
    print(f"- HTML report: {html_report_filename}")
    
    return report_filename, html_report_filename

# Main execution
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run motivated reasoning simulations')
    parser.add_argument('--num-agents', type=int, default=20, help='Number of agents')
    parser.add_argument('--simulations', type=int, default=5, help='Number of simulation runs')
    parser.add_argument('--model', type=str, default=MODEL, help='LLM model to use')
    parser.add_argument('--temperature', type=float, default=T, help='Temperature parameter')
    parser.add_argument('--group-alignment', type=float, default=GROUP_ALIGNMENT_PROB, 
                        help='Probability of initial opinion aligning with group identity')
    args = parser.parse_args()
    
    # Update parameters (without using global)
    model = args.model
    temperature = args.temperature
    group_alignment_prob = args.group_alignment
    
    # Set module-level variables for other functions to use
    MODEL = model
    T = temperature
    GROUP_ALIGNMENT_PROB = group_alignment_prob
    
    print(f"Running motivated reasoning simulation with {args.num_agents} agents, {args.simulations} simulations")
    print(f"Model: {model}, Temperature: {temperature}, Group-Opinion Alignment: {group_alignment_prob}")
    
    # Create directories if they don't exist
    for dir_path in ["data/motivated_reasoning/with_identity", 
                     "data/motivated_reasoning/without_identity",
                     "data/motivated_reasoning/comparative",
                     "data/motivated_reasoning/reports"]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    # PART 1: Generate transition probability curves
    print("\n=== PART 1: Generating transition probability curves ===")
    generate_transition_curves(args.num_agents, group_identity=False)
    generate_transition_curves(args.num_agents, group_identity=True)
    
    # PART 1B: Measure identity correlation impact
    print("\n=== PART 1B: Measuring identity correlation impact ===")
    measure_identity_alignment_impact(args.num_agents, group_identity=True)
    measure_identity_alignment_impact(args.num_agents, group_identity=False)
    
    # PART 2: Run comparative dynamic simulations
    print("\n=== PART 2: Running comparative dynamic simulations ===")
    run_comparative_simulations(args.num_agents, args.simulations, args.group_alignment)
    
    # Generate parameter report
    generate_parameter_report(args.num_agents, args.simulations, args.group_alignment)
    
    print("\nMotivated reasoning simulation complete!") 