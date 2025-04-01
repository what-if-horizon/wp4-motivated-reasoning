# LLMs-Opinion-Dynamics: Motivated Reasoning in Large Language Models

This repository contains code for studying opinion dynamics and motivated reasoning in Large Language Models (LLMs).

## Overview

This project investigates how group identity affects opinion dynamics in LLM agents. It simulates multi-agent systems where agents can see (or not see) group affiliations, and measures how this impacts their opinion choices, consensus formation, and overall system behavior. The simulation examines whether LLMs exhibit motivated reasoning - the tendency to process information in a way that supports pre-existing beliefs or group identities.

## Key Features

- Transition probability measurements for single agents
- Multi-agent dynamic simulations with and without group identity
- Group-opinion alignment impact analysis
- Comprehensive visualization of results

## Requirements

- Python 3.7+
- OpenAI API key (stored in a `.env` file)
- Required packages: `openai`, `numpy`, `pandas`, `matplotlib`, `scipy`

## Setup

1. Clone this repository
2. Create a `.env` file in the root directory with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
3. Install required packages:
   ```
   pip install openai numpy pandas matplotlib scipy python-dotenv
   ```

## Main Scripts

### motivated_reasoning_simulation.py

This script performs the core simulations and data collection. It has several major components:

1. **Single-Agent Transition Probability Measurement**: Measures how individual agents respond to different distributions of opinions in the population with and without group identity information.

2. **Group Alignment Impact Measurement**: Examines how the alignment between group identity and opinions affects agent decision-making. This varies the strength of group-opinion correlation and measures its impact on agent choices.

3. **Multi-Agent Dynamic Simulations**: Simulates how opinions evolve over time in a population of interacting agents. The system starts with balanced opinions, and agents update sequentially until consensus is reached or a maximum number of iterations is hit.

Key parameters:
- `--num-agents`: Number of agents in the simulation (default: 20)
- `--simulations`: Number of simulation runs (default: 5)
- `--model`: LLM model to use (default: gpt-4o-2024-11-20)
- `--temperature`: Temperature parameter for the LLM (default: 0.7)
- `--group-alignment`: Probability of initial opinion aligning with group identity (default: 0.7)

Example usage:
```bash
python motivated_reasoning_simulation.py --num-agents 15 --simulations 3
```

### plots_motivated_reasoning_main.py

This script generates visualizations from the data collected by the simulation script. It creates five main types of plots:

1. **Transition Probability Curves**: Shows how the probability of adopting opinion 'k' varies with the overall magnetization (opinion balance) of the system, both with and without identity information.

2. **Magnetization Time Series**: Displays how the magnetization (overall opinion balance) evolves over time in multiple simulations, comparing the with-identity and without-identity conditions.

3. **Group Opinion Dynamics**: Analyzes group-level opinion patterns, showing opinion distributions by group and group-level polarization over time.

4. **Comparative Metrics**: Compares key metrics like consensus time and opinion stability between the two conditions.

5. **Group-Opinion Alignment Impact**: Visualizes how varying the strength of group-opinion alignment affects decision-making in both conditions.

Example usage:
```bash
python plots_motivated_reasoning_main.py
```

### plots_motivated_reasoning_SI.py

This script generates supplementary visualizations for more in-depth analysis:

1. **Group Balance Effect**: Analyzes how the relative sizes of groups affect opinion dynamics.

2. **Opinion Flip Analysis**: Examines when and why agents change their opinions at the individual level.

3. **Statistical Comparison**: Provides statistical tests comparing the with-identity and without-identity conditions.

4. **In-Group vs. Out-Group Influence**: Analyzes how opinions from an agent's own group versus other groups influence decision-making.

Example usage:
```bash
python plots_motivated_reasoning_SI.py
```

### Shell Scripts

The repository includes shell scripts to automate the execution of multiple components:

#### run_motivated_reasoning_pipeline.sh

This script runs the complete motivated reasoning simulation pipeline from data generation to analysis and visualization.

Features:
- Checks for required dependencies and installs them if needed
- Offers options to clean previous data before starting
- Runs all three phases of the simulation (transition probabilities, alignment impact, dynamic simulations)
- Automatically generates all plots using both main and supplementary plotting scripts
- Provides detailed progress information and timing statistics

Parameters:
- `-n, --num-agents NUM`: Number of agents (default: 20)
- `-s, --simulations NUM`: Number of simulation runs (default: 5)
- `-m, --model MODEL`: LLM model to use (default: gpt-4o-mini-2024-07-18)
- `-t, --temperature TEMP`: Temperature parameter (default: 0.7)
- `-a, --group-alignment-prob PROB`: Group-opinion alignment probability (default: 0.7)
- `-c, --clean`: Clean previous data before running
- `-h, --help`: Display help message

Example usage:
```bash
./run_motivated_reasoning_pipeline.sh --num-agents 15 --simulations 3 --model "gpt-4o-2024-11-20"
```

#### run_group_alignment_analysis.sh

This script focuses specifically on the group alignment impact analysis, running only that component of the simulation and generating the associated visualizations.

Features:
- Uses a dedicated test directory for output to avoid overwriting main results
- Creates specialized plots showing how group-opinion alignment affects decision-making
- Requires less API calls than the full pipeline, making it useful for focused experiments

Parameters:
- `-n, --num-agents NUM`: Number of agents (default: 20)
- `-m, --model MODEL`: LLM model to use (default: gpt-4o-mini-2024-07-18)
- `-t, --temperature TEMP`: Temperature parameter (default: 0.7)
- `-g, --group-alignment PROB`: Group-opinion alignment probability (default: 0.95)
- `-c, --clean`: Clean previous group alignment data before running
- `-h, --help`: Display help message

Example usage:
```bash
./run_group_alignment_analysis.sh --num-agents 15 --model "gpt-4o-2024-11-20" --group-alignment 0.8
```

## Data Structure

The simulation generates data in several directories:
- `data/motivated_reasoning/with_identity/`: Data from simulations where agents can see group identities
- `data/motivated_reasoning/without_identity/`: Data from simulations where agents only see opinions
- `data/motivated_reasoning/comparative/`: Comparative metrics between the two conditions
- `data/motivated_reasoning/plots/`: Generated visualizations
- `data/motivated_reasoning/reports/`: Parameter reports and simulation summaries

## How It Works

1. The simulation uses OpenAI's API to query LLMs about opinion choices
2. Agents are shown a population with existing opinions, and asked to state their own
3. In the "with identity" condition, agents also see group affiliations (Group A or Group B)
4. The simulation measures how often agents align with their group's favored opinion
5. Plots are generated to visualize patterns and compare between conditions

## Results Interpretation

The key metrics to examine in the results:
- **β values** in transition probability curves: Higher β values indicate stronger conformity
- **Consensus time**: How quickly the system reaches uniform opinion
- **Opinion changes**: Frequency of agents changing opinions throughout the simulation
- **Group-level polarization**: Whether groups converge to different opinions
- **Alignment impact**: How strong group-opinion correlation affects agent choices

