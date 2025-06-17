# This file is Python code for running our statistics analysis.
# Not to perform the experiment itself.

import warnings             # To suppress warnings
import time                 # To measure the time taken by the program
import pandas as pd         # For data manipulation
from scipy import stats     # For statistical tests

def compare_agents_performance(csv_random_agent, csv_dqn_agent):
    df_random = pd.read_csv(csv_random_agent)
    df_dqn = pd.read_csv(csv_dqn_agent)
    
    rewards_random = df_random['Score']
    rewards_dqn = df_dqn['Score']
    
    t_stat, p_value = stats.ttest_ind(rewards_random, rewards_dqn, equal_var=False)  
    
    formatted_output = f"T-statistic: {t_stat:.5f}, P-value: {p_value:.20f}"
    print(formatted_output)
    if p_value < 0.05:
        print("The difference in average rewards between the random and DQN agents is statistically significant.")
    else:
        print("No significant difference in average rewards was found between the random and DQN agents.")
    
    return t_stat, p_value


def main():

    path_to_random_agent_no_wind = "../data/model_Random_no_wind_no_turbulence_20240204_184940/scores_model_Random_no_wind_no_turbulence.csv"
    path_to_dqn_agent_no_wind_3L = "../data/model_DQN_3_layers_no_wind_no_turbulence_20240204_184940/scores_model_DQN_3_layers_no_wind_no_turbulence.csv"

    path_to_random_agent_wind_turbulence = "../data/model_Random_wind_15.0_turbulence_1.5_20240204_185133/scores_model_Random_wind_15.0_turbulence_1.5.csv"
    path_to_dqn_agent_wind_turbulence_4L = "../data/model_DQN_4_layers_wind_15.0_turbulence_1.5_20240204_185147/scores_model_DQN_4_layers_wind_15.0_turbulence_1.5.csv"

    path_to_random_agen_wind_nTurbulence = "../data/model_Random_wind_15.0_no_turbulence_20240204_185023/scores_model_Random_wind_15.0_no_turbulence.csv"
    path_to_dqn_agent_wind_nTurbulence_4L = "../data/model_DQN_4_layers_wind_15.0_no_turbulence_20240204_185023/scores_model_DQN_4_layers_wind_15.0_no_turbulence.csv"

    print("\nComparting No wind and no turbulence - 3 Layers DQN and Random Agent")
    compare_agents_performance(path_to_random_agent_no_wind, path_to_dqn_agent_no_wind_3L)

    print("\nComparing the performance of the agents with wind and turbulence 4 Layers DQN and Random Agent")
    compare_agents_performance(path_to_random_agent_wind_turbulence, path_to_dqn_agent_wind_turbulence_4L)

    print("\nComparing the performance of the agents with wind and no turbulence 4 Layers DQN and Random Agent")
    compare_agents_performance(path_to_random_agen_wind_nTurbulence, path_to_dqn_agent_wind_nTurbulence_4L)
    
    return

# Main function Caller
if __name__ == '__main__':
    # Suppress warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Take Staring time 
    start = time.time()

    # Call main function
    main()

    # Take Ending time
    end = time.time()

    # Print the time difference
    print(f"Runtime of the program is {end - start}")
