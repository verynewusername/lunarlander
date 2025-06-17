# Reinforcement Learning - Final Project
# Date: 30/01/2024
# Authors:
#   Efe Görkem Şirin    S4808746
#   Nihat Aksu          S4709039

import os                           # OS
import gym                          # OpenAI Gym
import time                         # Time
import torch                        # PyTorch
from tqdm import tqdm               # Progress Bar
import warnings                     # Warnings
from collections import deque       # Deque
import matplotlib.pyplot as plt     # Matplotlib

from dqnAgent import DQNAgent       # DQN Agent
from linearAgent import LinearAgent # Linear Q Agent
from randomAgent import RandomAgent # Random Agent

# Get current time for output directory - GLOBAL
current_time = time.strftime("%Y%m%d_%H%M%S")

# Graph Title Generator
def get_graph_title(wind, wind_power, turbulence_power, modelName, agent):

    title = ""

    if modelName == "DQN":
        title = "DQN"
    elif modelName == "Linear":
        title = "Linear Q Learning"
    else:
        title = "Random"

    if modelName == "DQN":
        layerCount = agent.q_network.get_active_layer_count()

    if modelName == "DQN":
        title += " - " + str(layerCount) + " Layers"

    title += " with Wind" if wind else ""
    if wind:
        title += " - Wind Power: " + str(wind_power)
    if turbulence_power > 0:
        title += " - Turbulence Power: " + str(turbulence_power)
    
    return title

# Model Name Generator
def get_model_name(wind, wind_power, turbulence_power, modelName, agent):

    name = "model_" + modelName

    if modelName == "DQN":
        name += "_" + str(agent.q_network.get_active_layer_count()) + "_layers"

    if wind:
        name += "_wind_"
        name += str(wind_power) 
    else:
        name += "_no_wind"

    if turbulence_power > 0:
        name += "_turbulence_"
        name += str(turbulence_power)
    else:
        name += "_no_turbulence"

    return name


# TRAINING
def train(agent, env, wind, wind_power, turbulence_power, modelName, episodes=2000):
    print("Training Start - " + get_graph_title(wind, wind_power, turbulence_power, modelName, agent))
    traing_start = time.time()

    # TRAINING HYPERPARAMS #
    epsilon_decay = 0.995       # Epsilon decay rate
    epsilon_min = 0.0           # Minimum epsilon value
    epsilon = 1.0               # Starting epsilon value

    newModelSaved = False

    # Initialize the scores
    scores = deque()

    for episode in tqdm(range(1, episodes + 1), desc="Processing Episodes", unit="episode"):

        state = env.reset() # Reset the environment

        done = False    # To keep track of the end of the episode
        score = 0       # To keep track of the score
        idx = 0         # To keep track of the initial state

        # Hard limit for the episode -> 1000 steps
        for _ in range(1, 1000 + 1):

            # For initial state -> values are different -> fixes the problem
            if idx == 0:
                state = state[0]

            # Get the action from the agent
            action = agent.act(state, epsilon)

            # Take the action in the environment
            next_state, reward, done, info , _= env.step(action)

            # Add the experience to the memory
            agent.step(state, action, reward, next_state, done)

            # Set the next state to the current state
            state = next_state        

            # Update the reward
            score += reward

            # If the episode is finished, break the loop
            if done:
                break
            
            # Update the epsilon value
            epsilon = max(epsilon * epsilon_decay, epsilon_min)

            # If the reward is greater than 200, the environment is solved
            if score >= 200:
                fileName = get_model_name(wind, wind_power, turbulence_power, modelName, agent) + ".pth"
                agent.checkpoint(fileName)
                newModelSaved = True
                break

            idx += 1 # Update the index

        scores.append(score)
    # print("final epsilon value:" + str(epsilon))
    # print if the model is saved
    if newModelSaved:
        print("New Model Saved")
    else:
        print("Model Not Saved")

    env.close()

    # Plot the scores
    plt.plot(scores)
    plt.title(get_graph_title(wind, wind_power, turbulence_power, modelName, agent), fontsize=9)
    plt.xlabel('# of episodes')
    plt.ylabel('score')

    # Get current time for output directory
    global current_time

    # Create output directory
    output_dir = f"../data/" + get_model_name(wind, wind_power, turbulence_power, modelName, agent) + "_" + current_time
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

    # Create last output directory - this will be used to store the last run
    last_output_dir = "../data/last"
    os.makedirs(last_output_dir, exist_ok=True)  # Create directory if it doesn't exist

    # Clean "last" directory
    os.system(f"rm -rf {last_output_dir}/*")

    # Save the plot
    plt.savefig(f"{output_dir}/plot.png")
    plt.savefig(f"{last_output_dir}/plot.png")

    # Show the plot
    plt.show()

    traing_end = time.time()
    print("Training End - " + modelName)
    print(f"Training time: {traing_end - traing_start}")
    return

# TESTING
def test(agent, env, wind, wind_power, turbulence_power, modelName, episodes=1000):
    print("Testing Start - " + get_graph_title(wind, wind_power, turbulence_power, modelName, agent))
    test_start = time.time()

    fileName = get_model_name(wind, wind_power, turbulence_power, modelName, agent) + ".pth"

    if modelName != "Random":
        # Safety for file not found
        try:
            # Load the weights from file
            agent.q_network.load_state_dict(torch.load(fileName))
            print("Model Loaded")
        except FileNotFoundError:
            print("File not found")
            print("Aborting Testing")
            return

        # Set the network in evaluation mode (turn off dropout, batchnorm etc.)
        agent.q_network.eval()

    # Initialize the environment
    state = env.reset()

    scoreList = []
    idx = 0
    passNumber = 0

    for episode in tqdm(range(1, episodes + 1), desc="Processing Episodes", unit="episode"):

        state = env.reset() # Reset the environment
        score = 0
        idx = 0

        # Play the game!
        for _ in range(1000):

            # For initial state -> values are different -> fixes the problem
            if idx == 0:
                state = state[0]

            # Render the game
            env.render()

            # Choose an action greedily from the Q-network
            action = agent.act(state)

            # Take the action (a) in the environment
            next_state, reward, done, info, _  = env.step(action)

            # Update the score
            score += reward

            # Set the next state
            state = next_state

            if score >= 200:
                passNumber += 1
                break

            # If the episode is finished, break the loop
            if done:
                break

            idx += 1
        
        scoreList.append(score)
        
    
    # Close the environment
    env.close()

    # Print Number of passes and non passes
    print(f"Passes: {passNumber}")
    print(f"Non Passes: {episodes - passNumber}")
    
    # Plot Pie chart for passes vs non passes
    labels = ['Successfull Landing', 'Unsuccessfull Landing']
    sizes = [passNumber, episodes - passNumber]

    fig1, ax1 = plt.subplots()
    plt.title(get_graph_title(wind, wind_power, turbulence_power, modelName, agent) + " - Passes vs Non Passes", fontsize=9)
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Get current time for output directory
    global current_time

    # Create output directory
    output_dir = f"../data/" + get_model_name(wind, wind_power, turbulence_power, modelName, agent) + "_" + current_time
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

    # Create last output directory - this will be used to store the last run
    last_output_dir = "../data/last"
    os.makedirs(last_output_dir, exist_ok=True)  # Create directory if it doesn't exist

    # Clean "last" directory
    # os.system(f"rm -rf {last_output_dir}/*")  # No longer needed

    # Save the plot
    plt.savefig(f"{output_dir}/pie" + get_model_name(wind, wind_power, turbulence_power, modelName, agent) + ".png")
    plt.savefig(f"{last_output_dir}/pie" + get_model_name(wind, wind_power, turbulence_power, modelName, agent) + ".png")


    # == Save Scores with coma separated values == #
    # Create the file
    with open(f"{output_dir}/scores_" + get_model_name(wind, wind_power, turbulence_power, modelName, agent) + ".csv", "w") as file:
        # Write the header
        file.write("Score\n")
        # Write the scores
        for score in scoreList:
            file.write(f"{score}\n")

    # Create the file
    with open(f"{last_output_dir}/scores_" + get_model_name(wind, wind_power, turbulence_power, modelName, agent) + ".csv", "w") as file:
        # Write the header
        file.write("Score\n")
        # Write the scores
        for score in scoreList:
            file.write(f"{score}\n")

    # Show the plot
    plt.show()

    test_end = time.time()
    print("Testing End - " + modelName)
    print(f"Testing time: {test_end - test_start}")
    return

# int main(int argc, char **argv)
def main():
    print("Main Start")

    # Set the random seed
    seedInput = 31

    # Set Seeds of everything
    torch.manual_seed(seedInput)

    randomAgent = True
    dqnMethod = True
    linearQLearning = True

    wind_powerInput = float(15.0)
    turbulence_powerInput = float(1.5)
    wind = True
    # wind_powerInput = float(0)
    # turbulence_powerInput = float(0)
    # wind = False


    # Create the environment -> for training 
    envTraining = gym.make(
        "LunarLander-v2",
        continuous = False,
        enable_wind = wind,
        wind_power = float(wind_powerInput),
        turbulence_power = float(turbulence_powerInput),
        # render_mode='human'
    )

    # Create the environment -> for testing
    envTesting = gym.make(
        "LunarLander-v2",
        continuous = False,
        enable_wind = wind,
        wind_power = float(wind_powerInput),
        turbulence_power = float(turbulence_powerInput),
        # render_mode='human'
    )

    # DQN Method
    if dqnMethod:
        print("DQN Method")

        state_size = envTraining.observation_space.shape[0]
        action_size = envTraining.action_space.n

        # Create the agent
        agent = DQNAgent(state_size, action_size, seed=seedInput)
    
        # Train the agent
        train(agent, envTraining, wind, wind_powerInput, turbulence_powerInput, "DQN")

        # Test the agent
        test(agent, envTesting, wind, wind_powerInput, turbulence_powerInput, "DQN")

    # Linear Q Learning Method
    if linearQLearning:
        print("Linear Q Learning Method")

        state_size = envTraining.observation_space.shape[0]
        action_size = envTraining.action_space.n

        # Create the agent
        agent = LinearAgent(state_size, action_size, seed=seedInput)
        # Train the agent
        train(agent, envTraining, wind, wind_powerInput, turbulence_powerInput, "Linear")

        # Test the agent
        test(agent, envTesting, wind, wind_powerInput, turbulence_powerInput, "Linear")

    # Random Agent
    if randomAgent:
        print("Random Agent")

        # Create the agent
        agent = RandomAgent(envTraining)
        # Test the agent
        test(agent, envTesting, wind, wind_powerInput, turbulence_powerInput, "Random")
        
    print("Main End")
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
