# Lunar Lander Reinforcement Learning Project

A comprehensive reinforcement learning project comparing different agent types for autonomous lunar landing using OpenAI Gymnasium's LunarLander-v2 environment. This project demonstrates the superior performance of Deep Q-Networks (DQN) over Linear Q-Learning and Random agents across various environmental conditions.

## 🚀 Project Overview

This project implements and compares three different reinforcement learning approaches:
- **Random Agent**: Baseline random action selection
- **Linear Q-Learning Agent**: Traditional linear Q-learning approach
- **Deep Q-Network (DQN) Agent**: Neural network-based Q-learning with experience replay

The agents are tested under various environmental conditions including wind and turbulence to evaluate robustness and performance.

## 📊 Key Results

- **DQN Agent**: Achieves up to **90.3% success rate** in optimal conditions
- **Linear Q-Learning**: Moderate performance with simpler implementation
- **Random Agent**: Baseline performance for comparison

Performance varies significantly under different wind and turbulence conditions, with DQN showing superior adaptability.

## 🔧 Features

- **Multiple Agent Types**: Random, Linear Q-Learning, and DQN implementations
- **Environmental Variations**: Testing with/without wind and turbulence
- **Comprehensive Analysis**: Statistical comparison using t-tests
- **Visualization**: Training curves, performance plots, and success rate pie charts
- **Data Export**: CSV files with episode scores for further analysis
- **Model Persistence**: Save and load trained models

## 📋 Requirements

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

### Dependencies
- `gym==0.26.2` - OpenAI Gymnasium environment
- `matplotlib==3.8.2` - Plotting and visualization
- `numpy==1.26.3` - Numerical computations
- `pandas==2.2.0` - Data manipulation and analysis
- `scipy==1.12.0` - Statistical tests
- `torch==2.1.2` - PyTorch for neural networks
- `tqdm==4.65.0` - Progress bars

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/lunarlander.git
cd lunarlander
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Training and Testing
```bash
cd src
python main.py
```

### 4. Run Statistical Analysis
```bash
cd src
python stats.py
```

## 📁 Project Structure

```
lunarlander/
├── src/
│   ├── main.py              # Main training and testing script
│   ├── stats.py             # Statistical analysis and comparison
│   ├── dqnAgent.py          # Deep Q-Network agent implementation
│   ├── linearAgent.py       # Linear Q-Learning agent implementation
│   └── randomAgent.py       # Random agent implementation
├── data/                    # Generated data and results
│   ├── last/               # Latest run results
│   └── model_*/            # Timestamped experiment results
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🎮 Usage

### Training Agents

The main script (`main.py`) allows you to train and test different agents:

```python
# Configure environment conditions
wind = True                    # Enable/disable wind
wind_powerInput = 15.0        # Wind strength
turbulence_powerInput = 1.5   # Turbulence level

# Enable/disable agent types
randomAgent = True
dqnMethod = True
linearQLearning = True
```

### Statistical Analysis

Run the statistical comparison:

```python
python stats.py
```

This performs t-tests comparing agent performance under different conditions.

## 🧠 Agent Implementations

### DQN Agent
- **Architecture**: Configurable neural network layers
- **Features**: Experience replay, epsilon-greedy exploration, target network
- **Performance**: Best overall performance with 90.3% success rate

### Linear Q-Learning Agent
- **Architecture**: Linear function approximation
- **Features**: Traditional Q-learning with linear state representation
- **Performance**: Moderate success rate, faster training

### Random Agent
- **Architecture**: Random action selection
- **Features**: Baseline comparison agent
- **Performance**: Low success rate, used for statistical significance testing

## 📈 Results and Analysis

### Performance Metrics
- **Success Rate**: Percentage of successful landings (score ≥ 200)
- **Average Score**: Mean episode reward
- **Statistical Significance**: T-test comparisons between agents

### Environmental Conditions Tested
1. **No Wind, No Turbulence**: Optimal conditions
2. **Wind (15.0), No Turbulence**: Wind-only challenge
3. **Wind (15.0), Turbulence (1.5)**: Most challenging conditions

### Key Findings
- DQN significantly outperforms other agents across all conditions
- Environmental complexity affects all agents but DQN shows best adaptability
- Statistical tests confirm significant performance differences

## 🔬 Technical Details

### Training Configuration
- **Episodes**: 2000 training episodes
- **Testing**: 1000 test episodes
- **Epsilon Decay**: 0.995 with minimum 0.0
- **Episode Limit**: 1000 steps maximum

### Model Saving
- Models achieving score ≥ 200 are automatically saved
- Saved models include timestamp and configuration details
- Models can be loaded for continued training or testing

### Data Output
- **Training Plots**: Episode scores over time
- **Test Results**: Success/failure pie charts
- **CSV Files**: Raw score data for each experiment
- **Statistical Analysis**: T-test results and significance

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). This is part of a Reinforcement Learning course final project.

For more details, see the [LICENSE](LICENSE) file or visit [https://www.gnu.org/licenses/agpl-3.0.html](https://www.gnu.org/licenses/agpl-3.0.html).

## 👥 Authors

- **Efe Görkem Şirin** - S4808746
- **Nihat Aksu** - S4709039

*Date: 30/01/2024*

## 🙏 Acknowledgments

- OpenAI Gymnasium for the LunarLander-v2 environment
- PyTorch team for the deep learning framework
- Course instructors and teaching assistants
