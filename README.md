# REINFORCE Algorithm for CartPole

This project implements the REINFORCE algorithm to solve the CartPole-v1 environment. The algorithm uses policy gradients to optimize the agent’s performance by reinforcing actions that lead to higher returns.

---

## Overview

### Environment Details:
- **Name**: CartPole-v1
- **State Space**: 4-dimensional vector comprising:
  - Cart position.
  - Cart velocity.
  - Pole angle.
  - Pole angular velocity.
- **Action Space**: 2 discrete actions:
  1. Move left.
  2. Move right.
- **Reward**: The agent receives a reward of 1 for every timestep the pole remains upright. Maximum possible reward per episode is 500.
- **Termination**:
  - The episode ends if the pole falls over or 500 timesteps are reached.
- **Goal**: Achieve an average return of 500 across multiple episodes.

---

## Methodology

### REINFORCE Algorithm:
The REINFORCE algorithm optimizes the policy directly by using policy gradients. It adjusts the policy parameters to maximize the expected cumulative reward.

#### Key Steps:
1. **Policy Gradient Objective**:
   - Optimize the objective function:
     \[ J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [\Psi(\tau)] \]
     where \( \Psi(\tau) \) is the cumulative reward.

2. **Gradient Computation**:
   - Compute the gradient of the log-probability of actions, weighted by the returns:
     \[ \nabla_\theta J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \Psi_t \nabla_\theta \log \pi_\theta(a_t|s_t) \right] \]

3. **Neural Network Policy**:
   - The policy is represented by a neural network:
     - **Input**: Observation vector.
     - **Hidden Layers**: Two fully connected layers with 20 units each.
     - **Output**: Action probabilities.

4. **Return Calculation**:
   - Compute discounted returns for each timestep using:
     \[ G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k} \]

---

## Implementation Details

### Neural Network:
- Built using Haiku, a neural network library for JAX.
- Outputs logits, which are converted into action probabilities using the softmax function.

### Optimizer:
- Adam optimizer with a learning rate of 1e-3.

### Exploration:
- Actions are sampled from a categorical distribution defined by the policy’s action probabilities.

---

## Training Process

- **Episodes**: 2,500.
- **Learning Steps per Episode**: 2.
- **Discount Factor (\(\gamma\))**: 0.99.

**Training Performance Placeholder**:
![Training Performance](visuals/training_performance_reinforce.png)

---

## Results

### Performance Metrics:
- Initial Performance: Episode returns close to 0.
- Final Performance: Episode returns reliably reaching 500.


---

## Visualizations

### Policy Behavior:
- Visualize the policy’s performance at specific episodes.

**Video of the Policy**:
[Evaluation Video - Episode 1700](visuals/cartpoleReinforce_policy.mp4)

---

## Instructions

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Training**:
   ```bash
   python cartpoleReinforce.py
   ```

3. **Visualize Results**:
   - Plots and videos will be saved in the `visuals/` directory.

---

## Future Directions

1. Extend the algorithm to more complex environments like LunarLander.
2. Experiment with advanced policy gradient algorithms, such as PPO or A2C.
3. Fine-tune hyperparameters for improved convergence speed and stability.

---

## References
- OpenAI Gym Documentation.
- Haiku and Optax Libraries for Neural Network and Optimization.

---

**End of Report**

