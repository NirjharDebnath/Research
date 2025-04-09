<center><h1> Reinforcement Learning Based Hyperparameter Tuning of Particle Swarm Optimisation Algorithm </h1></center>
<hr>

## 1.0 Abstract
This study aims to improve the performance of the particle swarm optimization agorithm by using Deep Reinforcement Learning with Transfer Learning

<center> <h4>Keywords</h4>Particle Swarm Optimisation, Reinforcement Learning, Transfer Learning </center>

## 2.0 Introduction

### 2.1 Particle Swarm Optimization
Particle Swarm Optimization (PSO) is a widely recognized bio-inspired optimization algorithm introduced by Kennedy and Eberhart in 1995. Inspired by the collective behavior of social organisms such as flocks of birds or schools of fish, PSO simulates the movement and interaction of particles in a multidimensional search space to identify optimal solutions. Unlike gradient-based optimization methods, PSO relies solely on the evaluation of an objective function, making it versatile and applicable to a wide range of optimization problems, including those with non-linear or complex solution spaces. Its simplicity, minimal reliance on hyperparameters, and ability to converge efficiently have established PSO as a robust tool for solving engineering, scientific, and computational challenges

#### **Mathematical Description of Particle Swarm Optimization**

Particle Swarm Optimization (PSO) is a population-based stochastic optimization algorithm inspired by the social behavior of organisms such as bird flocks or fish schools. It operates by iteratively improving candidate solutions in a search space using simple mathematical formulas for position and velocity updates.

#### **Key Components**
- **Particles**: Each particle represents a potential solution in the search space.
- **Swarm**: A collection of particles working collaboratively to find the optimal solution.
- **Objective Function**: A function $$f(x)$$ that evaluates the quality of each particle's position.
- **Best Positions**:
  - $$p_{\text{best}}$$: The best position found by an individual particle.
  - $$g_{\text{best}}$$: The best position found by the entire swarm.

#### **Mathematical Model**
1. **Position and Velocity Updates**:
   Each particle's movement is governed by its velocity and current position. The velocity $$v_i$$ and position $$x_i$$ of particle $$i$$ at iteration $$t+1$$ are updated using the following equations:

   $v_i^{t+1} = w v_i^t + c_1 r_1 (p_{\text{best},i} - x_i^t) + c_2 r_2 (g_{\text{best}} - x_i^t)$

   $x_i^{t+1} = x_i^t + v_i^{t+1}$

   Where:
   - $$w$$: Inertia weight controlling exploration vs exploitation.
   - $$c_1, c_2$$: Cognitive and social coefficients, respectively.
   - $$r_1, r_2$$: Uniformly distributed random numbers.
   - $$p_{\text{best},i}$$: Personal best position of particle $$i$$.
   - $$g_{\text{best}}$$: Global best position among all particles.

2. **Fitness Evaluation**:
   The fitness of each particle is computed using the objective function $$f(x)$$. Particles update their personal best ($$p_{\text{best}}$$) and global best ($$g_{\text{best}}$$) based on fitness values.

3. **Termination Criteria**:
   The algorithm iterates until a predefined stopping condition is met, such as:
   - Maximum number of iterations.
   - Achieving a satisfactory objective function value.

#### **Algorithm Steps**
1. Initialize positions $$x_i^0$$ and velocities $$v_i^0$$ randomly within bounds.
2. Evaluate the fitness of each particle using $$f(x)$$.
3. Update $$p_{\text{best},i}$$ and $$g_{\text{best}}$$.
4. Update velocities and positions using the equations above.
5. Repeat steps 2–4 until termination criteria are satisfied.

#### **Advantages**
- Does not require gradient information, making it suitable for non-differentiable or complex functions.
- Simple implementation with few parameters to tune.

### 2.2 Reinforcement Learning
Reinforcement learning (RL) is an interdisciplinary area of machine learning and optimal control concerned with how an intelligent agent should take actions in a dynamic environment in order to maximize a reward signal. Reinforcement learning is one of the three basic machine learning paradigms, alongside supervised learning and unsupervised learning. Reinforcement learning differs from supervised learning in not needing labelled input-output pairs to be presented, and in not needing sub-optimal actions to be explicitly corrected. Instead, the focus is on finding a balance between exploration (of uncharted territory) and exploitation (of current knowledge) with the goal of maximizing the cumulative reward (the feedback of which might be incomplete or delayed). The search for this balance is known as the exploration–exploitation dilemma.

#### **Policy Gradient Algorithm**
Policy gradient algorithm is a policy iteration approach where policy is directly manipulated to reach the optimal policy that maximises the expected return. The environment dynamics or transition probability is indicated as below:

$$ p(s_{t+1} | s, a) $$

It can be read the probability of reaching the next state $s_{t+1}$ by taking the action from the current state s. Policy 𝜋 is a distribution over actions given states. In other words, the policy defines the behaviour of the agent.

$$ \pi(a | s) $$

**Return and reward** - 
We can define our return as the sum of rewards from the current state to the goal state i.e. the sum of rewards in a trajectory(we are just considering finite undiscounted horizon).

$$ R(\tau) = \sum_{t=0}^{T-1} R(s_t, a_t) $$

Where τ = $(s_0​,a_0​,…,s_{T−1}​,a_{T−1}​)$.

**Objective function** - 
In policy gradient, the policy is usually modelled with a parameterized function respect to θ, $π_θ(a|s)$. From a mathematical perspective, an objective function is to minimise or maximise something. We consider a stochastic, parameterized policy $π_θ$ and aim to maximise the expected return using objective function $J(π_θ)$

$$ J(\pi_{\theta}) = E_{\pi_{\theta}}\left[\sum_{t=0}^{T-1}R(s_t, a_t)\right] = E_{\pi_{\theta}}[R(\tau)] $$

Here $R(s_t, a_t)$ is defined as reward obtained at timestep t by performing an action $a_t$ from the state $s_t$. We know the fact that $R(s_t, a_t)$ can be represented as $R(τ)$.

We can maximise the objective function J to maximises the return by adjusting the policy parameter θ to get the best policy. The best policy will always maximise the return. The gradient ascent is the optimisation algorithm that iteratively searches for optimal parameters that maximise the objective function.

If we can find out the gradient ∇ of the objective function J, as shown below:

$$ \nabla J(\pi_{\theta}) = \nabla E_{\pi_{\theta}}[R(\tau)] $$

Then, we can update the policy parameter θ(for simplicity, we are going to use θ instead of $π_θ$), using the gradient ascent rule. This way, we can update the parameters θ in the direction of the gradient(Remember the gradient gives the direction of the maximum change, and the magnitude indicates the maximum rate of change ). The gradient update rule is as shown below:

$$ \theta = \theta + \alpha \nabla E_{\pi_\theta}[R(\tau)] $$

<center>or</center> 

$$ \theta = \theta + \alpha \nabla J(\theta) $$
