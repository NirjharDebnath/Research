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
