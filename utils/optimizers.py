import streamlit as st
import numpy as np
from typing import Callable, Dict, Any, List, Tuple
import random
import uuid # For unique IDs if needed
from .knowledge_graph import KnowledgeGraph
# Define a base class for optimizers for consistency
class BaseOptimizer:
    def __init__(self, problem: 'OptimizationProblem', max_iter: int = 50, **kwargs):
        self.problem = problem
        self.max_iter = max_iter
        self.lb = np.array([b[0] for b in self.problem.bounds])
        self.ub = np.array([b[1] for b in self.problem.bounds])
        self.dim = self.problem.dimensions
        self.history = [] # To store optimization progress

    def _initialize_population(self, pop_size: int) -> np.ndarray:
        return np.random.uniform(self.lb, self.ub, (pop_size, self.dim))

    def _evaluate_population(self, population: np.ndarray, agent_outputs: Dict[str, Any], kg_state: 'KnowledgeGraph') -> np.ndarray:
        return np.array([self.problem.objective_function(p, agent_outputs, kg_state) for p in population])

    def _clip_position(self, position: np.ndarray) -> np.ndarray:
        return np.clip(position, self.lb, self.ub)

    def optimize(self, agent_outputs: Dict[str, Any], kg_state: 'KnowledgeGraph') -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement this method.")

# --- Optimization Problem Definition ---
def _default_objective_function(params: List[float], agent_outputs: Dict[str, Any], kg_state: 'KnowledgeGraph') -> float:
    """
    Placeholder objective function that can be overridden.
    In the main experiment, a more sophisticated function will be used.
    It minimizes a composite cost based on parameters and agent outputs.
    """
    base_cost = np.sum(np.square(params - 0.5)) # Penalize deviation from center
    
    code_complexity_score = agent_outputs.get('code_metrics', {}).get('cyclomatic_complexity', 50.0)
    maintainability_index = agent_outputs.get('code_metrics', {}).get('maintainability_index', 50.0)
    conflict_count = len(agent_outputs.get('conflicts', []))
    test_pass_rate = agent_outputs.get('test_results', {}).get('test_pass_rate', 0.0) # Higher is better, so penalize (100-rate)

    cost = base_cost
    cost += code_complexity_score * 0.1 # Penalize high complexity
    cost += (100 - maintainability_index) * 0.5 # Penalize low maintainability
    cost += conflict_count * 5 # Penalize conflicts
    cost += (100 - test_pass_rate) * 0.3 # Penalize low test pass rate

    return max(0.1, cost + random.uniform(0, 0.1)) # Ensure positive and add noise


class OptimizationProblem:
    """
    Defines the optimization problem for metaheuristic algorithms.
    The objective function now takes agent outputs and KG state.
    """
    def __init__(self,
                 objective_function: Callable[[List[float], Dict[str, Any], 'KnowledgeGraph'], float],
                 bounds: List[Tuple[float, float]],
                 dimensions: int):
        self.objective_function = objective_function
        self.bounds = bounds # List of (min, max) for each dimension
        self.dimensions = dimensions

# --- Optimizer Implementations (Robust Versions) ---

class SparrowSearchAlgorithm(BaseOptimizer):
    """
    Sparrow Search Algorithm (SSA) - Refined
    Reference: Xue, J., & Shen, B. (2020). A novel swarm intelligence optimization approach: Sparrow search algorithm.
    """
    def __init__(self, problem: OptimizationProblem, num_sparrows: int = 20, max_iter: int = 50,
                 predator_ratio: float = 0.2, safety_value: float = 0.8, **kwargs):
        super().__init__(problem, max_iter)
        self.num_sparrows = num_sparrows
        self.predator_ratio = predator_ratio
        self.safety_value = safety_value # R2 value, ST in paper

    def optimize(self, agent_outputs: Dict[str, Any], kg_state: 'KnowledgeGraph') -> Dict[str, Any]:
        pos = self._initialize_population(self.num_sparrows)
        fitness = self._evaluate_population(pos, agent_outputs, kg_state)

        best_pos = pos[np.argmin(fitness)].copy()
        best_fitness = np.min(fitness)
        self.history = []

        for t in range(self.max_iter):
            sorted_indices = np.argsort(fitness)
            pos = pos[sorted_indices]
            fitness = fitness[sorted_indices]

            best_pos = pos[0].copy()
            best_fitness = fitness[0]
            self.history.append({"iteration": t, "best_fitness": best_fitness.item(), "best_position": best_pos.tolist()})

            num_discoverers = int(self.num_sparrows * self.predator_ratio)
            
            # Update discoverers
            for i in range(num_discoverers):
                if random.random() < self.safety_value: # R2 < ST
                    pos[i] = pos[i] * np.exp(-i / (self.max_iter * np.random.rand()))
                else:
                    pos[i] = pos[i] + np.random.normal(0, 1, self.dim) # Random walk

            # Update scavengers
            for i in range(num_discoverers, self.num_sparrows):
                if i > self.num_sparrows / 2: # Sparrow in a poor position
                    pos[i] = best_pos + np.random.normal(0, 1, self.dim) * np.exp((pos[-1] - pos[i]) / i**2)
                else:
                    pos[i] = best_pos + np.abs(pos[i] - best_pos) * np.random.normal(0, 1, self.dim) # Move towards best_pos

            # Handle sparrows aware of danger (randomly selected S_D sparrows)
            num_aware_of_danger = int(self.num_sparrows * 0.2) # Typically 10-20%
            rand_aware_indices = np.random.choice(self.num_sparrows, num_aware_of_danger, replace=False)
            for i in rand_aware_indices:
                if fitness[i] > best_fitness: # Sparrow is in a dangerous (high fitness) position
                    pos[i] = best_pos + np.random.normal(0, 1, self.dim) * np.abs(pos[i] - best_pos)
                else: # Sparrow is near a safe spot
                    pos[i] = pos[i] + (2 * np.random.rand(self.dim) - 1) * np.abs(pos[i] - pos[-1]) / (fitness[i] - fitness[-1] + 1e-8) # Move randomly near best/worst

            pos = self._clip_position(pos)
            fitness = self._evaluate_population(pos, agent_outputs, kg_state)
            
        final_best_pos_idx = np.argmin(fitness)
        final_best_pos = pos[final_best_pos_idx].copy()
        final_best_fitness = fitness[final_best_pos_idx]

        self.history.append({"iteration": self.max_iter, "best_fitness": final_best_fitness.item(), "best_position": final_best_pos.tolist()})

        return {"best_position": final_best_pos.tolist(), "best_fitness": final_best_fitness.item(), "history": self.history}

class GeneticAlgorithm(BaseOptimizer):
    """
    Genetic Algorithm (GA) - Robust implementation
    """
    def __init__(self, problem: OptimizationProblem, population_size: int = 50, max_iter: int = 50,
                 mutation_rate: float = 0.05, crossover_rate: float = 0.8, **kwargs):
        super().__init__(problem, max_iter)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def optimize(self, agent_outputs: Dict[str, Any], kg_state: 'KnowledgeGraph') -> Dict[str, Any]:
        population = self._initialize_population(self.population_size)
        fitness = self._evaluate_population(population, agent_outputs, kg_state)

        best_pos = population[np.argmin(fitness)].copy()
        best_fitness = np.min(fitness)
        self.history = []

        for t in range(self.max_iter):
            # Selection (Roulette Wheel Selection)
            # Minimize objective, so invert fitness for selection probability
            inv_fitness = 1 / (fitness + 1e-6) # Add small epsilon to avoid division by zero
            prob = inv_fitness / np.sum(inv_fitness)
            selected_indices = np.random.choice(self.population_size, self.population_size, p=prob, replace=True)
            selected_population = population[selected_indices]

            new_population = []
            for i in range(0, self.population_size, 2):
                parent1 = selected_population[i]
                parent2 = selected_population[i+1] if i + 1 < self.population_size else selected_population[i] # Handle odd pop size

                if random.random() < self.crossover_rate:
                    # Crossover (Arithmetic Crossover)
                    alpha = random.random()
                    child1 = alpha * parent1 + (1 - alpha) * parent2
                    child2 = alpha * parent2 + (1 - alpha) * parent1
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                new_population.extend([child1, child2])

            new_population = np.array(new_population[:self.population_size])

            # Mutation (Gaussian Mutation)
            mutation_mask = np.random.rand(self.population_size, self.dim) < self.mutation_rate
            new_population[mutation_mask] += np.random.normal(0, 0.1, new_population.shape)[mutation_mask] # Add small Gaussian noise

            population = self._clip_position(new_population)
            fitness = self._evaluate_population(population, agent_outputs, kg_state)
            
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_fitness:
                best_fitness = fitness[current_best_idx]
                best_pos = population[current_best_idx].copy()
            self.history.append({"iteration": t, "best_fitness": best_fitness.item(), "best_position": best_pos.tolist()})

        return {"best_position": best_pos.tolist(), "best_fitness": best_fitness.item(), "history": self.history}

class ParticleSwarmOptimization(BaseOptimizer):
    """
    Particle Swarm Optimization (PSO) - Robust implementation
    """
    def __init__(self, problem: OptimizationProblem, num_particles: int = 30, max_iter: int = 50,
                 w: float = 0.5, c1: float = 1.5, c2: float = 1.5, **kwargs):
        super().__init__(problem, max_iter)
        self.num_particles = num_particles
        self.w = w # inertia weight
        self.c1 = c1 # cognitive coefficient
        self.c2 = c2 # social coefficient

    def optimize(self, agent_outputs: Dict[str, Any], kg_state: 'KnowledgeGraph') -> Dict[str, Any]:
        positions = self._initialize_population(self.num_particles)
        velocities = np.random.uniform(-0.1 * (self.ub - self.lb), 0.1 * (self.ub - self.lb), (self.num_particles, self.dim)) # Initial velocities

        personal_best_positions = positions.copy()
        personal_best_fitness = self._evaluate_population(positions, agent_outputs, kg_state)

        global_best_idx = np.argmin(personal_best_fitness)
        global_best_position = personal_best_positions[global_best_idx].copy()
        global_best_fitness = personal_best_fitness[global_best_idx]
        self.history = []

        for t in range(self.max_iter):
            current_fitness = self._evaluate_population(positions, agent_outputs, kg_state)

            # Update personal best
            update_mask = current_fitness < personal_best_fitness
            personal_best_fitness[update_mask] = current_fitness[update_mask]
            personal_best_positions[update_mask] = positions[update_mask]

            # Update global best
            new_global_best_idx = np.argmin(personal_best_fitness)
            if personal_best_fitness[new_global_best_idx] < global_best_fitness:
                global_best_fitness = personal_best_fitness[new_global_best_idx]
                global_best_position = personal_best_positions[new_global_best_idx].copy()
            
            self.history.append({"iteration": t, "best_fitness": global_best_fitness.item(), "best_position": global_best_position.tolist()})

            r1 = np.random.rand(self.num_particles, self.dim)
            r2 = np.random.rand(self.num_particles, self.dim)

            # Update velocities
            velocities = (self.w * velocities +
                          self.c1 * r1 * (personal_best_positions - positions) +
                          self.c2 * r2 * (global_best_position - positions))

            # Update positions
            positions = positions + velocities

            positions = self._clip_position(positions)

        return {"best_position": global_best_position.tolist(), "best_fitness": global_best_fitness.item(), "history": self.history}

class GreyWolfOptimizer(BaseOptimizer):
    """
    Grey Wolf Optimizer (GWO) - Robust implementation
    Reference: Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). Grey Wolf Optimizer.
    """
    def __init__(self, problem: OptimizationProblem, num_wolves: int = 30, max_iter: int = 50, **kwargs):
        super().__init__(problem, max_iter)
        self.num_wolves = num_wolves

    def optimize(self, agent_outputs: Dict[str, Any], kg_state: 'KnowledgeGraph') -> Dict[str, Any]:
        positions = self._initialize_population(self.num_wolves)
        fitness = self._evaluate_population(positions, agent_outputs, kg_state)

        # Alpha, Beta, Delta (the three best wolves)
        sorted_indices = np.argsort(fitness)
        alpha_pos = positions[sorted_indices[0]].copy()
        beta_pos = positions[sorted_indices[1]].copy()
        delta_pos = positions[sorted_indices[2]].copy()

        global_best_fitness = fitness[sorted_indices[0]]
        self.history = []

        for t in range(self.max_iter):
            a = 2 - t * (2 / self.max_iter) # 'a' decreases linearly from 2 to 0

            for i in range(self.num_wolves):
                # Calculate D_alpha, D_beta, D_delta
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = np.abs(C1 * alpha_pos - positions[i])
                X1 = alpha_pos - A1 * D_alpha

                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = np.abs(C2 * beta_pos - positions[i])
                X2 = beta_pos - A2 * D_beta

                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = np.abs(C3 * delta_pos - positions[i])
                X3 = delta_pos - A3 * D_delta

                # Update current wolf's position
                positions[i] = (X1 + X2 + X3) / 3

            positions = self._clip_position(positions)
            fitness = self._evaluate_population(positions, agent_outputs, kg_state)

            sorted_indices = np.argsort(fitness)
            alpha_pos = positions[sorted_indices[0]].copy()
            beta_pos = positions[sorted_indices[1]].copy()
            delta_pos = positions[sorted_indices[2]].copy()
            global_best_fitness = fitness[sorted_indices[0]]

            self.history.append({"iteration": t, "best_fitness": global_best_fitness.item(), "best_position": alpha_pos.tolist()})

        return {"best_position": alpha_pos.tolist(), "best_fitness": global_best_fitness.item(), "history": self.history}

class WhaleOptimizationAlgorithm(BaseOptimizer):
    """
    Whale Optimization Algorithm (WOA) - Robust implementation
    Reference: Mirjalili, S., & Lewis, A. (2016). The Whale Optimization Algorithm.
    """
    def __init__(self, problem: OptimizationProblem, num_whales: int = 30, max_iter: int = 50, **kwargs):
        super().__init__(problem, max_iter)
        self.num_whales = num_whales

    def optimize(self, agent_outputs: Dict[str, Any], kg_state: 'KnowledgeGraph') -> Dict[str, Any]:
        positions = self._initialize_population(self.num_whales)
        fitness = self._evaluate_population(positions, agent_outputs, kg_state)

        global_best_idx = np.argmin(fitness)
        global_best_position = positions[global_best_idx].copy()
        global_best_fitness = fitness[global_best_idx]
        self.history = []

        for t in range(self.max_iter):
            a = 2 - t * (2 / self.max_iter) # 'a' decreases linearly from 2 to 0
            a2 = -1 + t * (-1 / self.max_iter) # 'a2' decreases linearly from -1 to -2

            for i in range(self.num_whales):
                r1, r2 = random.random(), random.random()
                A = 2 * a * r1 - a # Coefficient vector A
                C = 2 * r2         # Coefficient vector C
                b = 1              # Parameter for spiral shape
                l = (a2 - 1) * random.random() + 1 # Random number for spiral

                p = random.random() # Probability of switching between shrinking and spiral

                if p < 0.5: # Shrinking encircling mechanism
                    if np.abs(A) < 1:
                        D = np.abs(C * global_best_position - positions[i])
                        positions[i] = global_best_position - A * D
                    else: # Search for prey
                        rand_leader_idx = np.random.randint(0, self.num_whales)
                        rand_leader_pos = positions[rand_leader_idx].copy()
                        D = np.abs(C * rand_leader_pos - positions[i])
                        positions[i] = rand_leader_pos - A * D
                else: # Spiral-shaped attacking mechanism
                    D_prime = np.abs(global_best_position - positions[i])
                    positions[i] = D_prime * np.exp(b * l) * np.cos(2 * np.pi * l) + global_best_position

            positions = self._clip_position(positions)
            fitness = self._evaluate_population(positions, agent_outputs, kg_state)

            current_global_best_idx = np.argmin(fitness)
            if fitness[current_global_best_idx] < global_best_fitness:
                global_best_fitness = fitness[current_global_best_idx]
                global_best_position = positions[current_global_best_idx].copy()
            
            self.history.append({"iteration": t, "best_fitness": global_best_fitness.item(), "best_position": global_best_position.tolist()})

        return {"best_position": global_best_position.tolist(), "best_fitness": global_best_fitness.item(), "history": self.history}

class FireflyAlgorithm(BaseOptimizer):
    """
    Firefly Algorithm (FA) - Robust implementation
    Reference: Yang, X. S. (2010). Firefly algorithm, stochastic test functions and design optimisation.
    """
    def __init__(self, problem: OptimizationProblem, num_fireflies: int = 30, max_iter: int = 50,
                 alpha: float = 0.5, beta0: float = 1.0, gamma: float = 1.0, **kwargs):
        super().__init__(problem, max_iter)
        self.num_fireflies = num_fireflies
        self.alpha = alpha # Randomization parameter
        self.beta0 = beta0 # Attractiveness at r=0
        self.gamma = gamma # Light absorption coefficient

    def optimize(self, agent_outputs: Dict[str, Any], kg_state: 'KnowledgeGraph') -> Dict[str, Any]:
        positions = self._initialize_population(self.num_fireflies)
        fitness = self._evaluate_population(positions, agent_outputs, kg_state)

        global_best_idx = np.argmin(fitness)
        global_best_position = positions[global_best_idx].copy()
        global_best_fitness = fitness[global_best_idx]
        self.history = []

        for t in range(self.max_iter):
            for i in range(self.num_fireflies):
                for j in range(self.num_fireflies):
                    if fitness[i] > fitness[j]: # Firefly i is less bright (has higher fitness/cost) than firefly j
                        r = np.linalg.norm(positions[i] - positions[j]) # Distance between fireflies
                        beta = self.beta0 * np.exp(-self.gamma * r**2) # Attractiveness
                        
                        # Movement equation
                        positions[i] += beta * (positions[j] - positions[i]) + self.alpha * (np.random.rand(self.dim) - 0.5)
            
            positions = self._clip_position(positions)
            fitness = self._evaluate_population(positions, agent_outputs, kg_state)
            
            current_global_best_idx = np.argmin(fitness)
            if fitness[current_global_best_idx] < global_best_fitness:
                global_best_fitness = fitness[current_global_best_idx]
                global_best_position = positions[current_global_best_idx].copy()
            
            self.history.append({"iteration": t, "best_fitness": global_best_fitness.item(), "best_position": global_best_position.tolist()})

        return {"best_position": global_best_position.tolist(), "best_fitness": global_best_fitness.item(), "history": self.history}


def get_available_optimizers():
    """
    Returns a list of implemented metaheuristic optimizers.
    """
    return [
        "Sparrow Search Algorithm (SSA)",
        "Genetic Algorithm (GA)",
        "Particle Swarm Optimization (PSO)",
        "Grey Wolf Optimizer (GWO)",
        "Whale Optimization Algorithm (WOA)",
        "Firefly Algorithm (FA)"
    ]

def run_optimizer(optimizer_name: str, problem: OptimizationProblem, agent_outputs: Dict[str, Any], kg_state: 'KnowledgeGraph', num_iterations: int) -> Dict[str, Any]:
    """
    Runs the selected optimizer on the given problem, passing agent outputs and KG state.
    """
    optimizer_instance: BaseOptimizer = None
    common_params = {"problem": problem, "max_iter": num_iterations}

    if optimizer_name == "Sparrow Search Algorithm (SSA)":
        optimizer_instance = SparrowSearchAlgorithm(**common_params)
    elif optimizer_name == "Genetic Algorithm (GA)":
        optimizer_instance = GeneticAlgorithm(**common_params)
    elif optimizer_name == "Particle Swarm Optimization (PSO)":
        optimizer_instance = ParticleSwarmOptimization(**common_params)
    elif optimizer_name == "Grey Wolf Optimizer (GWO)":
        optimizer_instance = GreyWolfOptimizer(**common_params)
    elif optimizer_name == "Whale Optimization Algorithm (WOA)":
        optimizer_instance = WhaleOptimizationAlgorithm(**common_params)
    elif optimizer_name == "Firefly Algorithm (FA)":
        optimizer_instance = FireflyAlgorithm(**common_params)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    st.write(f"Running {optimizer_name} for {num_iterations} internal iterations...")
    result = optimizer_instance.optimize(agent_outputs, kg_state)
    st.write(f"Optimizer {optimizer_name} finished.")
    return result