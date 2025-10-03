import random
import math

# --- 1. DE Parameters (Tune these for your problem) ---
POPULATION_SIZE = 50
NUM_GENERATIONS = 300
F = 0.8  # Scaling Factor
CR = 0.7 # Crossover Rate

# --- 2. Problem-Specific Setup (MODIFIED FOR ACKLEY FUNCTION) ---

NUM_DIMENSIONS = 2
BOUNDS = [(-5.0, 5.0)] * NUM_DIMENSIONS

def objective_function(vector):
    """
        Calculates the score for the Ackley function.
        We want to MINIMIZE this function.
    """
    x, y = vector
    part1 = -20.0 * math.exp(-0.2 * math.sqrt(0.5 * (x**2 + y**2)))
    part2 = -math.exp(0.5 * (math.cos(2.0 * math.pi * x) + math.cos(2.0 * math.pi * y)))
    return part1 + part2 + math.e + 20.0

# --- 3. Core DE Logic (REMAINS THE SAME) ---

def ensure_bounds(vector):
    """Clips the vector's values to stay within the defined BOUNDS."""
    bounded_vector = []
    for i in range(len(vector)):
        val = vector[i]
        min_val, max_val = BOUNDS[i]
        if val < min_val:
            val = min_val
        elif val > max_val:
            val = max_val
        bounded_vector.append(val)
    return bounded_vector

def main():
    # Initialize Population
    population = [[random.uniform(BOUNDS[i][0], BOUNDS[i][1]) for i in range(NUM_DIMENSIONS)] for _ in range(POPULATION_SIZE)]

    best_solution_overall = None
    best_score_overall = float('inf')

    # Run Evolution Loop
    for generation in range(NUM_GENERATIONS):
        for i in range(POPULATION_SIZE):
            target_vector = population[i]

            # Mutation
            candidates = list(range(POPULATION_SIZE))
            candidates.remove(i)
            a_idx, b_idx, c_idx = random.sample(candidates, 3)
            a, b, c = population[a_idx], population[b_idx], population[c_idx]

            mutant_vector = [a[j] + F * (b[j] - c[j]) for j in range(NUM_DIMENSIONS)]
            mutant_vector = ensure_bounds(mutant_vector)

            # Crossover
            trial_vector = []
            j_rand = random.randrange(NUM_DIMENSIONS)
            for j in range(NUM_DIMENSIONS):
                if random.random() < CR or j == j_rand:
                    trial_vector.append(mutant_vector[j])
                else:
                    trial_vector.append(target_vector[j])

            # Selection
            target_score = objective_function(target_vector)
            trial_score = objective_function(trial_vector)

            if trial_score <= target_score:
                population[i] = trial_vector
                if trial_score < best_score_overall:
                    best_score_overall = trial_score
                    best_solution_overall = trial_vector
            # else, the target_vector remains in the population
        
        print(f"Generation {generation+1:03}: Best Score = {best_score_overall:.6f}")

    print("\nEvolution finished!")
    print(f"Best solution found: {best_solution_overall}")
    print(f"With score: {best_score_overall:.6f}")

if __name__ == "__main__":
    main()