import random

# --- 1. DE Parameters (Tune these for your problem) ---
POPULATION_SIZE = 50
NUM_GENERATIONS = 300
F = 0.8  # Scaling Factor (differential weight), usually in [0, 2]
CR = 0.7 # Crossover Rate, usually in [0, 1]

# --- 2. Problem-Specific Setup (YOU WILL MODIFY THESE) ---

# Define the boundaries for each variable in the solution vector.
# EXAMPLE: For a 2D problem (e.g., f(x, y)), with x and y between -5 and 5.
NUM_DIMENSIONS = 2
BOUNDS = [(-5.0, 5.0)] * NUM_DIMENSIONS # [(min, max), (min, max), ...]

def objective_function(vector):
    """
    Calculates the score of a solution vector.
    This is the function you want to MINIMIZE.
    If you want to MAXIMIZE a function, simply return its negative value.
    """
    # EXAMPLE: Sphere function -> f(x, y) = x^2 + y^2
    # The global minimum is 0 at (0, 0).
    x, y = vector
    return x**2 + y**2

# --- 3. Core DE Logic (USUALLY REMAINS THE SAME) ---

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
    # 1. Initialize Population
    # Create a population of random vectors, where each value is within its bounds.
    population = []
    for _ in range(POPULATION_SIZE):
        individual = [random.uniform(BOUNDS[i][0], BOUNDS[i][1]) for i in range(NUM_DIMENSIONS)]
        population.append(individual)

    best_solution_overall = None
    best_score_overall = float('inf') # For minimization, we start with infinity

    # 2. Run Evolution Loop
    for generation in range(NUM_GENERATIONS):
        new_population = []
        for i in range(POPULATION_SIZE):
            target_vector = population[i]

            # a. Mutation
            # Select three distinct individuals a, b, and c from the population
            candidates = list(range(POPULATION_SIZE))
            candidates.remove(i)
            a_idx, b_idx, c_idx = random.sample(candidates, 3)
            a, b, c = population[a_idx], population[b_idx], population[c_idx]

            # Create the mutant vector
            mutant_vector = [a[j] + F * (b[j] - c[j]) for j in range(NUM_DIMENSIONS)]
            mutant_vector = ensure_bounds(mutant_vector) # Clip values to bounds

            # b. Crossover (Recombination)
            trial_vector = []
            j_rand = random.randrange(NUM_DIMENSIONS) # Index for guaranteed swap
            for j in range(NUM_DIMENSIONS):
                if random.random() < CR or j == j_rand:
                    trial_vector.append(mutant_vector[j])
                else:
                    trial_vector.append(target_vector[j])

            # c. Selection
            target_score = objective_function(target_vector)
            trial_score = objective_function(trial_vector)

            # If trial vector is better (or equal), it replaces the target
            if trial_score <= target_score:
                new_population.append(trial_vector)
                if trial_score < best_score_overall:
                    best_score_overall = trial_score
                    best_solution_overall = trial_vector
            else:
                new_population.append(target_vector)
                # Check if the old vector was the best one so far
                if target_score < best_score_overall:
                    best_score_overall = target_score
                    best_solution_overall = target_vector
        
        population = new_population
        
        print(f"Generation {generation+1:03}: Best Score = {best_score_overall:.6f}")

    print("\nEvolution finished!")
    print(f"Best solution found: {best_solution_overall}")
    print(f"With score: {best_score_overall:.6f}")

if __name__ == "__main__":
    main()