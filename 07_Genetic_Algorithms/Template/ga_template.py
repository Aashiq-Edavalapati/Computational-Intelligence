import random

# --- 1. GA Parameters (Tune these for the problem) ---
POPULATION_SIZE = 100
MUTATION_RATE = 0.01
NUM_GENERATIONS = 200
# Add any other problem-specific constants here
# e.g., TARGET_LENGTH = 10

# --- 2. Problem-Specific Functions ---
def create_individual():
    """
        Creates a single individual (chromosome) for the population.
        The structure of an individual depends entirely on the problem.
            - For a password guesser, it's a string.
            - For a math problem (e.g., find max of f(x)), it might be a binary string or a number.
            - For a traveling salesman problem, it might be a list of cities.
    """
    # EXAMPLE: A list of 10 random numbers between 0 and 1
    return [random.random() for _ in range(10)]

def calculate_fitness(individual):
    """
        Calculates the fitness score of an individual.
        This function is the heart of the GA. It must return a single
        numerical score indicating how "good" the solution is.
    """
    # EXAMPLE: Sum of the numbers in the list (we want to maximize this)
    return sum(individual)

# --- 3. Core GA Logic (USUALLY REMAINS THE SAME) ---
def selection(population):
    """
        Selects parents based on fitness. Higher fitness means a higher
        chance of being selected.
    """
    fitness_scores = [calculate_fitness(ind) for ind in population]
    total_fitness = sum(fitness_scores)
    
    # Handle case where total_fitness is zero to avoid division errors
    if total_fitness == 0:
        # If all fitnesses are 0, select randomly
        return [random.choice(population) for _ in range(POPULATION_SIZE)]

    # Create a mating pool where fitter individuals appear more often
    mating_pool = []
    for i in range(len(population)):
        # Number of times an individual is added is proportional to its fitness
        num_copies = int((fitness_scores[i] / total_fitness) * POPULATION_SIZE * 2)
        for _ in range(num_copies):
            mating_pool.append(population[i])
    
    # If the mating pool is empty due to very low fitness scores, fill it randomly
    if not mating_pool:
        return [random.choice(population) for _ in range(POPULATION_SIZE)]
        
    return mating_pool

def crossover(parent1, parent2):
    """
        Creates a child by combining two parents' genes.
        This single-point crossover works well for list-based individuals.
    """
    if len(parent1) < 2: return parent1 # Cannot perform crossover on single-item list
    midpoint = random.randint(1, len(parent1) - 1)
    child = parent1[:midpoint] + parent2[midpoint:]
    return child

def mutate(individual):
    """
        Applies random mutations to an individual's genes.
        The way mutation works might need to be adapted to the individual's structure.
    """
    mutated_individual = individual[:] # Create a copy
    for i in range(len(mutated_individual)):
        if random.random() < MUTATION_RATE:
            # The mutation logic depends on the gene type.
            # EXAMPLE: For a list of numbers, we replace it with a new random number.
            mutated_individual[i] = random.random()
    return mutated_individual

# --- 4. Main Execution ---

def main():
    # 1. Initialize Population
    population = [create_individual() for _ in range(POPULATION_SIZE)]
    
    best_solution_overall = None
    best_fitness_overall = -1

    # 2. Run Evolution Loop
    for generation in range(NUM_GENERATIONS):
        # a. Selection
        mating_pool = selection(population)
        
        # b. Create Next Generation
        new_population = []
        for _ in range(POPULATION_SIZE):
            parent1 = random.choice(mating_pool)
            parent2 = random.choice(mating_pool)
            
            # c. Crossover
            child = crossover(parent1, parent2)
            
            # d. Mutation
            mutated_child = mutate(child)
            
            new_population.append(mutated_child)
        
        population = new_population
        
        # --- Logging and tracking progress ---
        best_individual_current_gen = None
        best_fitness_current_gen = -1

        for individual in population:
            fitness = calculate_fitness(individual)
            if fitness > best_fitness_current_gen:
                best_fitness_current_gen = fitness
                best_individual_current_gen = individual
        
        if best_fitness_current_gen > best_fitness_overall:
            best_fitness_overall = best_fitness_current_gen
            best_solution_overall = best_individual_current_gen
        
        print(f"Generation {generation+1:03}: Best Fitness = {best_fitness_current_gen:.4f}")

    print("\nEvolution finished!")
    print(f"Best solution found: {best_solution_overall}")
    print(f"With fitness: {best_fitness_overall:.4f}")

if __name__ == "__main__":
    main()