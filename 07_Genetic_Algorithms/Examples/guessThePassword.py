import random

# --- The problem to solve ---
TARGET_PASSWORD = "Hello"
VALID_CHARACTERS = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890,.'!?"

# --- GA Parameters ---
POPULATION_SIZE = 100
MUTATION_RATE = 0.01

def create_individual():
    """Creates a single random individual (a guess)."""
    return ''.join(random.choice(VALID_CHARACTERS) for _ in range(len(TARGET_PASSWORD)))

def calculate_fitness(individual):
    """Calculates the fitness score of an individual."""
    score = 0
    for i in range(len(TARGET_PASSWORD)):
        if individual[i] == TARGET_PASSWORD[i]:
            score += 1
    return score

def selection(population):
    """Selects parents for the next generation."""
    fitness_scores = [calculate_fitness(ind) for ind in population]
    mating_pool = []
    for i in range(len(population)):
        # Add the individual to the pool a number of times proportional to its fitness
        for _ in range(fitness_scores[i] + 1):
             mating_pool.append(population[i])
    return mating_pool

def crossover(parent1, parent2):
    """Creates a child by combining two parents' genes."""
    midpoint = random.randint(1, len(TARGET_PASSWORD) - 1)
    child = parent1[:midpoint] + parent2[midpoint:]
    return child

def mutate(individual):
    """Applies random mutations to an individual."""
    mutated_individual = list(individual)
    for i in range(len(mutated_individual)):
        if random.random() < MUTATION_RATE:
            mutated_individual[i] = random.choice(VALID_CHARACTERS)
    return "".join(mutated_individual)

# --- Main GA Loop ---
def main():
    # Create the initial population
    population = [create_individual() for _ in range(POPULATION_SIZE)]
    
    generation = 0
    found = False

    while not found:
        generation += 1
        
        # 1. Selection
        mating_pool = selection(population)
        
        # 2. Create the next generation
        new_population = []
        for _ in range(POPULATION_SIZE):
            parent1 = random.choice(mating_pool)
            parent2 = random.choice(mating_pool)
            child = crossover(parent1, parent2)
            mutated_child = mutate(child)
            new_population.append(mutated_child)
            
        population = new_population
        
        # --- Check for solution ---
        best_fitness = 0
        best_individual = ""
        for individual in population:
            fitness = calculate_fitness(individual)
            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = individual

        print(f"Generation {generation:03}: Best Guess = '{best_individual}', Fitness = {best_fitness}/{len(TARGET_PASSWORD)}")
        
        if best_individual == TARGET_PASSWORD:
            print(f"\nTarget '{TARGET_PASSWORD}' found in generation {generation}!")
            found = True

if __name__ == "__main__":
    main()