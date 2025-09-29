import random
import numpy as np

NUM_ITEMS = 5
DEMAND = [100, 150, 80, 120, 90]
HOLDING_COST = [1, 1.5, 1.2, 1, 1.3]
SHORTAGE_COST = [5, 6, 4, 5.5, 6]

GENS = 15
MUT_RATE = 0.1
POP_SIZE = 20

def create_individual():
    return [random.randint(50, 200) for _ in range(NUM_ITEMS)]

# fitness fn
def fitness(individual):
    total_cost = 0
    for i in range(NUM_ITEMS):
        reorder_point = individual[i]
        expected_demand = DEMAND[i]
        holding_cost = HOLDING_COST[i]
        shortage_cost = SHORTAGE_COST[i]
        avg_inventory = max(0, reorder_point - expected_demand)
        avg_shortage = max(0, expected_demand - reorder_point)
        cost = holding_cost * avg_inventory + shortage_cost * avg_shortage
        total_cost += cost
    return -total_cost

def selection(population, scores):
    adjusted_scores = [s - min(scores) + 1 for s in scores]
    selected = random.choices(population=population, weights=adjusted_scores, k=2)
    return selected

def crossover(p1, p2):
    point = random.randint(1, NUM_ITEMS - 1)
    return p1[:point] + p2[point:], p2[:point] + p1[point:]

def mutate(individual):
    for i in range(NUM_ITEMS):
        if random.random() < MUT_RATE:
            individual[i] = random.randint(50, 200)

def genetic_algorithm():
    population = [create_individual() for _ in range(POP_SIZE)]
    best_solution = None
    best_score = float('-inf')
    for gen in range(GENS):
        scores = [fitness(ind) for ind in population]
        for i in range(POP_SIZE):
            if scores[i] > best_score:
                best_score = scores[i]
                best_solution = population[i]
        next_gen = []
        while len(next_gen) < POP_SIZE:
            p1, p2 = selection(population, scores)
            child1, child2 = crossover(p1, p2)
            mutate(child1)
            mutate(child2)
            next_gen.extend([child1, child2])
        population = next_gen[:POP_SIZE]
        if gen % 10 == 0:
            print(f"Generation {gen}: Best Cost = {-best_score:.2f}")
    return best_solution, -best_score

if __name__ == "__main__":
    best_solution, best_cost = genetic_algorithm()
    print("\nBest reorder points:", best_solution)
    print("Minimum total cost:", best_cost)
