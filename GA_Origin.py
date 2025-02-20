import json, os, sys
import GA_modules as ga_mod
from collections import defaultdict
import math, random
import pandas as pd
import time
from os import path

def read_json_data(filepath):
    '''
    Loads JSON file and returns a dictionary.
    '''
    try:
        with open(filepath, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        sys.exit(f"The '{filepath}' file is missing!")
    except json.JSONDecodeError as e:
        sys.exit(f"Error decoding JSON from '{filepath}': {e}")
    except Exception as e:
        sys.exit(f"Unexpected error reading '{filepath}': {e}")


def read_files(*filepaths):
    '''
    Reads multiple JSON files and returns their contents.
    '''
    return [read_json_data(filepath) for filepath in filepaths]

def is_nan_or_invalid(zone_id):
    """Check if the zone_id is NaN or otherwise invalid (None or empty string)."""
    # If zone_id is a string and not empty, it's considered valid.
    if isinstance(zone_id, str) and zone_id:
        return False
    # Use pandas to check for NaN or None, which works for both numbers and strings.
    if pd.isna(zone_id):
        return True
    # Add more conditions as needed, depending on what's considered 'invalid' in your context.
    return False
def closestNode(stops, node, bigM=1e9):
    """
    Finds the closest node with a zone_id to the given node within the same route.

    Parameters:
    - route: The route identifier.
    - node: The node identifier for which to find the closest node with a zone_id.
    - dataRoute: A dictionary containing route data, including stops with their lat, lng, and zone_id.
    - bigM: A large number used for initializing the minimum distance.

    Returns:
    - The identifier of the closest node with a zone_id, or an empty string if none found.
    """
    x_node = stops[node]['lat']
    y_node = stops[node]['lng']
    min_node = bigM
    which_node = ''  # Initialize with an empty string for cases where no node is found.

    for n, details in stops.items():
        if n == node or 'zone_id' not in details or not isinstance(details['zone_id'], str):
            continue
        x1 = details['lat']
        y1 = details['lng']
        dist_node = math.sqrt((x1 - x_node) ** 2 + (y1 - y_node) ** 2)
        if 0 < dist_node < min_node:
            min_node = dist_node
            which_node = n

    return which_node


def find_closest_unvisited(current_stop, remaining_stops, travel_distances):
    min_distance = float('inf')
    closest_stop = None
    for stop, _ in remaining_stops:
        if stop not in travel_distances[current_stop]: continue
        distance = travel_distances[current_stop][stop]
        # Check if the distance is non-negative and less than the minimum found so far
        if 0 <= distance < min_distance:
            min_distance = distance
            closest_stop = stop

    # If no stop is closer than infinity (i.e., all stops are unreachable), return None
    if min_distance == float('inf'):
        return None, None

    return closest_stop, min_distance

#closest_stop_id, transit_time_to_closest = find_closest_unvisited(current_stop_id, sorted_sequence[i + 1:],  travel[route_id])
def greedy_shortest_path(stops, distance_func):
    if not stops:
        return []

    route = [stops[0]]  # Start with the first stop as arbitrary starting point
    unvisited = set(stops[1:])  # Initially, all other stops are unvisited

    while unvisited:
        last_stop = route[-1]
        # Find the nearest unvisited stop
        nearest_stop, _ = min(((stop, distance_func(last_stop, stop)) for stop in unvisited), key=lambda x: x[1])
        route.append(nearest_stop)
        unvisited.remove(nearest_stop)

    return route

def nearest_neighbor_solution(stops, data_travel):
    """
    Generate a route using the Nearest Neighbor algorithm based on travel distances.

    Args:
        stops (list): List of stops, each with a .stop_ID attribute.
        data_travel (dict): Dictionary with distances between stops.

    Returns:
        list: Route generated using the Nearest Neighbor heuristic.
    """
    if not stops:
        return []

    # Initialize the route with the first stop (assuming it's the depot/station)
    route = [stops[0]]
    remaining_stops = stops[1:]  # Exclude the first stop (depot) from the remaining stops

    # Keep adding the nearest neighbor to the route until all stops are included
    while remaining_stops:
        last_stop = route[-1]
        # Find the nearest next stop
        next_stop = min(remaining_stops, key=lambda stop: data_travel[last_stop.stop_ID][stop.stop_ID])
        route.append(next_stop)
        remaining_stops.remove(next_stop)

    return route


def reverse_nearest_neighbor_solution(stops, data_travel):
    if not stops:
        return []

    # Begin with the station as the starting point
    route = [stops[0]]
    remaining_stops = stops[1:]

    # Find the nearest previous stop to the current first stop in the route
    while remaining_stops:
        next_stop = route[0]  # This should be the latest added stop which is closest to the remaining stops
        prev_stop = min(remaining_stops, key=lambda stop: data_travel[stop.stop_ID][next_stop.stop_ID])
        route.insert(0, prev_stop)  # Insert the found stop at the beginning to maintain the reverse build-up
        remaining_stops.remove(prev_stop)

    # Since the route was built in reverse, you can reverse it back to start with the station
    route.reverse()
    return route


def randomized_nearest_neighbor_solution(stops, data_travel, top_n=3):
    if not stops:
        return []

    route = [stops[0]]
    remaining_stops = stops[1:]

    while remaining_stops:
        last_stop = route[-1]
        # Sort neighbors by distance and pick a random one from the top N
        sorted_neighbors = sorted(remaining_stops, key=lambda stop: data_travel[last_stop.stop_ID][stop.stop_ID])
        if len(sorted_neighbors) > top_n:
            next_stop = random.choice(sorted_neighbors[:top_n])
        else:
            next_stop = sorted_neighbors[0]
        route.append(next_stop)
        remaining_stops.remove(next_stop)

    return route


def divide_into_sectors(stops):
    # Implement your logic to divide stops into sectors
    # This is a placeholder function
    return [stops]  # Simple placeholder, replace with actual sector division


def generate_initial_population(population_size, stops, data_travel):
    population = []
    if population_size > 0:
        # Standard Nearest Neighbor
        nn_solution = nearest_neighbor_solution(stops, data_travel)
        population.append(nn_solution)

        # Variants for additional heuristic-based solutions
        for variant in range(1, min(population_size // 2, 10)):  # Limit to a fraction of the population
            if variant == 1:
                # Example: Reverse NN or another heuristic
                modified_solution = reverse_nearest_neighbor_solution(stops, data_travel)
            else:
                # Randomized NN
                modified_solution = randomized_nearest_neighbor_solution(stops, data_travel)
            population.append(modified_solution)

    # Fill the rest with randomly shuffled routes for diversity
    while len(population) < population_size:
        shuffled_stops = stops[1:]  # Exclude the depot
        random.shuffle(shuffled_stops)
        chromosome = [stops[0]] + shuffled_stops
        population.append(chromosome)

    return population

def crossover(parent1, parent2):
    length = len(parent1)
    # Ensure the station is not included in the crossover points
    start, end = sorted(random.sample(range(1, length), 2))

    # Initialize offspring with None
    offspring1 = [None] * length
    offspring2 = [None] * length

    # Copy the selected range from each parent to the corresponding offspring
    offspring1[start:end] = parent1[start:end]
    offspring2[start:end] = parent2[start:end]

    # Function to fill in the remaining stops for offspring
    def fill_remaining_offspring(offspring, parent):
        current_pos = end
        for stop in parent:
            if stop not in offspring:  # Skip stops already present
                if current_pos >= length:  # Wrap around if at the end
                    current_pos = 1
                offspring[current_pos] = stop
                current_pos += 1

    # Fill in the remaining stops for each offspring
    fill_remaining_offspring(offspring1, parent2)
    fill_remaining_offspring(offspring2, parent1)

    offspring1[0] = parent1[0]  # Assuming the first element is the station
    offspring2[0] = parent2[0]
    return offspring1, offspring2


def incorporate_elitism(old_population, fitnesses, num_elites):
    """
    Incorporate elitism into the genetic algorithm by carrying over a specified number of elite individuals.

    Args:
        old_population (list): The current generation of individuals.
        fitnesses (list): The fitness scores corresponding to each individual in the old_population.
        num_elites (int): The number of elite individuals to carry over to the next generation.

    Returns:
        list: A list of elite individuals.
    """
    # Pair each individual with its fitness score
    paired_population = list(zip(old_population, fitnesses))

    # Sort the paired population by fitness score in descending order
    sorted_by_fitness = sorted(paired_population, key=lambda x: x[1], reverse=True)

    # Extract just the individuals from the sorted list to form the elite group
    elites = [individual for individual, fitness in sorted_by_fitness[:num_elites]]

    return elites

def swap_mutation(chromosome, mutation_rate):
    """
    Perform swap mutation on a chromosome with a given mutation rate.

    Args:
        chromosome (list): The chromosome to mutate; a list of stops.
        mutation_rate (float): Probability of mutation for each gene.

    Returns:
        list: The mutated chromosome.
    """
    mutated_chromosome = chromosome[:]
    for i in range(1, len(chromosome)):  # Start from 1 to exclude the depot/station if it's fixed
        if random.random() < mutation_rate:
            swap_with = random.randint(1, len(chromosome) - 1)  # Ensure we don't select the depot/station
            # Swap the two stops
            mutated_chromosome[i], mutated_chromosome[swap_with] = mutated_chromosome[swap_with], mutated_chromosome[i]
    return mutated_chromosome


def evaluate_population(population, data_travel):
    fitnesses = []

    for chromosome in population:
        travel_length = 0
        # Iterate through each gene in the chromosome except the last one
        for i in range(len(chromosome) - 1):
            # Current gene's stop ID
            current_stop_id = chromosome[i].stop_ID
            # Next gene's stop ID
            next_stop_id = chromosome[i + 1].stop_ID
            # Add the distance from the current stop to the next

            travel_length += data_travel[current_stop_id][next_stop_id]

        # Optionally, if the route is circular, add distance back to the start
        # first_stop_id = chromosome[0].stop_ID
        # last_stop_id = chromosome[-1].stop_ID
        # travel_length += data_travel[last_stop_id][first_stop_id]

        # The fitness could be the inverse of the travel length if you want to minimize distance
        # This way, a shorter route has a higher fitness score.
        fitness = 1 / travel_length if travel_length > 0 else 0
        fitnesses.append(fitness)

    return fitnesses


def tournament_selection(population, fitnesses, tournament_size=3):
    """Performs tournament selection to choose a single parent.

    Args:
        population (list): The current population of chromosomes.
        fitnesses (list): The fitness values corresponding to each chromosome.
        tournament_size (int): The number of individuals to select for each tournament.

    Returns:
        A single chromosome selected as a parent.
    """
    # Selecting random indices for the tournament
    selected_indices = random.sample(range(len(population)), tournament_size)

    # Finding the individual with the highest fitness in the tournament
    best_index = selected_indices[0]
    for index in selected_indices[1:]:
        if fitnesses[index] > fitnesses[best_index]:
            best_index = index

    # Returning the winning chromosome
    return population[best_index]


def select_parents(population, fitnesses, num_parents):
    """Selects a specified number of parents using tournament selection.

    Args:
        population (list): The current population of chromosomes.
        fitnesses (list): The fitness values corresponding to each chromosome.
        num_parents (int): The number of parents to select.

    Returns:
        A list of chromosomes selected to be parents.
    """
    parents = []
    for _ in range(num_parents):
        parent = tournament_selection(population, fitnesses)
        parents.append(parent)
    return parents


def Main(data_route, data_travel, population_size, num_generations, mutation_rate, tournament_size, time_limit):
    start_time = time.time()
    stops = data_route['stops']

    # Initialize an empty list to hold the gene instances
    chromosome = []

    station_gene = None
    other_genes = []

    for stop_id, stop_details in stops.items():
        # check if the zone id is present
        if stop_details['type'] != 'Station' and is_nan_or_invalid(stop_details['zone_id']):
            node_zone_id = closestNode(stops, stop_id)  # Assuming closestNode returns the correct format
            if node_zone_id:  # Ensure it's not empty or invalid
                stop_details['zone_id'] = stops[node_zone_id][ 'zone_id']
        # Create a Gene object for each stop
        gene = ga_mod.Gene(stop_id=stop_id, lat=stop_details['lat'], lng=stop_details['lng'],  stop_type=stop_details['type'], zone_id=stop_details['zone_id'])

        # Check if this stop is the station and handle accordingly
        if stop_details['type'] == 'Station':
            station_gene = gene
        else:
            other_genes.append(gene)

    # Ensure the station is at the front of the chromosome
    if station_gene:
        chromosome.append(station_gene)

    # Append the rest of the genes
    chromosome.extend(other_genes)
    population = generate_initial_population(population_size, chromosome, data_travel)

    for generation in range(num_generations):
        # Evaluate
        fitness_scores = evaluate_population(population, data_travel)
        new_population = incorporate_elitism(population, fitness_scores, int(population_size/5))
        while len(new_population) < len(population):
            # Selection
            parents = select_parents(population, fitness_scores, 2)

            # Crossover
            offspring1, offspring2 = crossover(parents[0], parents[1])

            # Mutation
            offspring1 = swap_mutation(offspring1, mutation_rate)
            offspring2 = swap_mutation(offspring2, mutation_rate)

            # Add offspring to the new population

            new_population.append(offspring1)
            if len(new_population) < population_size:
                new_population.append(offspring2)

        # Ensure the new_population is not larger than the original population size
        population = new_population[:population_size]

        # Check Termination Condition
        if time.time() - start_time > time_limit:
            print("Time limit reached. Terminating...")
            break
    final_fitness_scores = evaluate_population(population, data_travel)
    best_index = final_fitness_scores.index(max(final_fitness_scores))
    best_solution = population[best_index]
    best_fitness = final_fitness_scores[best_index]
    best_travel_length = 1 / best_fitness if best_fitness > 0 else float('inf')  # Avoid division by zero

    return best_solution


def create_stop_sequence_mapping(best_solution):
    # Initialize an empty dictionary to hold the mapping of stop_ID to sequence number
    stop_sequence_mapping = {}

    # Check if the best_solution is not empty
    if not best_solution:
        return stop_sequence_mapping  # Return empty mapping if the list is empty

    # The first stop (station) in the list should be assigned sequence number 0
    station_stop_id = best_solution[0].stop_ID
    stop_sequence_mapping[station_stop_id] = 0

    # Iterate over the stops in best_solution starting from the second element
    for sequence_number, stop in enumerate(best_solution[1:], start=1):
        # Map each stop's ID to its sequence number
        stop_sequence_mapping[stop.stop_ID] = sequence_number

    return stop_sequence_mapping


if __name__ == "__main__":
    t0 = time.time()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_paths = ['eval_route_data.json' ,'eval_travel_times.json']
    #file_paths = ['eval_route_data.json', 'eval_package_data.json', 'eval_travel_times.json']
    # Now the paths should correctly point to the 'data' folder
    full_paths = [os.path.join(base_dir, 'data/model_apply_inputs', path) for path in file_paths]

    # Read files
    routes,  travel = read_files(*full_paths)
    t1 = time.time() - t0
    print("Reading Data Done. Time elapsed: ", t1)
    answer = {}
    counter =0
    for route_id, route_details in routes.items():
        t2 = time.time()
        best_solution = Main(route_details, travel[route_id], population_size = 30, num_generations = 300, mutation_rate =0.4, tournament_size=3, time_limit=6)
        solution = create_stop_sequence_mapping(best_solution)
        proposed = {'proposed': solution}
        answer[route_id] = proposed
        t3 = time.time() - t2
        print(f"Solved GA Completed. Time Elapsed: {counter}. Execution Time: {t3}")
        counter += 1
    output_path = path.join(base_dir, 'data/model_apply_outputs/proposed_sequences.json')
    with open(output_path, 'w') as outfile:
        json.dump(answer, outfile)

    print("Success: The '{}' file has been saved".format(output_path))

    t4 = time.time() - t0
    print("All Done. Time elapsed: ", t4)
    print("test")

    print("test")
