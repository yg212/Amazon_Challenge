import json, os, sys
import GA_modules as ga_mod
from collections import defaultdict
import math
import pandas as pd
import random
import time
import copy
from os import path
import ast

random.seed(0)

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

def calculate_zone_distances(zone_groups, data_travel):
    """
    Calculate and return a dictionary of distances between each pair of zones.
    """
    zone_distances = {}
    for zone_id_1, stops_1 in zone_groups.items():
        for zone_id_2, stops_2 in zone_groups.items():
            if zone_id_1 != zone_id_2:
                # Simple approach: Use the distance between the first stop of each zone
                distance = data_travel[stops_1[0].stop_ID][stops_2[0].stop_ID]
                zone_distances[(zone_id_1, zone_id_2)] = distance
    return zone_distances


def calculate_historical_score(precedence_matrix, zone_a, zone_b):
    """
    Safely calculate the historical score between two zones, considering both directions.

    :param precedence_matrix: The matrix containing historical scores between zone pairs.
    :param zone_a: The ID of the first zone.
    :param zone_b: The ID of the second zone.
    :return: The total historical score for both directions.
    """
    # Initialize the historical score
    historical_score = 0

    # Add score from zone_a to zone_b if it exists
    if (zone_a, zone_b) in precedence_matrix:
        historical_score += precedence_matrix[(zone_a, zone_b)]

    # Add score from zone_b to zone_a if it exists
    if (zone_b, zone_a) in precedence_matrix:
        historical_score += precedence_matrix[(zone_b, zone_a)]

    return historical_score


def nearest_zone_sequence(zone_groups, data_travel, similarity_weight=0.5, variation_depth=5):
    """
    Determine a sequence to visit zones based on a combination of nearest zone approach and zone similarity,
    starting from a unique starting point (e.g., a station) that does not have similarity measures with other zones.
    Introduces variation in the first zone selection by considering one of the top N closest zones.

    :param zone_groups: Dictionary of zones with their details
    :param data_travel: Dictionary of travel distances between zones
    :param similarity_weight: Weight to give to similarity in decision-making (0 to 1)
    :param variation_depth: Number of top closest zones to consider for initial variation
    :return: List of zone IDs in the sequence they should be visited
    """
    zone_distances = calculate_zone_distances(zone_groups, data_travel)

    visited_zones = set()
    current_zone = next(iter(zone_groups))  # Start with the first zone, usually the station
    sequence = [current_zone]
    visited_zones.add(current_zone)

    # For the first step, select randomly among the top N closest zones to introduce variation
    if len(zone_groups) > 1:
        first_step_candidates = sorted([
            (zone_id, dist) for ((zone_1, zone_id), dist) in zone_distances.items()
            if is_nan_or_invalid(zone_1) and zone_id not in visited_zones], key=lambda x: x[1])
        variation_candidates = first_step_candidates[:variation_depth]  # Get the top N closest zones
        next_zone = random.choice(variation_candidates)[0]  # Randomly choose among the top N
        sequence.append(next_zone)
        visited_zones.add(next_zone)
        current_zone = next_zone

    # For subsequent zones, consider both distance and similarity
    while len(visited_zones) < len(zone_groups):
        candidates = [(zone_id, dist, calculate_similarity(current_zone, zone_id)) for (zone_1, zone_id), dist in
                      zone_distances.items() if zone_1 == current_zone and zone_id not in visited_zones]

        similarity_weight = random.random()
        # Adjust each candidate's score based on distance and similarity
        adjusted_candidates = [(zone_id, (similarity * similarity_weight)/4 -(dist * (1 - similarity_weight))/100) for
                               zone_id, dist, similarity in candidates]

        next_zone = max(adjusted_candidates, key=lambda x: x[1])[0]
        sequence.append(next_zone)
        visited_zones.add(next_zone)
        current_zone = next_zone

    return sequence


def generate_initial_population(zone_groups, data_travel, population_size):
    population = []
    all_zones = [zone_id for zone_id in zone_groups.keys() if
                 zone_id != 'nan' and not (isinstance(zone_id, float) and math.isnan(zone_id))]

    for _ in range(population_size):
        random.shuffle(all_zones)

        solution = {}
        last_stop_id = None  # Track the last stop from the previous zone

        # Assuming the station ('nan') is at the beginning and contains at least one station
        nan_key = next((key for key in zone_groups.keys() if key == 'nan' or (isinstance(key, float) and math.isnan(key))), None)
        if nan_key is not None and zone_groups[nan_key]:
            station_stop_id = zone_groups[nan_key][0].stop_ID  # First stop in the station zone
            last_stop_id = station_stop_id  # Starting point for the nearest neighbor search
            solution[nan_key] = zone_groups[nan_key]

        for zone_id in all_zones:
            stops = zone_groups[zone_id]
            # Use the last stop from the previous zone as the starting point
            if last_stop_id is not None and stops:
                sorted_stops = sorted(stops, key=lambda stop: data_travel[last_stop_id].get(stop.stop_ID, float('inf')))
                solution[zone_id] = nearest_neighbor_solution(sorted_stops, data_travel)
            else:
                solution[zone_id] = nearest_neighbor_solution(stops, data_travel)

            if solution[zone_id]:  # Update last_stop_id for the next zone
                last_stop_id = solution[zone_id][-1].stop_ID

        population.append(solution)

    return population

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

def calculate_similarity(zone1, zone2):
    def parse_zone(zone):
        parts = zone.split("-")
        if len(parts) < 2:
            return None, None, None, None
        P = parts[0]
        subparts = parts[1].split(".")
        if len(subparts) < 2:
            return P, None, None, None

        rkm = subparts[0]
        if len(subparts[1]) >= 2:
            rkm2 = subparts[1][:-1]  # All except the last character
            hrf = subparts[1][-1]  # The last character
        else:
            return P, rkm, None, None

        return P, rkm, rkm2, hrf

    def is_one_step_difference(a, b):
        try:
            # Try to handle as numbers first
            return abs(int(a) - int(b)) == 1
        except ValueError:
            # Handle as characters
            return abs(ord(a) - ord(b)) == 1

    P1, rkm11, rkm12, hrf1 = parse_zone(zone1)
    P2, rkm21, rkm22, hrf2 = parse_zone(zone2)

    if P1 != P2:
        return 0

    differences = 0
    close_differences = 0  # For tracking 'one step' differences

    # Check main part differences
    if rkm11 != rkm21:
        differences += 1
        if is_one_step_difference(rkm11, rkm21):
            close_differences += 1

    # Check sub-part differences
    if rkm12 != rkm22:
        differences += 1
        if is_one_step_difference(rkm12, rkm22):
            close_differences += 1

    # Check last character differences
    if hrf1 != hrf2:
        differences += 1
        if is_one_step_difference(hrf1, hrf2):
            close_differences += 1

    # Logic for scoring based on differences
    if differences == 1 and close_differences == 1:
        return 6  # Highest value for exactly one 'one step' difference
    elif differences == 1:
        return 4
    else:
        return 2

    return 0  # Fallback, should not ideally reach here


def is_nan(value):
    """Check if the value is float NaN."""
    try:
        return math.isnan(value)
    except TypeError:
        return False


def load_precedence_matrix(filename):
    with open(filename, 'r') as f:
        # Load the JSON data
        precedence_matrix_str_keys = json.load(f)

    # Convert string keys back to tuples for the inner dictionaries
    precedence_matrix = {
        station_code: {tuple(eval(pair)): count for pair, count in station_matrix.items()}
        for station_code, station_matrix in precedence_matrix_str_keys.items()
    }

    return precedence_matrix

# Usage
filtered_precedence_matrix_by_station = load_precedence_matrix("C:/Users/17657/PycharmProjects/GA_Amazon_Challenge/precedence_matrix_0324.json")
def compare_specific_zones(general_precedence_matrix_by_station, station_code, zone1, zone2):
    """
    Compare the precedence of two zones within the context of a specific station.

    :param general_precedence_matrix_by_station: The general precedence matrix organized by station codes.
    :param station_code: The station code to look up the precedence matrix.
    :param zone1: The first zone ID.
    :param zone2: The second zone ID.
    """
    # Access the specific station's precedence matrix
    precedence_matrix = general_precedence_matrix_by_station.get(station_code, {})

    # Check direct precedence
    direct_precedence = precedence_matrix.get((zone1, zone2), 0)
    #reverse_precedence = precedence_matrix.get((zone2, zone1), 0)

    # Print and return the comparison results
    return direct_precedence

def compare_specific_zones2(general_precedence_matrix_by_station, station_code, zone1, zone2):
    """
    Compare the precedence of two zones within the context of a specific station.

    :param general_precedence_matrix_by_station: The general precedence matrix organized by station codes.
    :param station_code: The station code to look up the precedence matrix.
    :param zone1: The first zone ID.
    :param zone2: The second zone ID.
    """
    # Access the specific station's precedence matrix
    precedence_matrix = general_precedence_matrix_by_station.get(station_code, {})

    # Check direct precedence
    direct_precedence = precedence_matrix.get((zone1, zone2), 0)
    reverse_precedence = precedence_matrix.get((zone2, zone1), 0)

    # Print and return the comparison results
    if direct_precedence > reverse_precedence:
        print(
            f"'{zone1}' is more commonly seen before '{zone2}' in station {station_code} ({direct_precedence} vs {reverse_precedence} times).")

    elif reverse_precedence > direct_precedence:
        print(
            f"'{zone2}' is more commonly seen before '{zone1}' in station {station_code} ({reverse_precedence} vs {direct_precedence} times).")

def evaluate_population(population, data_travel, precedence_matrix, station_code):
    evaluations = []

    for individual in population:
        total_travel_length = 0
        total_similarity = 0
        history_similarity_scores = []

        keys = list(individual.keys())
        station_zone_id = keys[0]  # Assume first key is the station zone
        station_stop_id = individual[station_zone_id][0].stop_ID  # First stop's ID

        # Compute initial travel from station to the first zone
        if len(keys) > 1:
            first_zone_first_stop_id = individual[keys[1]][0].stop_ID
            total_travel_length += data_travel.get(station_stop_id, {}).get(first_zone_first_stop_id, 0)

        # Iterate over zones for consecutive calculations
        for i in range(1, len(keys)):
            zone_id = keys[i]
            stops = individual[zone_id]

            # Calculate travel length within the zone
            for j in range(len(stops) - 1):
                travel_length = data_travel.get(stops[j].stop_ID, {}).get(stops[j + 1].stop_ID, 0)
                total_travel_length += travel_length

            # Inter-zone travel and similarity for consecutive zones
            if i < len(keys) - 1:
                next_zone_id = keys[i + 1]
                inter_zone_travel_length = data_travel.get(stops[-1].stop_ID, {}).get(individual[next_zone_id][0].stop_ID, 0)
                total_travel_length += inter_zone_travel_length
                similarity = calculate_similarity(zone_id, next_zone_id)
                total_similarity += similarity

        # Compute distance back to the station from the last zone
        if len(keys) > 1:
            last_zone_last_stop_id = individual[keys[-1]][-1].stop_ID
            travel_length_back_to_station = data_travel.get(last_zone_last_stop_id, {}).get(station_stop_id, 0)
            total_travel_length += travel_length_back_to_station

        missing_station_codes = []
        # Only consider valid zone pairs for historical similarity
        filtered_keys = [key for key in keys if not is_nan_or_invalid(key)]

        valid_pairs = [(filtered_keys[i], filtered_keys[j]) for i in range(len(filtered_keys) - 1) for j in  range(i + 1, len(filtered_keys))]
        if station_code in precedence_matrix:
            history_similarity_scores = [compare_specific_zones(precedence_matrix, station_code, pair[0], pair[1]) for pair in valid_pairs if pair in precedence_matrix[station_code]]

        else:
            missing_station_codes.append(station_code)
        #if station_code in precedence_matrix:
        #    for pair in valid_pairs:
        #        if compare_specific_zones2(precedence_matrix, station_code, pair[0], pair[1]):
        #            print("historical precendence")
        #            print(compare_specific_zones2(precedence_matrix, station_code, pair[0], pair[1]))
        history_similarity = sum(history_similarity_scores)  # Adjust the calculation as needed

        # Calculate fitness
        fitness = (total_similarity / 4 + history_similarity/5) - (total_travel_length / 100)
        evaluations.append((fitness, total_similarity, total_travel_length))

    return evaluations

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

    # Initialize a set for tracking seen individuals
    seen = set()
    elites = []

    # Define a helper function to add elites based on different criteria
    def add_elites(criteria, reverse=True):
        for individual, fitness in sorted(paired_population, key=lambda x: x[1][criteria], reverse=reverse)[
                                   :num_elites]:
            if id(individual) not in seen:
                elites.append(individual)
                seen.add(id(individual))

    # Add elites based on the different criteria
    add_elites(criteria=0)  # Best overall
    add_elites(criteria=1)  # Highest similarity
    add_elites(criteria=2, reverse=False)  # Lowest distance

    return elites

def weighted_random_choice(choices, weights):
    """Select a random item from choices with a given probability distribution."""
    total = sum(weights)
    r = random.uniform(0, total)
    upto = 0
    for c, w in zip(choices, weights):
        if upto + w >= r:
            return c
        upto += w
    assert False, "Shouldn't get here"
def pmx_crossover(parent1, parent2):
    offspring1 = parent1.copy()  # Make deep copies if the lists are objects
    offspring2 = parent2.copy()

    # Exclude the first zone (station) and prepare a weighted selection based on the number of stops
    zones = list(parent1.keys())[1:]
    weights = [len(parent1[zone]) for zone in zones]  # Weight by total stops in both parents

    # Choose a zone for crossover based on the weighted probability
    if not zones:
        return offspring1, offspring2  # Return as is if no valid zones

    selected_zone = weighted_random_choice(zones, weights)

    # Proceed with PMX crossover logic within the selected zone
    p1_zone_sequence = parent1[selected_zone]
    p2_zone_sequence = parent2[selected_zone]
    size = len(p1_zone_sequence)

    # Ensure there are enough elements for crossover
    if size < 2:
        return offspring1, offspring2

    cx_point1, cx_point2 = sorted(random.sample(range(size), 2))

    # Function to find the index of a stop by its ID, could be placed outside pmx_crossover for reuse
    def find_index_by_attribute(search_list, target_attribute_value):
        return next((i for i, item in enumerate(search_list) if item.stop_ID == target_attribute_value), -1)

    for j in range(size):
        if j < cx_point1 or j > cx_point2:
            # For offspring1, resolve conflicts by finding replacements in parent1 based on mapping established with parent2
            if offspring1[selected_zone][j] in offspring1[selected_zone][cx_point1:cx_point2 + 1]:
                conflict_gene_stop_ID = offspring1[selected_zone][j].stop_ID
                replacement_index = find_index_by_attribute(parent2[selected_zone], conflict_gene_stop_ID)
                while replacement_index != -1 and parent1[selected_zone][replacement_index] in offspring1[
                    selected_zone]:
                    replacement_index = find_index_by_attribute(parent2[selected_zone], parent1[selected_zone][replacement_index].stop_ID)
                if replacement_index != -1:
                    offspring1[selected_zone][j] = parent1[selected_zone][replacement_index]

            # Repeat the process for offspring2, using parent2 as the source of replacements
            if offspring2[selected_zone][j] in offspring2[selected_zone][cx_point1:cx_point2 + 1]:
                conflict_gene_stop_ID = offspring2[selected_zone][j].stop_ID
                replacement_index = find_index_by_attribute(parent1[selected_zone], conflict_gene_stop_ID)
                while replacement_index != -1 and parent2[selected_zone][replacement_index] in offspring2[selected_zone]:
                    replacement_index = find_index_by_attribute(parent1[selected_zone], parent2[selected_zone][replacement_index].stop_ID)
                if replacement_index != -1:
                    offspring2[selected_zone][j] = parent2[selected_zone][replacement_index]

    return offspring1, offspring2

def reverse_sequence_mutation(offspring, mutation_rate=0.1):
    # Choose a zone at random, excluding the station
    zones = list(offspring.keys())[1:]
    if not zones:
        return  # No zones to mutate

    selected_zone = random.choice(zones)
    stops = offspring[selected_zone]

    # Reverse the entire sequence of stops within the selected zone with a certain probability
    if len(stops) > 1 and random.random() < mutation_rate:  # Check there's more than one stop to reverse
        offspring[selected_zone] = stops[::-1]  # Reverse the whole sequence

    return offspring

def scramble_stops_mutation(offspring, mutation_rate=0.1):
    # Choose a zone at random, excluding the station
    zones = list(offspring.keys())[1:]
    if not zones:
        return  # No zones to mutate

    selected_zone = random.choice(zones)
    stops = offspring[selected_zone]

    # Scramble a subset of stops within the selected zone with a certain probability
    if len(stops) > 2 and random.random() < mutation_rate:  # Check there's more than two stops to scramble
        # Determine the subset to scramble (at least two stops)
        start_idx, end_idx = sorted(random.sample(range(len(stops)), 2))
        # Scramble the selected subset of stops
        subset_to_scramble = stops[start_idx:end_idx+1]
        random.shuffle(subset_to_scramble)
        # Replace the original subset with the scrambled one
        offspring[selected_zone] = stops[:start_idx] + subset_to_scramble + stops[end_idx+1:]

    return offspring

def multi_zone_scramble_mutation(individual, mutation_rate):
    # Assuming the first key is for the station and should be preserved
    zone_ids = list(individual.keys())[1:]  # Skip the station zone

    for selected_zone in zone_ids:
        if random.random() < mutation_rate:
            # Proceed with scrambling the stops within the selected zone
            stops = individual[selected_zone]
            # Ensure there's a meaningful number of stops to scramble
            if len(stops) > 2:
                scramble_section = stops[:]  # Copy the list to avoid altering the original sequence during iteration
                random.shuffle(scramble_section)  # Scramble the copied section
                individual[selected_zone] = scramble_section  # Replace the original section with the scrambled one

    return individual

def find_station_key(parent_dict):
    for key in parent_dict.keys():
        if isinstance(key, float) and math.isnan(key):
            return key
    raise ValueError("Station key not found in parent dictionary.")


def inter_zone_crossover(parent1, parent2):
    # Assuming the first key in both parents is 'nan' for the station
    station_key = next(iter(parent1))  # Gets the first key, which should be 'nan'

    # Extract zone IDs excluding the station
    zone_ids = [zone for zone in parent1.keys() if zone != station_key]

    # Ensure there are enough zones for a two-point crossover
    if len(zone_ids) <= 4:
        print("Not enough zones for a two-point crossover.")

    # Select two distinct points for the crossover
    crossover_point1, crossover_point2 = sorted(random.sample(range(1, len(zone_ids)), 2))

    # Creating new sequences for offspring by mixing between the crossover points
    middle_sequence1 = [zone for zone in parent2.keys() if zone in zone_ids[crossover_point1:crossover_point2]]
    middle_sequence2 = [zone for zone in parent1.keys() if zone in zone_ids[crossover_point1:crossover_point2]]

    new_sequence1 = zone_ids[:crossover_point1] + middle_sequence1 + zone_ids[crossover_point2:]
    new_sequence2 = zone_ids[:crossover_point1] + middle_sequence2 + zone_ids[crossover_point2:]

    # Construct offspring by explicitly following the new sequences, adding the station back at the start
    offspring1, offspring2 = {station_key: parent1[station_key]}, {station_key: parent2[station_key]}
    for zone in new_sequence1:
        offspring1[zone] = parent2.get(zone, parent1[zone])
    for zone in new_sequence2:
        offspring2[zone] = parent1.get(zone, parent2[zone])
    return offspring1, offspring2


def inversion_mutation(individual, mutation_rate):
    # Assuming the first key is 'nan' for the station and should be preserved
    station_key = next(iter(individual))  # Gets the first key, which should be 'nan'
    station_value = individual[station_key]

    # Extract zone IDs excluding the station 'nan'
    zone_ids = [zone for zone in individual.keys() if zone != station_key]

    if random.random() < mutation_rate:
        # Select a subset of the sequence for inversion
        start, end = sorted(random.sample(range(len(zone_ids)), 2))
        # Invert the order of the selected segment
        zone_ids[start:end] = reversed(zone_ids[start:end])

        # Reconstruct the individual with the mutated zone sequence, preserving the station
        new_individual = {station_key: station_value}
        for zone_id in zone_ids:
            new_individual[zone_id] = individual[zone_id]

    else:
        # If no mutation, return the individual as is
        new_individual = individual

    return new_individual


def scramble_mutation(individual, mutation_rate):
    # Assuming the first key is 'nan' for the station and should be preserved
    station_key = next(iter(individual))  # Gets the first key, which should be 'nan'
    station_value = individual[station_key]

    # Extract zone IDs excluding the station 'nan'
    zone_ids = [zone for zone in individual.keys() if zone != station_key]

    if random.random() < mutation_rate:
        # Select a subset of the sequence to scramble
        start, end = sorted(random.sample(range(len(zone_ids)), 2))
        # Extract the segment to be scrambled
        segment_to_scramble = zone_ids[start:end]
        # Shuffle the selected segment
        random.shuffle(segment_to_scramble)
        # Replace the original segment with the scrambled one
        zone_ids[start:end] = segment_to_scramble

        # Reconstruct the individual with the mutated zone sequence, preserving the station
        new_individual = {station_key: station_value}
        for zone_id in zone_ids:
            new_individual[zone_id] = individual[zone_id]

    else:
        # If no mutation, return the individual as is
        new_individual = individual

    return new_individual


def check_historical_coverage(zone_groups, precedence_matrix):
    # Count zones in historical data, excluding 'NaN' or invalid ones
    zones_in_history_count = 0
    valid_zones_count = 0

    for zone_id in zone_groups.keys():
        if not is_nan_or_invalid(zone_id):  # Assuming is_nan_or_invalid is a function you've defined
            valid_zones_count += 1
            if zone_id in precedence_matrix or any(
                    (zone_id, other_zone) in precedence_matrix or (other_zone, zone_id) in precedence_matrix for
                    other_zone in zone_groups if not is_nan_or_invalid(other_zone)):
                zones_in_history_count += 1

    # Determine if more than half of the valid zones are covered by historical data
    return zones_in_history_count / valid_zones_count > 0.5 if valid_zones_count > 0 else False

def Main(data_route, data_travel, population_size, num_generations, mutation_rate, tournament_size, time_limit, debug):
    start_time = time.time()
    stops = data_route['stops']
    station_code = data_route['station_code']

    # Initialize lists for station and other genes
    station_gene = None
    other_genes = []

    # Cache for closestNode results to avoid repeated computation
    closest_node_cache = {}

    for stop_id, stop_details in stops.items():
        # Handle missing or invalid zone IDs
        if stop_details['type'] != 'Station' and is_nan_or_invalid(stop_details['zone_id']):
            # Use cached result if available
            if stop_id not in closest_node_cache:
                closest_node_cache[stop_id] = closestNode(stops, stop_id)
            node_zone_id = closest_node_cache[stop_id]

            # Update zone_id if a valid node_zone_id is found
            if node_zone_id:
                stop_details['zone_id'] = stops[node_zone_id]['zone_id']

        # Create Gene object
        gene = ga_mod.Gene(stop_id=stop_id, lat=stop_details['lat'], lng=stop_details['lng'],
                           stop_type=stop_details['type'], zone_id=stop_details['zone_id'])

        if stop_details['type'] == 'Station':
            station_gene = gene
        else:
            other_genes.append(gene)

    # Construct the chromosome with the station gene at the front if it exists
    chromosome = ([station_gene] if station_gene else []) + other_genes

    # Group genes by zone_id using a defaultdict
    zone_groups = defaultdict(list)
    for gene in chromosome:
        zone_groups[gene.zone_id].append(gene)


    # Generate initial population and optionally evaluate it
    population = generate_initial_population(zone_groups, data_travel, population_size)
    # If you need to evaluate the initial population immediately:
    #fitness_scores_initial = evaluate_population(population, data_travel, filtered_precedence_matrix_by_station, station_code)

    for generation in range(num_generations):
        # Evaluate

        fitness_scores = evaluate_population(population, data_travel, filtered_precedence_matrix_by_station, station_code)

        new_population = incorporate_elitism(population, fitness_scores, 1)


        while len(new_population) < len(population):
            # Selection
            parents = select_parents(population, fitness_scores, 2)

            # Crossover
            offspring1, offspring2 = pmx_crossover(parents[0], parents[1])

            offspring1 = reverse_sequence_mutation(offspring1,mutation_rate)
            offspring2 = reverse_sequence_mutation(offspring2, mutation_rate)

            offspring1 = scramble_stops_mutation(offspring1, mutation_rate)
            offspring2 = scramble_stops_mutation(offspring2, mutation_rate)

            offspring1, offspring2 = inter_zone_crossover(offspring1, offspring2)

            offspring1 = inversion_mutation(offspring1, mutation_rate)
            offspring2 = inversion_mutation(offspring2, mutation_rate)

            offspring1 = scramble_mutation(offspring1, mutation_rate)
            offspring2 = scramble_mutation(offspring2, mutation_rate)
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

    final_fitness_scores = evaluate_population(population, data_travel, filtered_precedence_matrix_by_station, station_code)
    best_index = final_fitness_scores.index(max(final_fitness_scores))
    best_solution = population[best_index]
    best_fitness = final_fitness_scores[best_index]
    #if counter ==6:
    #    print("test")
    # fitness_scores1 = evaluate_population(population, data_travel)
    return best_solution


def create_stop_sequence_mapping(best_solution):
    # Initialize an empty dictionary to hold the mapping of stop_ID to sequence number
    stop_sequence_mapping = {}

    # Assuming the station is always the first entry in the list under the 'nan' key
    # and that its stop_ID needs to be mapped to 0
    station_stop_id = best_solution[list(best_solution.keys())[0]][0].stop_ID  # Get the station's stop_ID
    stop_sequence_mapping[station_stop_id] = 0  # Station is always the first in the sequence

    # Sequence number, starting from 1 for the first stop after the station
    sequence_number = 1

    # Iterate over each zone in the best solution
    for zone_id in list(best_solution.keys())[1:]:  # Skip the first key, assuming it's for the station
        stops = best_solution[zone_id]
        for stop in stops:
            # Map each stop's ID to its sequence number
            stop_sequence_mapping[stop.stop_ID] = sequence_number
            sequence_number += 1  # Increment the sequence number for the next stop

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
        best_solution = Main(route_details, travel[route_id], population_size = 30, num_generations = 300, mutation_rate =0.4, tournament_size=3, time_limit=6.5, debug=counter)
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
