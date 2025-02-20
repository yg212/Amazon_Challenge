import pandas as pd
import json, os, sys
import folium
from datetime import datetime
import hashlib

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

def filter_routes_by_score(routes, score):
    '''
    Filters routes by score and returns a dictionary of matching routes.
    '''
    return {route: info for route, info in routes.items() if info['route_score'] == score}

# Define base directory and file paths
base_dir = os.path.dirname(os.path.abspath(__file__))
file_paths = ['route_data.json', 'package_data.json', 'travel_times.json', 'actual_sequences.json']
full_paths = [os.path.join(base_dir, 'data/model_build_inputs', path) for path in file_paths]

# Read files
routes, packages, travel, actual_sequence = read_files(*full_paths)
def sort_stops_by_sequence(routes, actual_sequence):
    sorted_routes = {}
    for route_id, route_details in routes.items():
        # Extract the sequence information for the current route
        sequence_info = actual_sequence.get(route_id, {}).get('actual', {})

        # Sort the stops based on their sequence number
        sorted_stop_ids = sorted(sequence_info, key=sequence_info.get)

        # Reorder the stops according to the sorted sequence
        sorted_stops = {stop_id: route_details['stops'][stop_id] for stop_id in sorted_stop_ids if
                        stop_id in route_details['stops']}

        # Update the route details with the sorted stops
        sorted_routes[route_id] = {**route_details, 'stops': sorted_stops}

    return sorted_routes

sorted_routes = sort_stops_by_sequence(routes, actual_sequence)
sort_high_score_routes = filter_routes_by_score(sorted_routes, 'High')
sort_medium_score_routes = filter_routes_by_score(sorted_routes, 'Medium')
sort_low_score_routes = filter_routes_by_score(sorted_routes, 'Low')


import pandas as pd
import json, os, sys
import folium
from datetime import datetime


# Define base directory and file paths
base_dir = os.path.dirname(os.path.abspath(__file__))
file_paths = ['route_data.json', 'package_data.json', 'travel_times.json', 'actual_sequences.json']
full_paths = [os.path.join(base_dir, 'data/model_build_inputs', path) for path in file_paths]

# Read files
routes, packages, travel, actual_sequence = read_files(*full_paths)


def calculate_zone_transitions(routes, actual_sequence):
    zone_transitions_counts = {}

    for route_id, route_details in routes.items():
        # Retrieve the sequence mapping for the current route
        stop_sequence = actual_sequence[route_id]['actual']

        # Sort stops based on sequence value
        sorted_stops = sorted(stop_sequence.items(), key=lambda item: item[1])

        previous_zone_id = None
        zone_transition_count = 0

        # Updated logic to handle missing zone_id by assigning from nearest valid stop
        for i, (stop_id, _) in enumerate(sorted_stops):
            if stop_id in route_details['stops']:
                current_zone_id = route_details['stops'][stop_id].get('zone_id')

                # If zone_id is missing, attempt to assign it from the nearest valid stop
                if not current_zone_id:
                    # Look ahead to find the next valid zone_id
                    for ahead in sorted_stops[i+1:]:
                        next_zone_id = route_details['stops'][ahead[0]].get('zone_id')
                        if next_zone_id:
                            current_zone_id = next_zone_id
                            break
                    # If not found ahead, look backward
                    if not current_zone_id:
                        for behind in sorted_stops[:i][::-1]:
                            prev_zone_id = route_details['stops'][behind[0]].get('zone_id')
                            if prev_zone_id:
                                current_zone_id = prev_zone_id
                                break

                # Check for a zone transition
                if previous_zone_id is not None and current_zone_id != previous_zone_id:
                    zone_transition_count += 1

                previous_zone_id = current_zone_id

        # Store the zone transition count for this route
        zone_transitions_counts[route_id] = zone_transition_count

    return zone_transitions_counts


# Example usage
zone_transition_counts = calculate_zone_transitions(routes, actual_sequence)

def filter_routes_by_score(routes, score):
    '''
    Filters routes by score and returns a dictionary of matching routes.
    '''
    return {route: info for route, info in routes.items() if info['route_score'] == score}


def calculate_average_zone_transitions_by_score(routes, zone_transition_counts):
    """
    Calculates the average number of zone transitions for each route score.

    :param routes: Dictionary of routes with their details including route_score.
    :param zone_transition_counts: Dictionary of route IDs and their zone transition counts.
    :return: Dictionary of average zone transitions by route score.
    """
    # Initialize dictionaries to store total zone transitions and route counts by score
    total_zone_transitions_by_score = {}
    route_counts_by_score = {}

    for route_id, route_info in routes.items():
        route_score = route_info['route_score']
        # Only process if the route has a zone transition count
        if route_id in zone_transition_counts:
            # Initialize if this score has not been seen before
            if route_score not in total_zone_transitions_by_score:
                total_zone_transitions_by_score[route_score] = 0
                route_counts_by_score[route_score] = 0

            # Aggregate total zone transitions and increment route count
            total_zone_transitions_by_score[route_score] += zone_transition_counts[route_id]
            route_counts_by_score[route_score] += 1

    # Calculate the average zone transitions for each score
    average_zone_transitions_by_score = {
        score: total_zone_transitions_by_score[score] / route_counts_by_score[score]
        for score in total_zone_transitions_by_score
    }

    return average_zone_transitions_by_score


# Example usage
average_zone_transitions = calculate_average_zone_transitions_by_score(routes, zone_transition_counts)
print(average_zone_transitions)

def sort_stops_by_sequence(routes, actual_sequence):
    sorted_routes = {}
    for route_id, route_details in routes.items():
        # Extract the sequence information for the current route
        sequence_info = actual_sequence.get(route_id, {}).get('actual', {})

        # Sort the stops based on their sequence number
        sorted_stop_ids = sorted(sequence_info, key=sequence_info.get)

        # Reorder the stops according to the sorted sequence
        sorted_stops = {stop_id: route_details['stops'][stop_id] for stop_id in sorted_stop_ids if
                        stop_id in route_details['stops']}

        # Update the route details with the sorted stops
        sorted_routes[route_id] = {**route_details, 'stops': sorted_stops}

    return sorted_routes

sorted_routes = sort_stops_by_sequence(routes, actual_sequence)
sort_high_score_routes = filter_routes_by_score(sorted_routes, 'High')
sort_medium_score_routes = filter_routes_by_score(sorted_routes, 'Medium')
sort_low_score_routes = filter_routes_by_score(sorted_routes, 'Low')

def extract_stop_details(stop_details, package_details):
    # Initialize lists to hold unique package dimensions and time windows
    dimensions = []
    time_windows = []

    # Iterate over all packages to collect their dimensions and time windows
    for pkg in package_details.values():
        dimensions.append(pkg.get('dimensions', {}))
        time_window = {
            'start_time_utc': pkg.get('time_window', {}).get('start_time_utc', 'unknown'),
            'end_time_utc': pkg.get('time_window', {}).get('end_time_utc', 'unknown')
        }
        time_windows.append(time_window)

    # Extract service time from the first package assuming it's the same for all packages at this stop
    service_time = next(iter(package_details.values())).get('planned_service_time_seconds', 0) if package_details else 0

    # Combine stop details with aggregated package info
    combined_info = {
        **stop_details,
        'service_time': service_time,
        'package_count': len(package_details),
        'dimensions': dimensions,
        'time_windows': time_windows,
    }
    return combined_info

def format_time_windows(time_windows):
    """Formats all given time windows into a single string."""
    if not time_windows:
        return 'unknown to unknown'
    # Concatenate all time windows into a single string
    return ', '.join([f"{tw['start_time_utc']} to {tw['end_time_utc']}" for tw in time_windows])

# Dynamic color generation functions
def generate_base_color(zone_id):
    """Generates a base color based on the hash of the zone ID."""
    hash_value = int(hashlib.sha256(zone_id.encode()).hexdigest(), 16)
    r = (hash_value & 0xFF0000) >> 16
    g = (hash_value & 0x00FF00) >> 8
    b = hash_value & 0x0000FF
    return f'#{r:02x}{g:02x}{b:02x}'

def generate_shade(base_color, subzone):
    """Adjusts the brightness of the base color to generate a shade for subzones."""
    hash_value = int(hashlib.sha256(subzone.encode()).hexdigest(), 16) % 256
    adjustment = hash_value % 50 - 25
    r, g, b = int(base_color[1:3], 16), int(base_color[3:5], 16), int(base_color[5:], 16)
    r = max(0, min(255, r + adjustment))
    g = max(0, min(255, g + adjustment))
    b = max(0, min(255, b + adjustment))
    return f'#{r:02x}{g:02x}{b:02x}'


def adjust_color_shade(hex_color, adjustment=-20):
    """Adjust the color brightness. Positive adjustment = lighter, negative = darker."""
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:], 16)
    r = max(0, min(255, r + adjustment))
    g = max(0, min(255, g + adjustment))
    b = max(0, min(255, b + adjustment))
    return f'#{r:02x}{g:02x}{b:02x}'

def get_marker_size(subzone):
    """Determine marker size based on subzone. Implement your logic."""
    # Simplified example: larger size for primary subzone, smaller for others
    return 12
base_color_map = {}  # Cache for base colors

def get_color_for_zone_id(zone_id):
    # Handle cases where zone_id is None or not a string (e.g., float)
    if not isinstance(zone_id, str):
        return '#808080'  # Return a default color (e.g., grey) for invalid or missing zone IDs

    major_zone = zone_id[0]  # Assuming the major zone is indicated by the first character
    if major_zone not in base_color_map:
        base_color_map[major_zone] = generate_base_color(major_zone)
    base_color = base_color_map[major_zone]

    if '-' in zone_id:
        subzone = zone_id.split('-')[1]
        return generate_shade(base_color, subzone)
    else:
        return base_color
def generate_maps_for_routes(sorted_routes, packages, score_category, limit=20):
    """
    Generates and saves up to 'limit' maps for given sorted routes into categorized folders,
    displaying stop name, zone ID, package count, service time, and a single time window directly on the map.

    Parameters:
    - sorted_routes: Dictionary of sorted routes.
    - packages: Dictionary of package information keyed by route id and then by stop name.
    - score_category: String representing the score category (e.g., 'High', 'Medium', 'Low').
    - limit: Maximum number of maps to generate per category.
    """

    # Directory name based on score category
    maps_dir = f"maps_v2/{score_category}"
    if not os.path.exists(maps_dir):
        os.makedirs(maps_dir)

    # Predefined list of colors for zone IDs
    colors = ['blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue',
              'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray']
    zone_color_map = {}

    for index, (routeID, routeDetails) in enumerate(sorted_routes.items()):
        if index >= limit:
            break  # Stop generating maps after reaching the limit

        stops = routeDetails['stops']
        route_packages = packages.get(routeID, {})

        # Initialize a new map centered on the first stop of the current route
        m = folium.Map(location=[next(iter(stops.values()))['lat'], next(iter(stops.values()))['lng']], zoom_start=13)

        for sequence, (stopName, stopDetails) in enumerate(stops.items()):
            lat = stopDetails['lat']
            lng = stopDetails['lng']
            zone_id = stopDetails['zone_id']
            package_details = route_packages.get(stopName, {})

            hex_color = get_color_for_zone_id(zone_id)  # Get the base color for the zone

            # Adjust color shade based on subzone
            marker_size = 5
            if not isinstance(zone_id, str):
                pass
            elif '-' in zone_id:
                subzone = zone_id.split('-')[1]
                if subzone[-1].isdigit():  # Check if the last character is a digit
                    adjustment = int(subzone[-1]) * 10 - 30
                else:
                    adjustment = 0  # Fallback adjustment value if last character is not a digit
                hex_color = adjust_color_shade(hex_color, adjustment)
                marker_size = get_marker_size(subzone)
            # Assign a color to each zone ID if not already done
            #if zone_id not in zone_color_map:
                #zone_color_map[zone_id] = colors[len(zone_color_map) % len(colors)]

            # Use the previously defined extract_stop_details function
            combined_info = extract_stop_details(stopDetails, package_details)
            package_count = combined_info['package_count']
            service_time = combined_info['service_time']

            time_window_text = format_time_windows(combined_info['time_windows'])

            # Prepare tooltip text
            tooltip_text = (f"Sequence: {sequence}, Stop: {stopName}, Zone ID: {zone_id}, "
                            f"Packages: {package_count}, Service Time: {service_time}s, "
                            f"Time Window: {time_window_text}")

            # Marker with tooltip, using zone-based color
            folium.CircleMarker(
                location=[lat, lng],
                radius=marker_size,
                color=hex_color,
                fillColor=hex_color,
                fill=True,
                fillOpacity=0.7,
                tooltip=tooltip_text
            ).add_to(m)

        # Connect the stops with a polyline
        folium.PolyLine([(stop['lat'], stop['lng']) for stop in stops.values()], color="red", weight=2.5,
                        opacity=1).add_to(m)

        # Save the map with a simple filename based on routeID
        map_filename = os.path.join(maps_dir, f"map_{routeID}.html")
        m.save(map_filename)

# Example usage with your data and package information
generate_maps_for_routes(sort_high_score_routes, packages, 'High', 20)
generate_maps_for_routes(sort_medium_score_routes, packages, 'Medium', 20)
generate_maps_for_routes(sort_low_score_routes, packages, 'Low', 20)

print("Test")