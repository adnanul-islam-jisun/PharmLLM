import json

# Function to preprocess and filter data to keep specific keys
def extract_relevant_data(input_file, output_file):
    try:
        # Load the input JSON file
        with open(input_file, 'r') as file:
            data = json.load(file)

        # Keys to retain (including nested "openfda" keys)
        main_keys_to_keep = {"description", "adverse_reactions", "overdosage"}
        openfda_keys_to_keep = {"brand_name", "generic_name", "route"}

        # Ensure the structure contains 'results'
        if "results" not in data or not isinstance(data["results"], list):
            raise ValueError("Invalid JSON format: 'results' key is missing or not a list.")

        # Process the list of objects in 'results'
        filtered_results = []
        for item in data["results"]:
            if isinstance(item, dict):  # Ensure each element is a dictionary
                # Extract top-level keys
                filtered_item = {key: value for key, value in item.items() if key in main_keys_to_keep}

                # Extract nested 'openfda' keys if present
                openfda_data = item.get("openfda", {})
                if isinstance(openfda_data, dict):
                    filtered_item.update({key: value for key, value in openfda_data.items() if key in openfda_keys_to_keep})

                if filtered_item:
                    filtered_results.append(filtered_item)

        # Save the filtered data to a new JSON file
        with open(output_file, 'w') as file:
            json.dump(filtered_results, file, indent=4)

        print(f"Filtered data saved to: {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Input and output file paths
input_json_file = "G:\dowload\drugbank_all_full_database.xml\drug-label-0001-of-0012.json\drug-label-0001-of-0012.json"
output_json_file = "filtered_adverse.json"

# Call the function
extract_relevant_data(input_json_file, output_json_file)
