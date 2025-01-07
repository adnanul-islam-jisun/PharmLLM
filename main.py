import json
import os

raw_data = "dataset/raw/"


import json
import os

import json
import os

def extract_adverse_data(input_file, output_file):
    try:
        # Ensure the dataset directory exists
        output_dir = "dataset"
        os.makedirs(output_dir, exist_ok=True)

        # Load JSON data
        with open(input_file, "r") as file:
            data = json.load(file)

        # Validate JSON structure
        if "results" not in data or not isinstance(data["results"], list):
            raise ValueError(
                "Invalid JSON format: 'results' key is missing or not a list."
            )

        main_keys_to_keep = {"description", "adverse_reactions", "overdosage", "spl_product_data_elements"}
        openfda_keys_to_keep = {"brand_name", "generic_name", "route"}

        # Filter the data
        filtered_results = []
        for item in data["results"]:
            if isinstance(item, dict):
                
                # Ensure all main keys are included, even if missing
                filtered_item = {
                    key: item.get(key, "None") if item.get(key) is not None else "None"
                    for key in main_keys_to_keep
                }
                filtered_item["file"] = input_file
                # Process `openfda` data
                openfda_data = item.get("openfda", {})
                if isinstance(openfda_data, dict):
                    filtered_item.update(
                        {
                            key: openfda_data.get(key, "None")
                            if openfda_data.get(key) is not None
                            else "None"
                            for key in openfda_keys_to_keep
                        }
                    )
                else:
                    # Add `"None"` for openfda keys if `openfda` is missing or invalid
                    filtered_item.update(
                        {key: "None" for key in openfda_keys_to_keep}
                    )

                filtered_results.append(filtered_item)

        # Save filtered data
        output_path = os.path.join(output_dir, output_file)
        with open(output_path, "w") as file:
            json.dump(filtered_results, file, indent=4)

        print(f"Filtered data saved to: {output_path}")

    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON: {e}")
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except ValueError as e:
        print(f"Validation error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def extract_ddi_data(input_file, output_file):
    try:
        # Ensure the dataset directory exists
        output_dir = "dataset"
        os.makedirs(output_dir, exist_ok=True)

        # Load JSON data
        with open(input_file, "r") as file:
            data = json.load(file)

        # Validate JSON structure
        if "results" not in data or not isinstance(data["results"], list):
            raise ValueError(
                "Invalid JSON format: 'results' key is missing or not a list."
            )

        # Define keys to keep
        main_keys_to_keep = {
            "drug_interactions",
            "warnings",
            "precautions",
            "description",
        }
        openfda_keys_to_keep = {"brand_name", "generic_name", "route"}

        # Filter the data
        filtered_results = []
        for item in data["results"]:
            if isinstance(item, dict):
                # Ensure all main keys are included, even if missing
                filtered_item = {
                    key: item.get(key, "None") if item.get(key) is not None else "None"
                    for key in main_keys_to_keep
                }

                # Process `openfda` data
                openfda_data = item.get("openfda", {})
                if isinstance(openfda_data, dict):
                    filtered_item.update(
                        {
                            key: openfda_data.get(key, "None")
                            if openfda_data.get(key) is not None
                            else "None"
                            for key in openfda_keys_to_keep
                        }
                    )
                else:
                    # Add `"None"` for openfda keys if `openfda` is missing or invalid
                    filtered_item.update(
                        {key: "None" for key in openfda_keys_to_keep}
                    )

                # Skip items where all values are "None"
                if not all(value == "None" for value in filtered_item.values()):
                    filtered_results.append(filtered_item)

        # Save filtered data
        output_path = os.path.join(output_dir, output_file)
        with open(output_path, "w") as file:
            json.dump(filtered_results, file, indent=4)

        print(f"Filtered data saved to: {output_path}")

    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON: {e}")
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except ValueError as e:
        print(f"Validation error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def create_adverse_data():
    i = 1

    for file in os.listdir(raw_data):
        if file.endswith("of-0012.json"):
            extract_adverse_data(os.path.join(raw_data, file), f"filtered_adverse_{i}.json")
            i += 1


def create_ddi_data():
    i = 1

    for file in os.listdir(raw_data):
        if file.endswith("of-0012.json"):
            extract_ddi_data(os.path.join(raw_data, file), f"filtered_ddi_{i}.json")
            i += 1


def merge_file(files_name):
    try:
        merged_data = []
        for file_name in os.listdir("dataset"):
            file_path = os.path.join("dataset", file_name)

            if file_name.startswith(files_name):
                with open(file_path, "r") as file:
                    data = json.load(file)
                    merged_data.extend(data)

                os.remove(file_path)
                print(f"Deleted file: dataset/{file_name}")

        output_file = f"dataset/{files_name}.json"
        with open(output_file, "w") as file:
            json.dump(merged_data, file, indent=4)

        print(f"Merged data saved to: {files_name}.json")

    except Exception as e:
        print(f"An error occurred: {e}")


"""
    This extract the data for adverse then merge all 
    adverse data file into single file
"""
# create_adverse_data() # Extract data
# merge_file("filtered_adverse") # Merge data


"""
    This extract the data for drug-drug-interaction then merge all 
    drug-drug-interaction data file into single file
"""
# create_ddi_data() # Extract data
# merge_file("filtered_ddi") # Merge data
