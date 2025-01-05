import json
import os

raw_data = "dataset/raw/"


def extract_adverse_data(input_file, output_file):
    try:
        with open(input_file, "r") as file:
            data = json.load(file)

        main_keys_to_keep = {"description", "adverse_reactions", "overdosage"}
        openfda_keys_to_keep = {"brand_name", "generic_name", "route"}

        if "results" not in data or not isinstance(data["results"], list):
            raise ValueError(
                "Invalid JSON format: 'results' key is missing or not a list."
            )

        filtered_results = []
        for item in data["results"]:
            if isinstance(item, dict):
                filtered_item = {
                    key: value
                    for key, value in item.items()
                    if key in main_keys_to_keep
                }

                openfda_data = item.get("openfda", {})
                if isinstance(openfda_data, dict):
                    filtered_item.update(
                        {
                            key: value
                            for key, value in openfda_data.items()
                            if key in openfda_keys_to_keep
                        }
                    )

                if filtered_item:
                    filtered_results.append(filtered_item)

        with open(f"dataset/{output_file}", "w") as file:
            json.dump(filtered_results, file, indent=4)

        print(f"Filtered data saved to: dataset/{output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")


def extract_ddi_data(input_file, output_file):
    try:
        with open(input_file, "r") as file:
            data = json.load(file)

        main_keys_to_keep = {
            "drug_interactions",
            "warnings",
            "precautions",
            "description",
        }
        openfda_keys_to_keep = {"brand_name", "generic_name", "route"}

        if "results" not in data or not isinstance(data["results"], list):
            raise ValueError(
                "Invalid JSON format: 'results' key is missing or not a list."
            )

        filtered_results = []
        for item in data["results"]:
            if isinstance(item, dict):
                filtered_item = {
                    key: value
                    for key, value in item.items()
                    if key in main_keys_to_keep
                }

                openfda_data = item.get("openfda", {})
                if isinstance(openfda_data, dict):
                    filtered_item.update(
                        {
                            key: value
                            for key, value in openfda_data.items()
                            if key in openfda_keys_to_keep
                        }
                    )

                if filtered_item:
                    filtered_results.append(filtered_item)

        with open(f"dataset/{output_file}", "w") as file:
            json.dump(filtered_results, file, indent=4)

        print(f"Filtered data saved to: {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")


def create_adverse_data():
    i = 1
    current_directory = os.getcwd()

    for file in os.listdir(os.path.join(raw_data, current_directory)):
        if file.endswith("of-0012.json"):
            extract_adverse_data(file, f"filtered_adverse_{i}.json")
            i += 1


def create_ddi_data():
    i = 1
    current_directory = os.getcwd()

    for file in os.listdir(os.path.join(raw_data, current_directory)):
        if file.endswith("of-0012.json"):
            extract_ddi_data(file, f"filtered_ddi_{i}.json")
            i += 1


def merge_file(files_name):
    try:
        merged_data = []
        for file_name in os.listdir("dataset"):
            if file_name.startswith(files_name):
                with open(file_name, "r") as file:
                    data = json.load(file)
                    merged_data.extend(data)

                os.remove(file_name)
                print(f"Deleted file: {file_name}")
        output_file = f"{files_name}.json"
        with open(output_file, "w") as file:
            json.dump(merged_data, file, indent=4)

        print(f"Merged data saved to: {files_name}.json")

    except Exception as e:
        print(f"An error occurred: {e}")


"""
    This extract the data for adverse then merge all 
    adverse data file into single file
"""
# create_adverse_data()
# merge_file("filtered_adverse")


"""
    This extract the data for drug-drug-interaction then merge all 
    drug-drug-interaction data file into single file
"""
# create_ddi_data()
# merge_file("filtered_ddi")
