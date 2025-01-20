import json

# File paths
input_file_path = "C:\\Users\\adnan\\Desktop\\filtered_adverse.json"  # Replace with your raw data JSON file path
output_file_path = "adverse_reactions_dataset.json"

# Read raw data from JSON file
with open(input_file_path, "r") as file:
    raw_data = json.load(file)

# Create structured dataset
structured_data = []

for entry in raw_data:
    input_text = (
        f"Drug: {entry['spl_product_data_elements'][0]}; "
        f"Route: {entry['route'][0]}; "
        f"Description: {entry['description'][0]}"
    )
    response_text = entry['adverse_reactions'][0]
    
    structured_data.append({
        "input_text": input_text,
        "response_text": response_text
    })

# Save structured dataset to a JSON file
with open(output_file_path, "w") as json_file:
    json.dump(structured_data, json_file, indent=4)

print(f"Structured data saved to {output_file_path}")
