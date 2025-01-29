import json

# Load the raw JSON data
input_file_path = "dataset/filtered_ddi.json"
output_file_path = "dataset/drug_interaction.json"

with open(input_file_path, "r") as file:
    raw_data = json.load(file)

structured_data = []

for entry in raw_data:
    # Prepare input_text by including only non-empty fields
    input_text_parts = []

    if entry["warnings"] and entry["warnings"] != "None":
        input_text_parts.append(f"Warnings: {'; '.join(entry['warnings'])}")

    if entry["precautions"] and entry["precautions"] != "None":
        input_text_parts.append(f"Precautions: {entry['precautions']}")

    if entry["description"] and entry["description"] != "None":
        input_text_parts.append(f"Description: {entry['description']}")

    if entry["route"] and entry["route"] != "None":
        input_text_parts.append(f"Route: {entry['route']}")

    if entry["brand_name"] and entry["brand_name"] != "None":
        input_text_parts.append(f"Brand Name: {entry['brand_name']}")

    if entry["generic_name"] and entry["generic_name"] != "None":
        input_text_parts.append(f"Generic Name: {entry['generic_name']}")

    # Combine input text parts into a single instruction-style prompt
    input_text = " | ".join(input_text_parts)

    # Ensure input text is meaningful
    if not input_text:
        input_text = "Provide drug interaction details."

    # Create response_text from drug_interactions or use default text
    if entry["drug_interactions"] and entry["drug_interactions"] != "None":
        response_text = " | ".join(entry["drug_interactions"])
    else:
        response_text = "No drug interaction"

    # Format data for instruction tuning
    structured_data.append({"input_text": input_text, "response_text": response_text})


for entry in structured_data:
    if entry["response_text"].startswith("7 DRUG INTERACTIONS"):
        entry["response_text"] = entry["response_text"].replace(
            "7 DRUG INTERACTIONS", "DRUG INTERACTIONS", 1
        )

# Save the fine-tuning dataset
with open(output_file_path, "w") as json_file:
    json.dump(structured_data, json_file, indent=4)

print(f"Fine-tuning dataset saved to {output_file_path}")
