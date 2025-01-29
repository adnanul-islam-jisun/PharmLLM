import kagglehub

# Download latest version
path = kagglehub.dataset_download("subheysadi/adr-dataset-v1")

print("Path to dataset files:", path)