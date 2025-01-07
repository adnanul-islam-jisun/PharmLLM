# Project Setup and Usage

## Directory Structure
After cloning this repository, you need to set up the following directory structure:

```
PharmLLM/
├── dataset/
│   ├── raw/
│   │   ├── drug-label-0001-of-0012.json
│   │   ├── drug-label-0002-of-0012.json
│   │   └── ... (additional JSON files)
├── main.py
├── .gitignore
├── readme.md
```

### Steps to Set Up

1. **Clone the Repository**
   ```bash
   git clone <https://github.com/adnanul-islam-jisun/PharmLLM>
   cd <PharmLLM>
   ```

2. **Create the Directory Tree**
   Run the following commands to set up the required directories:
   ```bash
   mkdir -p dataset/raw
   ```

3. **Add JSON Files**
   Place your JSON files (e.g., `drug-label-0001-of-0012.json`, `drug-label-0002-of-0012.json`) inside the `dataset/raw/` directory.

4. **Verify the Structure**
   Use the `tree` command or manually check to ensure the directory structure matches:
   ```bash
   tree
   ```
   Output should look like:
   ```
   .
   ├── dataset/
   │   ├── raw/
   │   │   ├── drug-label-0001-of-0012.json
   │   │   ├── drug-label-0002-of-0012.json
   │   │   └── ...
   ├── main.py
   ├── .gitignore
   ```

## Running the Script

1. **Ensure You Have Python Installed**
   Make sure Python is installed on your machine (version 3.7 or higher).

2. **Run the Script**
   Execute the `main.py` script:
   ```bash
   python main.py
   ```

## Notes
- Ensure all required JSON files are present in the `dataset/raw/` directory before running the script.
- If there are any errors, please check that the directory structure and file paths match the setup instructions.

