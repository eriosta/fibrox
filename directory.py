import os
from pathlib import Path

# Define the root directory as the current directory
root_dir = "."  # Use current directory (fibrox/)

# Define the directory structure as a dictionary
structure = {
    "data": {
        "raw": {
            "nhanes_2017_2020": [
                "demographic.csv",
                "dietary.csv",
                "examination.csv"
            ],
            "yale_cohort": [
                "yale_data.csv"
            ],
            "nhanes_iii": [
                "adult.dat",
                "adult.sas"
            ]
        },
        "processed": [
            "nhanes_2017_2020_train.csv",
            "yale_validation.csv",
            "nhanes_iii_prognostication.csv"
        ]
    },
    "src": [
        "data_preprocessing.py",
        "model_training.py",
        "model_validation.py",
        "prognostication.py",
        "utils.py"
    ],
    "notebooks": [
        "exploratory_analysis.ipynb",
        "model_evaluation.ipynb"
    ],
    "outputs": {
        "models": [
            "xgboost_model.pkl"
        ],
        "results": [
            "training_metrics.csv",
            "validation_metrics.csv",
            "prognostication_metrics.csv",
            "feature_importance.png"
        ]
    },
    "config": [
        "config.yaml"
    ],
    "": [  # Root-level files in the current directory (fibrox/)
        "requirements.txt",
        "README.md"
    ]
}

def create_structure(base_path, structure):
    """
    Recursively create directories and files based on the structure dictionary.
    Only creates files if they don't already exist.
    """
    for key, value in structure.items():
        # If key is empty, we're at the root level (current directory)
        current_path = os.path.join(base_path, key) if key else base_path
        
        # Create directory if it's not the root-level files
        if key:
            Path(current_path).mkdir(parents=True, exist_ok=True)
        
        # If value is a list, create files only if they don't exist
        if isinstance(value, list):
            for item in value:
                file_path = os.path.join(current_path, item)
                # Create an empty file only if it doesn't already exist
                if not Path(file_path).exists():
                    Path(file_path).touch()
        # If value is a dict, recurse deeper
        elif isinstance(value, dict):
            create_structure(current_path, value)

# Create the structure in the current directory
if __name__ == "__main__":
    # Create the structure directly in the current directory
    create_structure(root_dir, structure)
    
    print(f"Directory structure created successfully in the current directory (fibrox/), skipping existing files")
