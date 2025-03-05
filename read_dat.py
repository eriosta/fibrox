import pandas as pd
import requests
from io import StringIO
import re

def dedup_names(names):
    seen = {}
    new_names = []
    for name in names:
        if name in seen:
            seen[name] += 1
            new_names.append(f"{name}.{seen[name]}")
        else:
            seen[name] = 0
            new_names.append(name)
    return new_names

def get_column_specs_and_labels(sas_url):
    response = requests.get(sas_url)
    if response.status_code != 200:
        raise Exception(f"Failed to download SAS file: {response.status_code}")
    sas_content = response.text

    col_specs = []
    col_labels = []
    in_input = False
    label_dict = {}

    # Extract labels from the SAS file
    label_section = False
    for line in sas_content.split('\n'):
        line = line.strip()
        if 'LABEL' in line.upper():
            label_section = True
            continue
        if label_section and ';' in line:
            label_section = False
        if label_section and line:
            # Accept both single and double quotes.
            match = re.match(r"(\w+)\s*=\s*[\"'](.+?)[\"']", line)
            if match:
                var_name, label = match.groups()
                label_dict[var_name] = label

    # Extract column specs from the SAS file
    for line in sas_content.split('\n'):
        line = line.strip()
        if 'INPUT' in line.upper():
            in_input = True
            continue
        if in_input and ';' in line:
            break
        if in_input and line:
            # Parse lines like "SEQN     1-5" or "DMPSTAT  11"
            parts = line.split()
            if len(parts) >= 2:
                name = parts[0]
                pos_spec = parts[-1]
                if '-' in pos_spec:
                    start_str, end_str = pos_spec.split('-')
                    if start_str.isdigit() and end_str.isdigit():
                        start = int(start_str) - 1  # Convert to 0-based index
                        end = int(end_str)
                        col_specs.append((start, end))
                        col_labels.append(label_dict.get(name, name))
                elif pos_spec.isdigit():
                    pos = int(pos_spec)
                    col_specs.append((pos - 1, pos))
                    col_labels.append(label_dict.get(name, name))

    # Handle duplicate column names using our custom function
    col_labels = dedup_names(col_labels)
    
    return col_specs, col_labels

def read_to_dataframe(dat_url, sas_url, output_filename):
    # Download the DAT file content
    response = requests.get(dat_url)
    if response.status_code != 200:
        raise Exception(f"Failed to download DAT file: {response.status_code}")
    dat_content = response.text

    # Get column specifications and labels
    col_specs, col_labels = get_column_specs_and_labels(sas_url)

    if not col_specs:
        raise ValueError("No column specifications parsed from SAS file")

    print(f"\nProcessing {output_filename}")
    print(f"Parsed {len(col_labels)} columns from SAS file")
    print(f"First few columns: {col_labels[:5]}")
    print(f"First few specs: {col_specs[:5]}")

    # Convert content to file-like object for pandas
    dat_file = StringIO(dat_content)

    # Read into DataFrame using fixed-width formatting
    try:
        df = pd.read_fwf(
            dat_file,
            colspecs=col_specs,
            names=col_labels,
            encoding='latin1'
        )

        # Ensure all columns are included, even if they contain only missing data
        for col in col_labels:
            if col not in df.columns:
                df[col] = pd.NA

        return df
    except Exception as e:
        print(f"Error reading fixed-width file: {str(e)}")
        print(f"Column specs: {col_specs[:5]}")  # Print first few specs for debugging
        raise

# Dictionary of datasets with their URLs and output filenames
datasets = {
    'adult': {
        'sas_url': "https://wwwn.cdc.gov/nchs/data/nhanes3/1a/adult.sas",
        'dat_url': "https://wwwn.cdc.gov/nchs/data/nhanes3/1a/adult.dat",
        'output': 'adult_data.csv'
    },
    'lab': {
        'sas_url': "https://wwwn.cdc.gov/nchs/data/nhanes3/1a/lab.sas",
        'dat_url': "https://wwwn.cdc.gov/nchs/data/nhanes3/1a/lab.dat",
        'output': 'lab_data.csv'
    },
    'exam': {
        'sas_url': "https://wwwn.cdc.gov/nchs/data/nhanes3/1a/exam.sas",
        'dat_url': "https://wwwn.cdc.gov/nchs/data/nhanes3/1a/exam.dat",
        'output': 'exam_data.csv'
    }
}

# Process all datasets
for dataset_name, info in datasets.items():
    try:
        # Ensure the correct SAS file is used for the corresponding DAT file
        if info['sas_url'].split('/')[-1].replace('.sas', '') != info['dat_url'].split('/')[-1].replace('.dat', ''):
            raise ValueError(f"SAS and DAT file mismatch for {dataset_name}")

        # Read into DataFrame
        df = read_to_dataframe(info['dat_url'], info['sas_url'], dataset_name)

        # Display results
        print(f"\n{dataset_name.capitalize()} DataFrame Info:")
        print(df.info())
        print(f"\nFirst 5 rows of {dataset_name}:")
        print(df.head())

        # Save to CSV
        df.to_csv(info['output'], index=False)
        print(f"\nSaved to {info['output']}")

    except Exception as e:
        print(f"Error processing {dataset_name}: {str(e)}")
