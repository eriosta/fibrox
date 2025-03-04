import pandas as pd
import requests
from io import StringIO

def get_column_specs(sas_url):
    response = requests.get(sas_url)
    if response.status_code != 200:
        raise Exception(f"Failed to download SAS file: {response.status_code}")
    sas_content = response.text
    
    col_specs = []
    col_names = []
    in_input = False
    
    for line in sas_content.split('\n'):
        line = line.strip()
        if 'INPUT' in line.upper():
            in_input = True
            continue
        if in_input and ';' in line:
            break
        if in_input and line:
            # Parse lines like "SEQN     1-5"
            parts = line.split()
            if len(parts) >= 2 and '-' in parts[-1]:
                name = parts[0]
                start_end = parts[-1].split('-')
                if len(start_end) == 2 and start_end[0].isdigit() and start_end[1].isdigit():
                    start = int(start_end[0]) - 1  # Convert to 0-based index
                    end = int(start_end[1])
                    col_specs.append((start, end))
                    col_names.append(name)
    
    return col_specs, col_names

def read_to_dataframe(dat_url, sas_url):
    # Download the DAT file content
    response = requests.get(dat_url)
    if response.status_code != 200:
        raise Exception(f"Failed to download DAT file: {response.status_code}")
    dat_content = response.text
    
    # Get column specifications
    col_specs, col_names = get_column_specs(sas_url)
    
    if not col_specs:
        raise ValueError("No column specifications parsed from SAS file")
    
    print(f"Parsed {len(col_names)} columns from SAS file")
    print(f"First few columns: {col_names[:5]}")
    print(f"First few specs: {col_specs[:5]}")
    
    # Convert content to file-like object for pandas
    dat_file = StringIO(dat_content)
    
    # Read into DataFrame
    try:
        df = pd.read_fwf(
            dat_file,
            colspecs=col_specs,
            names=col_names,
            encoding='latin1'
        )
        return df
    except Exception as e:
        print(f"Error reading fixed-width file: {str(e)}")
        print(f"Column specs: {col_specs[:5]}")  # Print first few specs for debugging
        raise

# URLs
sas_url = "https://wwwn.cdc.gov/nchs/data/nhanes3/1a/adult.sas"
dat_url = "https://wwwn.cdc.gov/nchs/data/nhanes3/1a/adult.dat"

try:
    # Read into DataFrame
    df = read_to_dataframe(dat_url, sas_url)
    
    # Display results
    print("\nDataFrame Info:")
    print(df.info())
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Save to CSV
    df.to_csv('adult_data.csv', index=False)
    print("\nSaved to adult_data.csv")
    
except Exception as e:
    print(f"Error: {str(e)}")
