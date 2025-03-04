import pandas as pd
import requests
from io import StringIO

def get_column_specs(sas_url):
    response = requests.get(sas_url)
    sas_content = response.text
    
    col_specs = []
    col_names = []
    in_input = False
    
    for line in sas_content.split('\n'):
        line = line.strip()
        if '@' in line and in_input:  # Look for @ position specifications
            # Example line: @1 SEQN 5. @6 HSSEX 1.
            parts = line.split()
            if len(parts) >= 3 and parts[0].startswith('@'):
                try:
                    start = int(parts[0][1:]) - 1  # Remove @ and convert to 0-based
                    name = parts[1]
                    width = int(float(parts[2].rstrip('.')))  # Handle formats like "5."
                    col_specs.append((start, start + width))
                    col_names.append(name)
                except (ValueError, IndexError):
                    continue
        elif 'INPUT' in line.upper():
            in_input = True
        elif in_input and ';' in line:
            break
    
    return col_specs, col_names

def read_to_dataframe(dat_url, sas_url):
    # Download the DAT file content
    response = requests.get(dat_url)
    if response.status_code != 200:
        raise Exception("Failed to download DAT file")
    dat_content = response.text
    
    # Get column specifications
    col_specs, col_names = get_column_specs(sas_url)
    
    if not col_specs:
        raise ValueError("No column specifications parsed from SAS file")
    
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
    print("DataFrame Info:")
    print(df.info())
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Save to CSV
    df.to_csv('adult_data.csv', index=False)
    print("\nSaved to adult_data.csv")
    
except Exception as e:
    print(f"Error: {str(e)}")
