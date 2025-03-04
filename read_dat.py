import pandas as pd
import requests

def get_column_specs(sas_url):
    # Download and parse SAS file
    response = requests.get(sas_url)
    sas_content = response.text
    
    # Extract column specifications
    col_specs = []
    col_names = []
    in_input = False
    
    for line in sas_content.split('\n'):
        if 'INPUT' in line.upper():
            in_input = True
            continue
        if in_input and ';' in line:
            break
        if in_input and line.strip():
            # Parse variable name and positions (e.g., "SEQN      1 -   5")
            parts = line.split()
            if len(parts) >= 3 and parts[1].isdigit():
                name = parts[0]
                start = int(parts[1]) - 1  # Convert to 0-based index
                end = int(parts[-1])
                col_specs.append((start, end))
                col_names.append(name)
    
    return col_specs, col_names

def read_to_dataframe(dat_url, sas_url):
    # Get column specifications from SAS file
    col_specs, col_names = get_column_specs(sas_url)
    
    # Read the fixed-width DAT file into DataFrame
    df = pd.read_fwf(
        dat_url,
        colspecs=col_specs,
        names=col_names,
        encoding='latin1'
    )
    
    return df

# URLs
sas_url = "https://wwwn.cdc.gov/nchs/data/nhanes3/1a/adult.sas"
dat_url = "https://wwwn.cdc.gov/nchs/data/nhanes3/1a/adult.dat"

# Read into DataFrame
try:
    df = read_to_dataframe(dat_url, sas_url)
    
    # Display basic information
    print("DataFrame Info:")
    print(df.info())
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Optional: Save to CSV
    df.to_csv('adult_data.csv', index=False)
    print("\nSaved to adult_data.csv")
    
except Exception as e:
    print(f"Error: {str(e)}")
