# import requests
# import pandas as pd
# import os

# class NHANES3Data:
#     def __init__(self, file_ext, data_files):
#         self.urls = [f"https://wwwn.cdc.gov/nchs/data/nhanes3/{ext}/{file}" for ext in file_ext for file in data_files]
#         self.data_files = data_files
#         self.out = pd.DataFrame(columns=['SEQN'])

#     def download_files(self):
#         for url in self.urls:
#             try:
#                 response = requests.get(url)
#                 if response.status_code == 200:
#                     filename = url.split('/')[-1]
#                     with open(filename, 'wb') as f:
#                         f.write(response.content)
#                     print(f"Downloaded: {filename}")
#                 else:
#                     print(f"Failed to download: {url}")
#             except Exception as e:
#                 print(f"Error downloading {url}: {str(e)}")

#     def merge_files(self):
#         for file in self.data_files:
#             tmp = pd.read_sas(file)
#             self.out = pd.merge(self.out, tmp, on='SEQN', how='outer')

#     def clean_directory(self):
#         for file in os.listdir():
#             if file.endswith(".xpt"):
#                 os.remove(file)

#     def get_nhanes3(self):
#         print("Downloading SAS files...")
#         self.download_files()
#         print("Merging files...")
#         self.merge_files()
#         print("Cleaning directory...")
#         self.clean_directory()
#         return self.out

# # Call function to get data
# dat = NHANES3Data(file_ext=['34a', '34B'], data_files=['HGUHS.xpt', 'HGUHSSE.xpt']).get_nhanes3()

# # Select columns of interest
# dat = dat[['SEQN', 'GUPHSPF', 'GURHSPF']]

# # Update "HS primary finding" based on the given instructions
# dat['HS primary finding'] = dat.apply(
#     lambda row: row['GURHSPF'] if pd.notna(row['GURHSPF']) else row['GUPHSPF'],
#     axis=1
# )

# dat['HS primary finding'].value_counts()

# # Load other datasets
# labs = pd.read_csv('data/lab_data.csv')
# exam = pd.read_csv('data/exam_data.csv')
# adult = pd.read_csv('data/adult_data.csv')

# adult = adult[adult['Age at interview (screener) - qty'] >= 18]

# labs = labs[(labs['Serum hepatitis B core antibody'] == 2) | (labs['Serum hepatitis B core antibody'] == 8) | (labs['Serum hepatitis B core antibody'].isna())]

# exam = exam[exam['Age at interview (Screener)'] >= 18]

# adult = adult[(adult['Have you consumed alcohol in last 30 min'] == 2) | (adult['Have you consumed alcohol in last 30 min'] == 8) | (adult['Have you consumed alcohol in last 30 min'].isna())]

# # Rename the first column in adult, labs, and exam to SEQN
# adult.rename(columns={adult.columns[0]: 'SEQN'}, inplace=True)
# labs.rename(columns={labs.columns[0]: 'SEQN'}, inplace=True)
# exam.rename(columns={exam.columns[0]: 'SEQN'}, inplace=True)

# # Remove duplicate columns from labs and exam that are already in adult
# labs = labs[[col for col in labs.columns if col not in adult.columns or col == 'SEQN']]
# exam = exam[[col for col in exam.columns if col not in adult.columns or col == 'SEQN']]

# # Inner join datasets together by SEQN 
# dat = dat.merge(adult, on='SEQN', how='inner')
# dat = dat.merge(labs, on='SEQN', how='left')
# dat = dat.merge(exam, on='SEQN', how='left')

# # Filter based on "HS primary finding" and "GUPHSPF"
# dat = dat[(dat['HS primary finding'].isin([2, 3, 4]))]

# dat.to_csv('nhanes3.csv',index=False)

import pandas as pd
import numpy as np

def is_masld(
    Sex, 
    'Body mass index', 
    'Waist circumference (cm) (2+ years)', 
    'Plasma glucose (mg/dL)', 
    'Glycated hemoglobin: (%)', 
    'Ever been told you have sugar/diabetes',
    'Are you now taking insulin', 
    'Are you now taking diabetes pills', 
    'Overall average K1, systolic, BP(age 5+)', 
    'Overall average K5, diastolic, BP(age5+)', 
    'Now taking prescribed medicine for HBP', 
    'Serum triglycerides (mg/dL)', 
    'Take prescribed med to lower cholesterol', 
    'Serum HDL cholesterol (mg/dL)'
):
    """
    Determines if a patient meets MASLD criteria based on NHANES III variables.

    Parameters (NHANES III):
        Sex: Sex (1 = Male, 2 = Female)
        'Body mass index': Body-mass index
        'Waist circumference (cm) (2+ years)': Waist circumference (cm)
        'Plasma glucose (mg/dL)': Plasma glucose (mg/dL)
        'Glycated hemoglobin: (%)': Glycated hemoglobin (HbA1c)
        'Ever been told you have sugar/diabetes': History of diabetes (1 = Yes, 2 = No)
        'Are you now taking insulin': Currently taking insulin (1 = Yes, 2 = No)
        'Are you now taking diabetes pills': Currently taking diabetes pills (1 = Yes, 2 = No)
        'Overall average K1, systolic, BP(age 5+)': Systolic blood pressure
        'Overall average K5, diastolic, BP(age5+)': Diastolic blood pressure
        'Now taking prescribed medicine for HBP': Taking medication for high blood pressure (1 = Yes, 2 = No)
        'Serum triglycerides (mg/dL)': Serum triglycerides (mg/dL)
        'Take prescribed med to lower cholesterol': Taking cholesterol-lowering medication (1 = Yes, 2 = No)
        'Serum HDL cholesterol (mg/dL)': Serum HDL cholesterol (mg/dL)

    Returns:
        dict: Dictionary with MASLD criteria results and individual criteria flags.
    """

    # Criterion 1, body: BMI or waist circumference
    bmi_criteria = 'Body mass index' >= 25
    wc_criteria = 'Waist circumference (cm) (2+ years)' > (94 if Sex == 1 else 80)
    is_body = int(bmi_criteria or wc_criteria)

    # Criterion 2, diabetes: Blood glucose or diabetes history/treatment
    is_diabetes = int(
        'Plasma glucose (mg/dL)' >= 100 or
        'Glycated hemoglobin: (%)' >= 5.7 or
        'Ever been told you have sugar/diabetes' == 1 or
        'Are you now taking insulin' == 1 or
        'Are you now taking diabetes pills' == 1
    )

    # Criterion 3, hypertension: Blood pressure or antihypertensive treatment
    is_hypertension = int(
        'Overall average K1, systolic, BP(age 5+)' >= 130 or
        'Overall average K5, diastolic, BP(age5+)' >= 85 or
        'Now taking prescribed medicine for HBP' == 1
    )

    # Criterion 4 and 5, dyslipidemia: Triglycerides, HDL cholesterol, or lipid-lowering treatment
    is_dyslipidemia = int(
        'Serum triglycerides (mg/dL)' >= 150 or
        'Take prescribed med to lower cholesterol' == 1 or
        ('Serum HDL cholesterol (mg/dL)' <= 40 if Sex == 1 else 'Serum HDL cholesterol (mg/dL)' <= 50) or
        'Take prescribed med to lower cholesterol' == 1
    )

    # MASLD criteria met if at least one of the criteria is true
    is_masld = int(any([is_body, is_diabetes, is_hypertension, is_dyslipidemia]))

    return {
        'is_masld': is_masld,
        'is_body': is_body,
        'is_diabetes': is_diabetes,
        'is_hypertension': is_hypertension,
        'is_dyslipidemia': is_dyslipidemia
    }


def replace_blank_and_dont_know_with_na(data):
    """
    Replaces 'Blank but applicable' and 'Don't know' values with NA (None) in the provided NHANES III variables.

    Parameters:
        data (dict): Dictionary with NHANES III variable names and values.

    Returns:
        dict: Dictionary with 'Blank but applicable' and 'Don't know' values replaced with NA (None).
    """
    # Define the values for 'Blank but applicable' and 'Don't know' by variable
    blank_values = {
        'Body mass index': 8888,
        'Waist circumference (cm) (2+ years)': 88888,
        'Plasma glucose (mg/dL)': 88888,
        'Glycated hemoglobin: (%)': 8888,
        'Ever been told you have sugar/diabetes': [8, 9],
        'Are you now taking insulin': 8,
        'Are you now taking diabetes pills': [8, 9],
        'Overall average K1, systolic, BP(age 5+)': 888,
        'Overall average K5, diastolic, BP(age5+)': 888,
        'Now taking prescribed medicine for HBP': 8,
        'Serum triglycerides (mg/dL)': 8888,
        'Take prescribed med to lower cholesterol': [8, 9],
        'Serum HDL cholesterol (mg/dL)': 888,
        'Do you smoke cigarettes now': 8,
        'Have you smoked 100+ cigarettes in life': 8,
        'Serum cholesterol (mg/dL)': 888
    }

    # Replace each 'Blank but applicable' or 'Don't know' value with None (NA) if it matches the criteria
    for key, blank_value in blank_values.items():
        if isinstance(blank_value, list):
            if data.get(key) in blank_value:
                data[key] = np.nan
        else:
            if data.get(key) == blank_value:
                data[key] = np.nan

    return data

def apply_is_masld(row):
    # First, replace blank and don't know values with NA
    row_dict = replace_blank_and_dont_know_with_na(row.to_dict())
    
    # Then apply the is_masld function
    result = is_masld(
        row_dict['Sex'], 
        row_dict['Body mass index'], 
        row_dict['Waist circumference (cm) (2+ years)'],
        row_dict['Plasma glucose (mg/dL)'], 
        row_dict['Glycated hemoglobin: (%)'], 
        row_dict['Ever been told you have sugar/diabetes'],
        row_dict['Are you now taking insulin'], 
        row_dict['Are you now taking diabetes pills'], 
        row_dict['Overall average K1, systolic, BP(age 5+)'],
        row_dict['Overall average K5, diastolic, BP(age5+)'], 
        row_dict['Now taking prescribed medicine for HBP'], 
        row_dict['Serum triglycerides (mg/dL)'],
        row_dict['Take prescribed med to lower cholesterol'], 
        row_dict['Serum HDL cholesterol (mg/dL)']
    )
    return pd.Series(result)

# Read the CSV file
df = pd.read_csv('nhanes3/nhanes3.csv')

# Apply the is_masld function to each row
result = df.apply(apply_is_masld, axis=1)
df[['is_masld', 'is_body', 'is_diabetes', 'is_hypertension', 'is_dyslipidemia']] = result

def replace_with_nan(df, column, value):
    df[column] = df[column].replace(value, np.nan)
    return df

# Filter for rows where is_masld is 1
df = df[df['is_masld'] == 1]

# Define columns and their respective values to be replaced with NaN
columns_to_replace = {
    'Serum aspartate aminotransferase (AST) (U/L)': 888,
    'Serum alanine aminotransferase (ALT) (U/L)': 888,
    'Serum gamma glutamyl transferase (GGT) (U/L)': 8888,
    'Platelet count (1000 cells/uL)': 88888, 
    'Age at interview (screener)': 888, 
    'Body mass index': 8888,
    'Serum albumin (g/dL)': 888,
    'Glycated hemoglobin: (%)': 8888,
    'Serum creatinine (mg/dL)': 8888,
    'Waist circumference (cm) (2+ years)': 88888 
}

for column, value in columns_to_replace.items():
    df = replace_with_nan(df, column, value)

# Function to calculate FIB4
def calculate_fib4(age, ast, alt, platelets):
    if pd.isna(age) or pd.isna(ast) or pd.isna(alt) or pd.isna(platelets) or alt == 0 or platelets == 0:
        return np.nan
    return (age * ast) / (platelets * np.sqrt(alt))

# Function to calculate NFS
def calculate_nfs(age, bmi, diabetes, ast, alt, platelets, albumin):
    if pd.isna(age) or pd.isna(bmi) or pd.isna(diabetes) or pd.isna(ast) or pd.isna(alt) or pd.isna(platelets) or pd.isna(albumin) or alt == 0:
        return np.nan
    ast_alt_ratio = ast / alt
    return (-1.675 + 
            (0.037 * age) + 
            (0.094 * bmi) + 
            (1.13 * diabetes) + 
            (0.99 * ast_alt_ratio) - 
            (0.013 * platelets) - 
            (0.66 * albumin))

# Apply the calculations to the DataFrame
df['FIB4'] = df.apply(lambda row: calculate_fib4(row['Age at interview (screener)'], row['ASPSI'], row['ATPSI'], row['PLP']), axis=1)
df['NFS'] = df.apply(lambda row: calculate_nfs(row['Age at interview (screener)'], row['Body mass index'], row['is_diabetes'], row['ASPSI'], row['ATPSI'], row['PLP'], row['AMP']), axis=1)

df['is_high_risk_f3_fib4'] = (df['FIB4'] > 1.30).astype(int)
df['is_high_risk_f3_nfs'] = (df['NFS'] > 0.676).astype(int)

def calculate_gfr(serum_cr, age, is_female):
    gfr = 175 * (serum_cr ** -1.154) * (age ** -0.203)
    if is_female:
        gfr *= 0.742
    return gfr

df['is_female'] = (df['Sex'] == 2).astype(int)
df['GFR'] = df.apply(lambda row: calculate_gfr(row['Serum creatinine (mg/dL)'], row['Age at interview (screener)'], row['is_female']), axis=1)

df = df.apply(lambda row: replace_blank_and_dont_know_with_na(row.to_dict()), axis=1, result_type='expand')

def calculate_framingham_risk_score(row):
    is_female = row['is_female']
    age = row['Age at interview (screener)']
    total_chol = row['Serum cholesterol (mg/dL)']
    hdl_chol = row['Serum HDL cholesterol (mg/dL)']
    syst_bp = row['Overall average K1, systolic, BP(age 5+)']
    bp_treated = 0 if pd.isna(row['Now taking prescribed medicine for HBP']) else (0 if row['Now taking prescribed medicine for HBP'] == 2 else row['Now taking prescribed medicine for HBP'])
    smoker = 0 if 'Do you smoke cigarettes now' in row and pd.isna(row['Do you smoke cigarettes now']) else (0 if row['Do you smoke cigarettes now'] == 2 else row['Do you smoke cigarettes now']) if 'Do you smoke cigarettes now' in row else (0 if pd.isna(row['Have you smoked 100+ cigarettes in life']) else (0 if row['Have you smoked 100+ cigarettes in life'] == 2 else row['Have you smoked 100+ cigarettes in life']))

    if any(pd.isna([age, total_chol, hdl_chol, syst_bp, bp_treated, smoker])):
        return np.nan

    if is_female == 1:
        if age <= 0 or total_chol <= 0 or hdl_chol <= 0 or syst_bp <= 0:
            return np.nan

        ln_age = np.log(age)
        ln_total_chol = np.log(total_chol)
        ln_hdl_chol = np.log(hdl_chol)
        ln_syst_bp = np.log(syst_bp)

        if age > 78:
            ln_age_smoker = np.log(78) * smoker
        else:
            ln_age_smoker = ln_age * smoker

        LWomen = (
            31.764001 * ln_age +
            22.465206 * ln_total_chol +
            (-1.187731) * ln_hdl_chol +
            2.552905 * ln_syst_bp +
            0.420251 * bp_treated +
            13.07543 * smoker +
            (-5.060998) * ln_age * ln_total_chol +
            (-2.996945) * ln_age_smoker -
            146.5933061
        )

        PWomen = 1 - 0.98767 * np.exp(LWomen)
        return PWomen

    else:
        if age <= 0 or total_chol <= 0 or hdl_chol <= 0 or syst_bp <= 0:
            return np.nan

        ln_age = np.log(age)
        ln_total_chol = np.log(total_chol)
        ln_hdl_chol = np.log(hdl_chol)
        ln_syst_bp = np.log(syst_bp)

        if age > 70:
            ln_age_smoker = np.log(70) * smoker
        else:
            ln_age_smoker = ln_age * smoker

        ln_age_sq = ln_age ** 2

        LMen = (
            52.00961 * ln_age +
            20.014077 * ln_total_chol +
            (-0.905964) * ln_hdl_chol +
            1.305784 * ln_syst_bp +
            0.241549 * bp_treated +
            12.096316 * smoker +
            (-4.605038) * ln_age * ln_total_chol +
            (-2.84367) * ln_age_smoker +
            (-2.93323) * ln_age_sq -
            172.300168
        )

        PMen = 1 - 0.9402 * np.exp(LMen)
        return PMen

# Apply the function to create the Framingham Risk Score column
df['Framingham_Risk_Score'] = df.apply(calculate_framingham_risk_score, axis=1)

df[[
    'Serum cholesterol (mg/dL)',
    'Serum HDL cholesterol (mg/dL)',
    'Overall average K1, systolic, BP(age 5+)',
    'Have you smoked 100+ cigarettes in life'
]].info()

df.to_csv('nhanes3_masld.csv', index=False)

mortality_df = pd.read_csv('NHANES_III_2019.csv')
df['seqn'] = df['SEQN'].astype(int)
merge_df = pd.merge(mortality_df, df, on='seqn', how='inner')
merge_df = merge_df[merge_df['eligstat'] != 2]
merge_df.to_csv('nhanes3_masld_mortality.csv')
