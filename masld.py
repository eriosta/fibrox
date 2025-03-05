import pandas as pd
import numpy as np

def is_masld(
    sex,
    bmi,
    waist_cm,
    glucose_mgdl,
    hba1c_percent,
    diabetes_history,
    insulin_use,
    diabetes_pills,
    systolic_bp,
    diastolic_bp,
    hbp_med,
    triglycerides_mgdl,
    cholesterol_med,
    hdl_mgdl
):
    """
    Determines if a patient meets MASLD criteria based on NHANES III variables.

    Parameters (NHANES III):
        sex: Sex (1 = Male, 2 = Female)
        bmi: Body-mass index
        waist_cm: Waist circumference (cm)
        glucose_mgdl: Plasma glucose (mg/dL)
        hba1c_percent: Glycated hemoglobin (HbA1c)
        diabetes_history: History of diabetes (1 = Yes, 2 = No)
        insulin_use: Currently taking insulin (1 = Yes, 2 = No)
        diabetes_pills: Currently taking diabetes pills (1 = Yes, 2 = No)
        systolic_bp: Systolic blood pressure
        diastolic_bp: Diastolic blood pressure
        hbp_med: Taking medication for high blood pressure (1 = Yes, 2 = No)
        triglycerides_mgdl: Serum triglycerides (mg/dL)
        cholesterol_med: Taking cholesterol-lowering medication (1 = Yes, 2 = No)
        hdl_mgdl: Serum HDL cholesterol (mg/dL)

    Returns:
        dict: Dictionary with MASLD criteria results and individual criteria flags.
    """

    # Criterion 1, body: BMI or waist circumference
    bmi_criteria = bmi >= 25
    wc_criteria = waist_cm > (94 if sex == 1 else 80)
    is_body = int(bmi_criteria or wc_criteria)

    # Criterion 2, diabetes: Blood glucose or diabetes history/treatment
    is_diabetes = int(
        glucose_mgdl >= 100 or
        hba1c_percent >= 5.7 or
        diabetes_history == 1 or
        insulin_use == 1 or
        diabetes_pills == 1
    )

    # Criterion 3, hypertension: Blood pressure or antihypertensive treatment
    is_hypertension = int(
        systolic_bp >= 130 or
        diastolic_bp >= 85 or
        hbp_med == 1
    )

    # Criterion 4 and 5, dyslipidemia: Triglycerides, HDL cholesterol, or lipid-lowering treatment
    is_dyslipidemia = int(
        triglycerides_mgdl >= 150 or
        cholesterol_med == 1 or
        (hdl_mgdl <= 40 if sex == 1 else hdl_mgdl <= 50) or
        cholesterol_med == 1
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
df = pd.read_csv('nhanes3.csv')

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
    'Aspartate aminotransferase: SI(U/L)': 888,
    'Alanine aminotransferase:  SI (U/L)': 888,
    'Gamma glutamyl transferase: SI(U/L)': 8888,
    'Platelet count': 88888, 
    'Age at interview (screener) - qty': 888, 
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
df['FIB4'] = df.apply(lambda row: calculate_fib4(row['Age at interview (screener) - qty'], row['Aspartate aminotransferase: SI(U/L)'], row['Alanine aminotransferase:  SI (U/L)'], row['Platelet count']), axis=1)
df['NFS'] = df.apply(lambda row: calculate_nfs(row['Age at interview (screener) - qty'], row['Body mass index'], row['is_diabetes'], row['Aspartate aminotransferase: SI(U/L)'], row['Alanine aminotransferase:  SI (U/L)'], row['Platelet count'], row['Serum albumin (g/dL)']), axis=1)

df['is_high_risk_f3_fib4'] = (df['FIB4'] > 1.30).astype(int)
df['is_high_risk_f3_nfs'] = (df['NFS'] > 0.676).astype(int)

def calculate_gfr_ckdepi(serum_cr, age, is_female):
    """
    Calculate GFR using the CKD-EPI equation.
    
    Args:
        serum_cr: Serum creatinine in mg/dL
        age: Age in years  
        is_female: Boolean indicating if patient is female
    
    Returns:
        Calculated GFR value
    """
    if is_female:
        if serum_cr <= 0.7:
            A = 0.7
            B = -0.241
        else:
            A = 0.7 
            B = -1.2
    else:
        if serum_cr <= 0.9:
            A = 0.9
            B = -0.302
        else:
            A = 0.9
            B = -1.2
            
    gfr = 142 * ((serum_cr/A)**B) * (0.9938**age)
    
    if is_female:
        gfr *= 1.012
        
    return gfr


df['is_female'] = (df['Sex'] == 2).astype(int)

df['GFR_EPI'] = df.apply(lambda row: calculate_gfr_ckdepi(row['Serum creatinine (mg/dL)'], row['Age at interview (screener) - qty'], row['is_female']), axis=1)

df = df.apply(lambda row: replace_blank_and_dont_know_with_na(row.to_dict()), axis=1, result_type='expand')

def calculate_framingham_risk_score(row):
    is_female = row['is_female']
    age = row['Age at interview (screener) - qty']
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

import tableone

tmp = merge_df.dropna(subset=['FIB4','NFS','mortstat'])

tmp['is_cardiac_mortality'] = ((tmp['ucod_leading'] == 1) | 
                              (tmp['ucod_leading'] == 5)).astype(int)

tmp['Race-ethnicity'] = tmp['Race-ethnicity'].map({
    1: 'Non-Hispanic White',
    2: 'Non-Hispanic Black', 
    3: 'Mexican-American',
    4: 'Other'
})

tmp['Sex'] = tmp['Sex'].map({
    1: 'Male',
    2: 'Female'
})

columns = [
    'Age at interview (screener) - qty',
    'Sex',
    'Race-ethnicity',
    'Body mass index',
    'Alanine aminotransferase:  SI (U/L)',
    'Aspartate aminotransferase: SI(U/L)',
    'Gamma glutamyl transferase: SI(U/L)',
    'Platelet count',
    'Glycated hemoglobin: (%)',
    'Serum HDL cholesterol (mg/dL)',
    'Serum triglycerides (mg/dL)',
    'GFR_EPI',
    'Serum creatinine (mg/dL)',
    'Serum albumin (g/dL)',
    'is_body',
    'is_diabetes',
    'is_hypertension',
    'is_dyslipidemia',
    'FIB4',
    'NFS',
    'mortstat',
    'is_cardiac_mortality'
]

categorical = [
    'Sex',
    'Race-ethnicity',
    'is_body',
    'is_diabetes',
    'is_hypertension',
    'is_dyslipidemia',
    'mortstat',
    'is_cardiac_mortality'
]

# Create the TableOne object
table1 = tableone.TableOne(tmp, columns=columns, categorical=categorical)

# Print the table
print(table1.tabulate(tablefmt="github"))

