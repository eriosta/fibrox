import os
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
import pickle
import xgboost as xgb

# Load and preprocess the data
df = pd.read_csv("nhanes3_masld_mortality.csv").dropna(subset=['mortstat', 'NFS', 'FIB4'])

df_subset = df[[
    'SEQN', 'Alanine aminotransferase:  SI (U/L)', 'Platelet count', 
    'Aspartate aminotransferase: SI(U/L)', 'Glycated hemoglobin: (%)', 'Body mass index', 'mortstat', 
    'ucod_leading', 'permth_exm', 'NFS', 'FIB4', 'GFR_EPI', 

    # Predictors of all-cause mortality in MASLD
    'Age at interview (screener) - qty', 'Sex', 
    'Race-ethnicity', 'Have you smoked 100+ cigarettes in life',
    'Poverty Income Ratio (unimputed income)', 'Highest grade or yr of school completed', 'is_hypertension', 'is_diabetes', 
    'Serum C-reactive protein (mg/dL)'
]]

# Drop rows where NaN values are present in the specified columns
df_subset = df_subset.dropna(subset=[
    'Age at interview (screener) - qty', 'Sex', 
    'Race-ethnicity', 'Have you smoked 100+ cigarettes in life',
    'Poverty Income Ratio (unimputed income)', 'Highest grade or yr of school completed', 'is_hypertension', 'is_diabetes'
])


# Filter out records where ucod_leading is 4
df_subset = df_subset[df_subset['ucod_leading'] != 4]

# Rename columns to more descriptive names
df_subset = df_subset.rename(columns={
    'Age at interview (screener) - qty': 'Age (years)', 
    'Glycated hemoglobin: (%)': 'Glycohemoglobin (%)',
    'Alanine aminotransferase:  SI (U/L)': 'Alanine aminotransferase (U/L)', 
    'Aspartate aminotransferase: SI(U/L)': 'Aspartate aminotransferase (U/L)',
    'Platelet count': 'Platelet count (1000 cells/µL)',
    'Body mass index': 'Body-mass index (kg/m**2)'
})

# Set index and drop missing values
index_cols = ['SEQN', 'mortstat', 'ucod_leading', 'permth_exm', 'NFS', 'FIB4']
df_subset = df_subset.set_index(index_cols).dropna()

# Load XGBoost model and make predictions
with open(r"C:\Users\erios\masldai_mortality\xgboost_model_isF3_Youden_Index.pkl", 'rb') as file:
    model = pickle.load(file)

features = [
    'Age (years)', 
    'Glycohemoglobin (%)', 
    'Alanine aminotransferase (U/L)',
    'Aspartate aminotransferase (U/L)',
    'Platelet count (1000 cells/µL)',
    'Body-mass index (kg/m**2)',
    'GFR_EPI'
]

data_dmatrix = xgb.DMatrix(df_subset[features])
y_pred_proba = model.predict(data_dmatrix)
# y_pred_proba = model.predict_proba(df_subset[features])[:, 1]


# Define cutoffs and create predictions
thresholds = {
    'Youden': 0.507
}

# Add predictions to dataframe
for key, threshold in thresholds.items():
    df_subset[f'Prediction_{key}'] = (y_pred_proba >= threshold).astype(int)
    df_subset[f'Probability_{key}'] = y_pred_proba

df_subset_reset = df_subset.reset_index()

# Create binary indicators
df_subset_reset['isfib4mod'] = (df_subset_reset['FIB4'] >= 1.3).astype(int)
df_subset_reset['isfib4high'] = (df_subset_reset['FIB4'] >= 2.67).astype(int)
df_subset_reset['isnfsmod'] = (df_subset_reset['NFS'] >= -1.455).astype(int)
df_subset_reset['isnfshigh'] = (df_subset_reset['NFS'] >= 0.676).astype(int)
df_subset_reset['is_cardiac_mortality'] = df_subset_reset['ucod_leading'].isin([1, 5]).astype(int)
df_subset_reset['is_malignancy_mortality'] = df_subset_reset['ucod_leading'].isin([2]).astype(int)
df_subset_reset['is_renal_mortality'] = df_subset_reset['ucod_leading'].isin([9]).astype(int)
df_subset_reset['is_diabetes_mortality'] = df_subset_reset['ucod_leading'].isin([7]).astype(int)
df_subset_reset['is_pneumonia_mortality'] = df_subset_reset['ucod_leading'].isin([8]).astype(int)

for spec, threshold in thresholds.items():
    df_subset_reset[f'FibroX_{spec}'] = (y_pred_proba >= threshold).astype(int)

# Create final mortality analysis dataset
df_mort = df_subset_reset[[
    'mortstat', 'is_cardiac_mortality', 'is_malignancy_mortality', 'is_renal_mortality', 'is_diabetes_mortality', 'is_pneumonia_mortality',
    'Prediction_Youden',
    'isnfsmod', 'isnfshigh',
    'isfib4mod', 'isfib4high',
    'Age (years)', 'Sex', 'permth_exm',
    'Race-ethnicity', 'Have you smoked 100+ cigarettes in life',
    'Poverty Income Ratio (unimputed income)', 'Highest grade or yr of school completed', 'is_hypertension', 'is_diabetes', 
    'Serum C-reactive protein (mg/dL)'
]].rename(columns={
    'Age (years)': 'age',
    'Sex': 'gender',
    'Race-ethnicity': 'race',
    'Have you smoked 100+ cigarettes in life': 'smoked_100_cigarettes',
    'Poverty Income Ratio (unimputed income)': 'poverty_income_ratio',
    'Highest grade or yr of school completed': 'education_level',
    'Serum C-reactive protein (mg/dL)': 'crp'
})

df_mort['permth_exm'].describe()/12

def run_survival_analysis(df_mort, follow_up_years, model_type, event_col, 
                          covariates=None, threshold_value=None):
    """
    Run survival analysis for specified follow-up period and model type
    
    Args:
        df_mort: DataFrame with mortality data
        follow_up_years: Follow-up period in years (10, 20, or 30)
        model_type: One of 'FibroX_90_spec', 'FibroX_95_spec', 'FibroX_99_spec', 
                   'isfib4mod', 'isfib4high'
        event_col: Column name for mortality event ('mortstat', 'is_cardiac_mortality', 
                   'is_malignancy_mortality', 'is_renal_mortality', 'is_diabetes_mortality', 
                   'is_pneumonia_mortality')
        covariates: List of covariate column names to include in the model
        threshold_value: Threshold value to show in legend (e.g. 1.3 for FIB-4 moderate)
    """
    
    # Filter by follow-up period
    df = df_mort[df_mort['permth_exm'] <= 12 * follow_up_years].copy()
    df['time'] = df['permth_exm'] / 12

    # Select relevant columns
    df = df[['time', event_col, model_type] + covariates].dropna()
    
    # Fit Cox model
    cph = CoxPHFitter()
    
    # Use splines for all models
    covariate_formula = " + ".join([f"C({cov})" if df[cov].dtype == 'object' else cov for cov in covariates])
    cph.fit(df, duration_col='time', event_col=event_col,
            formula=f"{covariate_formula} + {model_type}"
    )
    # Check proportional hazards assumption and save to file
    if event_col == "mortstat":
        event_type = f"{follow_up_years}-Year All-Cause Mortality"
    elif event_col == "is_cardiac_mortality":
        event_type = f"{follow_up_years}-Year Cardiovascular Mortality"
    elif event_col == "is_malignancy_mortality":
        event_type = f"{follow_up_years}-Year Malignancy Mortality"
    elif event_col == "is_renal_mortality":
        event_type = f"{follow_up_years}-Year Renal Mortality"
    elif event_col == "is_diabetes_mortality":
        event_type = f"{follow_up_years}-Year Diabetes Mortality"
    elif event_col == "is_pneumonia_mortality":
        event_type = f"{follow_up_years}-Year Pneumonia Mortality"
    else:
        event_type = f"{follow_up_years}-Year Mortality"
    
    with open('assumption_checks.txt', 'a') as f:
        f.write(f"\n\nResults for {model_type} - {follow_up_years} year follow-up - {event_type}:\n")
        f.write("="*80 + "\n")
        f.write("\nModel Summary:\n")
        f.write(str(cph.summary))
        f.write("\n\nProportional Hazards Test Results:\n")
        f.write(str(cph.check_assumptions(df, show_plots=False)))
        f.write("\n" + "="*80 + "\n")
    
    # Plot survival curves
    plt.figure()
    cph.plot_partial_effects_on_outcome(covariates=model_type, values=[0, 1], cmap='coolwarm')
    plt.title(f"Adjusted Cox Proportional Hazards ({follow_up_years}-Year Follow-Up)\n{event_type}")
    plt.xlabel("Years")
    plt.ylabel("Survival Probability")
    if follow_up_years == 10:
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
    elif follow_up_years == 20:
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(2))
    else:  # 30 years
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(3))
    
    # Set legend based on model type
    if model_type.startswith('Prediction_Youden'):
        plt.legend(["FibroX Score <0.507", 
                   "FibroX Score ≥0.507"])
    elif model_type.startswith('isfib4'):
        plt.legend([f"FIB-4 <{threshold_value}", f"FIB-4 ≥{threshold_value}"])
    elif model_type.startswith('isnfs'):
        plt.legend([f"NFS <{threshold_value}", f"NFS ≥{threshold_value}"])
    
    plt.grid(True)
    plt.ylim(0, 1)
    plt.xlim(0, follow_up_years)
    
    # Save plot
    mortality_type = 'all_cause' if event_col == 'mortstat' else 'cardiovascular' if event_col == 'is_cardiac_mortality' else 'malignancy' if event_col == 'is_malignancy_mortality' else 'renal' if event_col == 'is_renal_mortality' else 'diabetes' if event_col == 'is_diabetes_mortality' else 'pneumonia'

    os.makedirs('survival_plots', exist_ok=True)
    plt.savefig(f'survival_plots/{model_type}_{mortality_type}_{follow_up_years}yr.png')
    plt.close()
    
    # Return summary statistics
    summary = cph.summary.loc[model_type] if model_type.startswith('FibroX') else cph.summary.iloc[0]
    return {
        'Model': model_type,
        'Mortality Type': event_type,
        'Total N': len(df),
        'Deceased N': df[event_col].sum(),
        'exp(coef)': summary['exp(coef)'],
        'exp(coef) lower 95%': summary['exp(coef) lower 95%'],
        'exp(coef) upper 95%': summary['exp(coef) upper 95%'],
        'p': summary['p'],
        'p<0.05': summary['p'] < 0.05
    }

# Clear the assumption_checks.txt file before starting new analyses
open('assumption_checks.txt', 'w').close()

# Run analyses for different follow-up periods and models
follow_up_periods = [20,30]
fibrox_models = ['Prediction_Youden']
nfs_models = [('isnfsmod', -1.455), ('isnfshigh', 0.676)]
fib4_models = [('isfib4mod', 1.3), ('isfib4high', 2.67)]
event_cols = ['mortstat', 'is_cardiac_mortality']

covariates = ['age', 'gender', 'race', 
                'smoked_100_cigarettes', 'poverty_income_ratio', 'education_level', 
                'is_hypertension', 'is_diabetes', 'crp']
# Store results
results = []

# FibroX models
for event_col in event_cols:
    for years in follow_up_periods:
        for model in fibrox_models:
            results.append(run_survival_analysis(df_mort, years, model, event_col, covariates))

# NFS models        
for event_col in event_cols:
    for years in follow_up_periods:
        for model, threshold in nfs_models:
            results.append(run_survival_analysis(df_mort, years, model, event_col, covariates, threshold))

# FIB-4 models
for event_col in event_cols:
    for years in follow_up_periods:
        for model, threshold in fib4_models:
            results.append(run_survival_analysis(df_mort, years, model, event_col, covariates, threshold))

# Create summary dataframe
summary_df = pd.DataFrame(results)
print("\nSummary of all analyses:")
# summary_df = summary_df[summary_df['p<0.05'] == True]

print(summary_df)

# Rename models for output
summary_df['Model'] = summary_df['Model'].replace({
    'Prediction_Youden': 'FibroX',
    'isnfsmod': 'NFS >-1.455',
    'isnfshigh': 'NFS >=0.676',
    'isfib4mod': 'FIB-4 >1.3',
    'isfib4high': 'FIB-4 >=2.67'
})
summary_df.to_csv('adjusted_mortality.csv', index=False)
