import streamlit as st
import pandas as pd
import atrisk_ai
import fibrox_ai
import shap
import matplotlib.pyplot as plt

# Streamlit app
st.set_page_config(
    page_title="FibroX AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.sidebar.title("FibroX AI")

# Show the date/time of the last commit on GitHub
import subprocess

def get_last_commit_date():
    return pd.Timestamp.now().strftime('%B %d, %Y')
    
last_commit_date = get_last_commit_date()
st.sidebar.markdown(f"**Last update:** {last_commit_date}")

st.sidebar.markdown(
    'Read our paper here: [Njei et al. (2024). Scientific Reports.](https://www.nature.com/articles/s41598-024-59183-4)'
)

# Welcome message explaining the tool
st.expander("**How do I use this tool?**", expanded=False).markdown("""
    This tool is designed to assist healthcare providers in assessing patients with Metabolic Associated Steatotic Liver Disease (MASLD). 
    It provides predictive scores for two patient populations:
    
    - **At-Risk AI**: For adults who may develop at-risk MASH, using ALT, GGT, Platelets, Age, and BMI.
    - **FibroX AI**: For adults who may develop advanced fibrosis (≥F3) and are at risk of 30-year cause-specific mortality, using Age, Hemoglobin A1c, ALT, AST, Platelets, BMI, and GFR.
    
    Please ensure to input all required fields accurately to obtain reliable predictions.
""")

selected_model = st.sidebar.radio(
    "Choose the model:",
    ["FibroX AI", "At-Risk AI"],
    index=0,  # Set FibroX AI as default
    help="""
    **FibroX AI**: Identifies adults with MASLD who may develop advanced fibrosis (≥F3) and are at risk of 30-year cause-specific mortality. Uses Age, Hemoglobin A1c, ALT, AST, Platelets, BMI and GFR.
    
    **At-Risk AI**: Identifies adults with MASLD who may develop at-risk MASH. 
    Uses ALT, GGT, Platelets, Age and BMI.
    """
)

# Define display labels for user input fields
display_labels_map = {
    "At-Risk AI": {
        'ALANINE AMINOTRANSFERASE (ALT) (U/L)': 'ALT (U/L)',
        'GAMMA GLUTAMYL TRANSFERASE (GGT) (IU/L)': 'GGT (U/L)',
        'PLATELET COUNT (1000 CELLS/UL)': 'Platelets (1000 cells/µL)',
        'AGE IN YEARS AT SCREENING': 'Age (years)',
        'BODY MASS INDEX (KG/M**2)': 'BMI (kg/m²)'
    },
    "FibroX AI": {
        'Age (years)': 'Age (years)',
        'Glycohemoglobin (%)': 'Hemoglobin A1c (%)',
        'Alanine aminotransferase (U/L)': 'ALT (U/L)',
        'Aspartate aminotransferase (U/L)': 'AST (U/L)',
        'Platelet count (1000 cells/µL)': 'Platelets (1000 cells/µL)',
        'Body-mass index (kg/m**2)': 'BMI (kg/m²)',
        'GFR_EPI': 'eGFR (mL/min/1.73 m²)'
    }
}

# Load model and features
if selected_model == "At-Risk AI":
    model_features = atrisk_ai.MODEL_FEATURES
    display_labels = display_labels_map["At-Risk AI"]
    model = atrisk_ai.load_model()
    prediction_function = atrisk_ai.predict
else:
    model_features = fibrox_ai.MODEL_FEATURES
    display_labels = display_labels_map["FibroX AI"]
    model = fibrox_ai.load_model()
    prediction_function = fibrox_ai.predict

# Collect user inputs dynamically based on model features
data = {}
all_inputs_filled = True  # Initialize check for all required inputs
if selected_model == "FibroX AI":
    serum_cr = None
    is_female = None

    for feature in model_features:
        if feature == "GFR_EPI":
            serum_cr = st.sidebar.number_input('Creatinine (mg/dL)', min_value=0.0, value=None)
            gender = st.sidebar.radio('Gender', ['Male', 'Female'])
            is_female = gender == 'Female'

            # Use the age already entered for the model
            age_feature = 'Age (years)'
            if age_feature in data:
                age = data[age_feature]
            else:
                st.sidebar.warning("Please enter a valid age to calculate GFR.")
                all_inputs_filled = False

            # Calculate GFR if all required inputs are present
            if serum_cr is not None and 'Age (years)' in data:
                age = data['Age (years)']
                data['GFR_EPI'] = fibrox_ai.calculate_gfr(serum_cr, age, is_female)
                st.sidebar.info(f"eGFR: {data['GFR_EPI']:.2f} mL/min/1.73 m²")
            else:
                data['GFR_EPI'] = None
                all_inputs_filled = False
        else:
            label = display_labels[feature]
            input_value = st.sidebar.number_input(f'{label}', min_value=0.0, value=None)
            data[feature] = input_value
            if input_value is None:
                all_inputs_filled = False
else:
    for feature in model_features:
        label = display_labels[feature]
        input_value = st.sidebar.number_input(f'{label}', min_value=0.0, value=None)
        data[feature] = input_value
        if input_value is None:
            all_inputs_filled = False

# Convert collected data into DataFrame
data_df = pd.DataFrame([data])

# Conditionally display the Run button
if all_inputs_filled:
    if st.sidebar.button(f'Run {selected_model} Model'):
        with st.spinner('Running the model...'):
            y_pred, y_pred_proba = prediction_function(model, data_df)

            # Calculate SHAP values
            explainer = shap.Explainer(model)
            shap_values = explainer(data_df)

            # Update SHAP feature names to use display labels
            shap_values.feature_names = [display_labels[feature] for feature in model_features]

            st.toast(f'{selected_model} ran successfully! 🎉')

            # Custom output for FibroX AI
            st.subheader("Results")
            if selected_model == "FibroX AI" and y_pred[0] == 1:
                st.warning(
                    """
                    ### Likely Advanced Fibrosis
                    ###### Likely to have at least advanced liver fibrosis (≥F3)

                    """
                )
            elif selected_model == "FibroX AI" and y_pred[0] == 0:
                st.info("""
                    ### Likely Not Advanced Fibrosis     
                    ###### Likely less to have advanced liver fibrosis. Does not rule out minimal (F0-F1) or significant fibrosis (F2)            
                    
                """)
            else:
                # For At-Risk AI
                st.warning("####" + ("Likely High-Risk MASH" if y_pred[0] == 1 else "Likely Not High-Risk MASH"))

            # Display SHAP waterfall plot
            st.subheader("SHapley Additive exPlanations (SHAP) Explanations")
            fig = shap.plots.waterfall(shap_values[0], show=False)
            plt.savefig("shap_waterfall.png", dpi=500, bbox_inches='tight')
            st.image("shap_waterfall.png", caption="SHAP Explanations")

            # Add disclaimer about AI role
            st.info("""
                **Important**: This AI tool is designed to supplement, not replace, clinical decision making. 
                The results should be interpreted alongside clinical expertise, patient history, and other relevant medical information. 
                Healthcare providers should use their professional judgment when incorporating these predictions into patient care decisions.
            """)
else:
    st.sidebar.warning("Please fill in all required fields.")