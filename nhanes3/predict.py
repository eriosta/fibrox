import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from MLstatkit.stats import Delong_test

df = pd.read_csv("nhanes3_masld_mortality.csv").dropna(subset=['mortstat', 'NFS', 'FIB4'])
df_subset = df[['SEQN', 'Alanine aminotransferase:  SI (U/L)', 'Platelet count', 'Age at interview (screener) - qty', 'Aspartate aminotransferase: SI(U/L)', 'Glycated hemoglobin: (%)', 'GFR_EPI', 'Body mass index', 'mortstat', 'ucod_leading', 'permth_exm', 'NFS', 'FIB4']]

df_subset = df_subset[df_subset['ucod_leading'] != 4].rename(columns={
    
    'Age at interview (screener) - qty': 'Age (years)',
    'Glycated hemoglobin: (%)': 'Glycohemoglobin (%)', 
    'Alanine aminotransferase:  SI (U/L)': 'Alanine aminotransferase (U/L)',
    'Aspartate aminotransferase: SI(U/L)': 'Aspartate aminotransferase (U/L)',
    'Platelet count': 'Platelet count (1000 cells/µL)',
    'Body mass index': 'Body-mass index (kg/m**2)',
    
}).set_index(['SEQN', 'mortstat', 'ucod_leading', 'permth_exm', 'NFS', 'FIB4']).dropna()

with open("xgboost_model_isF3_Youden_Index.pkl", 'rb') as file:
    model = pickle.load(file)

features = ['Age (years)',
            'Glycohemoglobin (%)',
            'Alanine aminotransferase (U/L)',
            'Aspartate aminotransferase (U/L)',
            'Platelet count (1000 cells/µL)',
            'Body-mass index (kg/m**2)',
            'GFR_EPI']

df_subset['GFR_EPI'] = df_subset['GFR_EPI'].round(0)


data_dmatrix = xgb.DMatrix(df_subset[features])
y_pred_proba = model.predict(data_dmatrix)


df_subset_reset = df_subset.reset_index()
df_subset_reset['is_cardiac_mortality'] = ((df_subset_reset['ucod_leading'] == 1) | 
                                            (df_subset_reset['ucod_leading'] == 5)).astype(int)

def plot_auroc(df, model_predictions, comparison_scores_fib4, comparison_scores_nfs, save_path='auroc_probabilities.png'):
    """Plot AUROC for mortality labels and for comparison scores (FIB-4 and NFS), including DeLong's test p-values."""
    mortality_labels = ['mortstat', 'is_cardiac_mortality']
    mortality_titles = ['All-cause mortality', 'Cardiovascular-related mortality']
    plt.figure(figsize=(10, 5))

    # Plot AUROC for model predictions (FibroX)
    for i, (label, title) in enumerate(zip(mortality_labels, mortality_titles), start=1):
        # Calculate ROC curve and AUROC score for model predictions
        fpr, tpr, _ = roc_curve(df[label], model_predictions)
        auc = roc_auc_score(df[label], model_predictions)

        # Plot the ROC curve
        plt.subplot(1, len(mortality_labels), i)
        plt.plot(fpr, tpr, label=f'AUROC (FibroX) = {auc:.2f}', color='#2274E9')

        # Plot AUROC for comparison scores (FIB-4)
        fpr_fib4, tpr_fib4, _ = roc_curve(df[label], comparison_scores_fib4)
        auc_fib4 = roc_auc_score(df[label], comparison_scores_fib4)
        plt.plot(fpr_fib4, tpr_fib4, label=f'AUROC (FIB-4) = {auc_fib4:.2f}', color='gray')

        # Plot AUROC for comparison scores (NFS)
        fpr_nfs, tpr_nfs, _ = roc_curve(df[label], comparison_scores_nfs)
        auc_nfs = roc_auc_score(df[label], comparison_scores_nfs)
        plt.plot(fpr_nfs, tpr_nfs, label=f'AUROC (NFS) = {auc_nfs:.2f}', color='lightgray')

        # Perform DeLong test
        z_score_fib4, p_value_fib4 = Delong_test(df[label], comparison_scores_fib4, model_predictions)
        z_score_nfs, p_value_nfs = Delong_test(df[label], comparison_scores_nfs, model_predictions)

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{title}\nDeLong: FIB-4: {p_value_fib4:.3f}, NFS: {p_value_nfs:.3f}')
        plt.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

plot_auroc(df_subset_reset, y_pred_proba, df_subset_reset['FIB4'], df_subset_reset['NFS'], save_path='auroc_probabilities_12_21_2024.png')

thresholds = {
    'max_youden': 0.54
}

def analyze_thresholds(df_subset, y_pred_proba, thresholds, comparison_type='FIB4', comparison_thresholds=[1.3, 2.67]):
    """
    Analyze model performance at different thresholds compared to FIB4 or NFS scores.
    
    Args:
        df_subset: DataFrame with model features and outcomes
        y_pred_proba: Model prediction probabilities 
        thresholds: Dict of model probability thresholds to evaluate
        comparison_type: Either 'FIB4' or 'NFS'
        comparison_thresholds: List of thresholds for comparison score
    """
    # Loop through each model threshold
    for threshold_name, threshold in thresholds.items():
        print(f"Processing threshold {threshold_name}: {threshold}")
        
        df_subset['Prediction'] = (y_pred_proba >= threshold).astype(int)
        df_subset['Probability'] = y_pred_proba

        df_subset_reset = df_subset.reset_index()
        df_subset_reset['is_cardiac_mortality'] = ((df_subset_reset['ucod_leading'] == 1) | 
                                                    (df_subset_reset['ucod_leading'] == 5)).astype(int)

        # Calculate and export metrics function
        def calculate_and_export_metrics(df_subset_reset, comparisons):
            def calculate_metrics(true_labels, predictions, probabilities, label_name):
                auroc = roc_auc_score(true_labels, probabilities)
                threshold_auroc = roc_auc_score(true_labels, predictions)
                tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
                npv = tn / (tn + fn) if (tn + fn) > 0 else 0
                dor = (sensitivity / (1 - sensitivity)) / (1 - specificity) if (1 - specificity) > 0 else 0
                total_n = len(true_labels)
                
                return {
                    'Label': label_name,
                    'AUROC': auroc,
                    'Threshold_AUROC': threshold_auroc,
                    'Sensitivity': sensitivity,
                    'Specificity': specificity,
                    'PPV': ppv,
                    'NPV': npv,
                    'DOR': dor,
                    'Total_N': total_n
                }

            results = []
            
            # Create comparison score binary variables based on thresholds
            for comp_threshold in comparison_thresholds:
                col_name = f'{comparison_type}_{str(comp_threshold).replace(".", "_")}'
                df_subset_reset[col_name] = (df_subset_reset[comparison_type] >= comp_threshold).astype(int)
            
            # Calculate metrics for each outcome
            true_labels = ['mortstat', 'is_cardiac_mortality']
            for true_label in true_labels:
                # Model prediction metrics
                metrics = calculate_metrics(
                    df_subset_reset[true_label],
                    df_subset_reset['Prediction'],
                    df_subset_reset['Probability'],
                    'Prediction'
                )
                metrics['True_Label'] = true_label
                results.append(metrics)

                # Comparison score metrics for each threshold
                for comp_threshold in comparison_thresholds:
                    col_name = f'{comparison_type}_{str(comp_threshold).replace(".", "_")}'
                    metrics = calculate_metrics(
                        df_subset_reset[true_label],
                        df_subset_reset[col_name],
                        df_subset_reset[comparison_type],
                        f'{comparison_type}_{comp_threshold}'
                    )
                    metrics['True_Label'] = true_label
                    results.append(metrics)

            metrics_df = pd.DataFrame(results)

            # DeLong Test for AUROC comparisons
            def perform_delong_test(true_labels, probabilities, comparisons):
                results = []
                for model_a, model_b in comparisons:
                    z_score, p_value = Delong_test(true_labels, probabilities[model_a], probabilities[model_b])
                    results.append({
                        'Model_A': model_a,
                        'Model_B': model_b,
                        'Z-Score': z_score,
                        'P-Value': p_value
                    })
                return pd.DataFrame(results)

            true_labels = df_subset_reset[['mortstat', 'is_cardiac_mortality']].values
            probabilities = {
                'XGB': df_subset_reset['Probability'].values
            }

            # Add each comparison threshold to probabilities for DeLong test
            for comp_threshold in comparison_thresholds:
                col_name = f'{comparison_type}_{str(comp_threshold).replace(".", "_")}'
                probabilities[col_name] = df_subset_reset[comparison_type].values

            all_results = []
            for i, true_label in enumerate(true_labels.T):
                # Create comparisons for each threshold
                comparisons = [(col_name, 'XGB') for col_name in probabilities if col_name != 'XGB']
                results = perform_delong_test(true_label, probabilities, comparisons)
                all_results.append(results)

            delong_df = pd.concat(all_results, keys=['mortstat', 'is_cardiac_mortality'])

            # Export results
            output_filename = f'metrics_and_delong_results_{threshold_name}_{comparison_type}.xlsx'
            with pd.ExcelWriter(output_filename) as writer:
                metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
                delong_df.to_excel(writer, sheet_name='DeLong Test', index=True)
            
            return metrics_df, delong_df

        calculate_and_export_metrics(df_subset_reset, [])

analyze_thresholds(df_subset, y_pred_proba, thresholds, comparison_type='NFS', comparison_thresholds=[-1.455, 0.676])

analyze_thresholds(df_subset, y_pred_proba, thresholds, comparison_type='FIB4', comparison_thresholds=[1.3, 2.67])


def generate_shap_plots(model, df_subset, features, y_pred_proba, true_labels, threshold=0.5, output_folder='shap_plots'):
    """Generate SHAP plots for model interpretation."""
    
    # Initialize SHAP Explainer
    np.random.seed(42)
    explainer = shap.TreeExplainer(model)
    
    # Pass the DataFrame directly to preserve feature names
    shap_values = explainer(df_subset[features])
    
    # Extract base values and SHAP values
    base_values = shap_values.base_values
    shap_values_array = shap_values.values
    
    # Create a DataFrame for SHAP values and additional metrics
    shap_df = pd.DataFrame(shap_values_array, columns=features)
    shap_df['Total_SHAP'] = shap_df.abs().sum(axis=1)
    shap_df['Prediction'] = (y_pred_proba >= threshold).astype(int)
    shap_df['Probability'] = y_pred_proba
    shap_df['Base_Value'] = base_values
    
    # Combine SHAP values with original data
    final_df = pd.concat([df_subset.reset_index(drop=True)[['mortstat', 'ucod_leading'] + features], shap_df], axis=1)
    os.makedirs(output_folder, exist_ok=True)
    
    # Generate SHAP summary plots for each true label
    for true_label in true_labels:
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values_array, df_subset[features], plot_type="dot", show=False)
        plt.title(f"SHAP Summary Plot for {true_label}")
        plt.savefig(os.path.join(output_folder, f"shap_summary_plot_{true_label}.tiff"), format='tiff', dpi=300)
        plt.close()

        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values_array, df_subset[features], plot_type="bar", show=False)
        plt.title(f"SHAP Summary Bar Plot for {true_label}")
        plt.savefig(os.path.join(output_folder, f"shap_summary_bar_plot_{true_label}.tiff"), format='tiff', dpi=300)
        plt.close()

        # Calculate and plot the average SHAP value and standard deviation for each feature
        avg_shap_values = np.mean(np.abs(shap_values_array), axis=0)
        std_shap_values = np.std(np.abs(shap_values_array), axis=0)
        
        # Sort features by average SHAP values in descending order
        sorted_indices = np.argsort(avg_shap_values)
        sorted_features = np.array(features)[sorted_indices]
        sorted_avg_shap_values = avg_shap_values[sorted_indices]
        sorted_std_shap_values = std_shap_values[sorted_indices]
       
        plt.figure(figsize=(10, 8))
        plt.barh(sorted_features, sorted_avg_shap_values, xerr=sorted_std_shap_values, capsize=5, color='gray')
        plt.yticks(rotation=0, fontsize=14)
        plt.title(f"Average SHAP Values with SD for {true_label}", fontsize=14)
        plt.xlabel("Average SHAP Value", fontsize=14)
        plt.xticks(fontsize=14)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"shap_avg_sd_barplot_{true_label}.tiff"), format='tiff', dpi=300)
        plt.close()

    # Generate SHAP heatmap
    import matplotlib.colors as mcolors

    instance_order = np.argsort(shap_values_array.sum(1))
    ax_heat = shap.plots.heatmap(shap_values, instance_order=instance_order, show=False, cmap='PiYG')
    ax_heat.set_title("SHAP Heatmap")  # set title on the returned axes
    fig = ax_heat.figure  # retrieve the associated figure

    pos = ax_heat.get_position()
    new_ax_rect = [pos.x0, pos.y0 - 0.15, pos.width, 0.1]
    ax_outcome = fig.add_axes(new_ax_rect)

    y_true = df_subset[true_labels[0]].to_numpy()
    y_true_ordered = y_true[instance_order]
    outcome_array = y_true_ordered[np.newaxis, :]

    binary_cmap = mcolors.ListedColormap(["lightgray", "black"])
    ax_outcome.imshow(outcome_array, aspect='auto', cmap=binary_cmap)
    ax_outcome.set_xticks([])
    ax_outcome.set_yticks([])

    
    fig.savefig(os.path.join(output_folder, "shap_heatmap_with_true_y.tiff"), format='tiff', dpi=300)
    plt.close(fig)



generate_shap_plots(model, df_subset_reset, features, y_pred_proba, true_labels=['mortstat', 'is_cardiac_mortality'], threshold=0.5, output_folder='shap_plots')
