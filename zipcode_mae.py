import pandas as pd
import numpy as np
import re
import lightgbm as lgb
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')


def generate_random_groups(divisions, n_groups):
    """
    Generate random groups from the given divisions.
    """
    divisions = sorted(list(set(divisions)))
    total_divisions = len(divisions)
    base_size = total_divisions // n_groups
    remainder = total_divisions % n_groups

    groups = []
    current_idx = 0

    for i in range(n_groups):
        group_size = base_size + (1 if i < remainder else 0)
        group = divisions[current_idx:current_idx + group_size]
        groups.append(group)
        current_idx += group_size

    return groups


def get_past_columns(division_type, df_columns):
    """
    Retrieve past columns based on the division type.
    """
    if division_type.startswith('SIZE:'):
        size = division_type.split(':')[1]
        return [col for col in df_columns if 'past_cluster' in col and f'_{size}_' in col]
    else:
        return [col for col in df_columns if division_type in col]


# Load and preprocess data
original_results_df = pd.read_csv('results_with_traffic_clusters_polars2.csv').head(2000)
for col in original_results_df.columns:
    print(col)

# Define filters
incident_types = [
    'Medical Emergencies',
    'NonMedical Emergencies',
    'NonStructural Fires',
    'Structural Fires',
    'NonMedical MFAs'
]
past_incident_filter = ["Medical Emergencies + NonMedical Emergencies + NonStructural Fires + Structural Fires + NonMedical MFAs"]
future_incident_filter = ["Medical Emergencies + NonMedical Emergencies + NonStructural Fires + Structural Fires + NonMedical MFAs"]

# Extract traffic columns
traffic_columns = [col for col in original_results_df.columns if 'traffic' in col.lower()]
traffic_columns = list(dict.fromkeys(traffic_columns))

# Define division configurations
past_divisions = [
    'INCIDENT_BOROUGH', 'ZIPCODE'
]
future_divisions = [2, 4]

all_columns = original_results_df.columns.tolist()

# Basic preprocessing
original_results_df['datetime'] = pd.to_datetime(original_results_df['DATETIME'])
original_results_df = original_results_df.sort_values(by='datetime')
original_results_df.drop_duplicates(inplace=True)
original_results_df = original_results_df.sample(frac=0.98, random_state=42)
original_results_df = original_results_df.reset_index(drop=True)

original_results_df['time_of_day'] = original_results_df['datetime'].dt.hour
original_results_df['weekday'] = original_results_df['datetime'].dt.weekday + 1

# Create train and test masks
train_mask = ~(
    (original_results_df['datetime'].dt.year < 2022)
)
test_mask = (
    (original_results_df['datetime'].dt.month >= -1) &
    (original_results_df['datetime'].dt.year > 2021)
)

results_table = []
prediction_results = []

# Prepare future columns for each incident type
future_columns_dict = {}
for incident in incident_types:
    future_columns = [col for col in all_columns
                      if col.startswith('future') and incident in col]
    future_clusters = sorted(list(set([
        int(re.search(r'_8_(\d+)_timewindow', col).group(1))
        for col in future_columns
        if re.search(r'_8_(\d+)_timewindow', col)
    ])))
    future_columns_dict[incident] = {
        'columns': future_columns,
        'clusters': future_clusters
    }

# Initialize list to store feature importances per zipcode
zipcode_feature_importance_list = []

# Main processing loop
for with_traffic in [True, False]:
    print(f"{'Including' if with_traffic else 'Excluding'} traffic columns in the model.")
    for past_division in past_divisions:
        past_columns = get_past_columns(past_division, original_results_df.columns)
        if not past_columns:
            continue

        for future_div_count in future_divisions:
            total_mae = 0
            total_r2 = 0
            total_variance = 0
            total_samples = 0

            for incident in incident_types:
                future_info = future_columns_dict[incident]
                future_columns = future_info['columns']
                future_clusters = future_info['clusters']

                if not future_columns or not future_clusters:
                    print(f"No future columns or clusters found for incident type: {incident}")
                    continue

                future_groups = generate_random_groups(future_clusters, future_div_count)

                # Prepare independent variables
                independent_vars = past_columns + ['time_of_day', 'weekday']
                if with_traffic:
                    independent_vars += traffic_columns
                    print("traffic cols", traffic_columns)

                for i, group in enumerate(future_groups):
                    group_columns = [
                        col for col in future_columns
                        if col.endswith('_future') and
                           any(f'_8_{div}_timewindow' in col for div in group)
                    ]

                    if not group_columns:
                        print(f"No group columns found for incident type {incident} in group {group}")
                        continue

                    group_name = f'{incident}_future_group_{future_div_count}_{i}'
                    original_results_df[group_name] = original_results_df[group_columns].sum(axis=1)

                    X = original_results_df[independent_vars].copy()
                    y = original_results_df[group_name]

                    X = X.fillna(0)
                    valid_rows = ~y.isna()
                    X = X[valid_rows]
                    y = y[valid_rows]

                    X_train = X[train_mask[valid_rows]]
                    X_test = X[test_mask[valid_rows]]
                    y_train = y[train_mask[valid_rows]]
                    y_test = y[test_mask[valid_rows]]

                    if X_train.empty or X_test.empty:
                        print(f"No training or testing data for incident type {incident} and group {i}")
                        continue

                    X_train_inner, X_valid, y_train_inner, y_valid = train_test_split(
                        X_train, y_train, test_size=0.2, random_state=42
                    )

                    lgb_model = lgb.LGBMRegressor(
                        num_leaves=52,
                        n_estimators=300,
                        random_state=42,
                        n_jobs=6,
                        early_stopping_rounds=5
                    )

                    lgb_model.fit(
                        X_train_inner,
                        y_train_inner,
                        eval_set=[(X_valid, y_valid)],
                    )

                    y_pred = lgb_model.predict(X_test)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    variance = y_test.var()
                    n_samples = len(y_test)

                    total_mae += mae * n_samples
                    total_r2 += r2 * n_samples
                    total_variance += variance * n_samples
                    total_samples += n_samples

                    # Collect data for plotting
                    temp_df = pd.DataFrame({
                        'datetime': original_results_df.loc[X_test.index, 'datetime'],
                        'incident_type': incident,
                        'actual': y_test,
                        'predicted': y_pred,
                        'time_of_day': original_results_df.loc[X_test.index, 'time_of_day'],
                        'with_traffic': 'With Traffic' if with_traffic else 'Without Traffic'
                    })
                    prediction_results.append(temp_df)

                    # If the current division is ZIPCODE, calculate feature importance per zipcode
                    if past_division == 'ZIPCODE':
                        # Extract feature importances
                        feature_importances = lgb_model.feature_importances_
                        feature_names = lgb_model.feature_name_

                        feature_importance_df = pd.DataFrame({
                            'feature': feature_names,
                            'importance': feature_importances
                        })

                        # Extract zipcodes from feature names
                        zipcode_pattern = r'ZIPCODE_(\d+\.?\d*)'

                        feature_importance_df['zipcode'] = feature_importance_df['feature'].apply(
                            lambda x: re.search(zipcode_pattern, x).group(1)
                            if re.search(zipcode_pattern, x) else None
                        )

                        # Sum importances per zipcode
                        zipcode_importance = feature_importance_df.dropna(subset=['zipcode']).groupby('zipcode').agg({
                            'importance': 'sum'
                        }).reset_index()

                        zipcode_importance['incident_type'] = incident
                        zipcode_importance['with_traffic'] = 'With Traffic' if with_traffic else 'Without Traffic'

                        zipcode_feature_importance_list.append(zipcode_importance)

            # Calculate averages
            if total_samples > 0 and past_division != 'ZIPCODE':
                avg_mae = total_mae / total_samples
                avg_r2 = total_r2 / total_samples
                avg_variance = total_variance / total_samples

                results_table.append({
                    'past_division': past_division,
                    'future_divisions': future_div_count,
                    'total_mae': avg_mae,
                    'r_squared': avg_r2,
                    'variance': avg_variance,
                    'with_traffic': 'With Traffic' if with_traffic else 'Without Traffic'
                })

# Process zipcode_feature_importance_list to compute average importance per zipcode per incident type
if zipcode_feature_importance_list:
    zipcode_importance_df = pd.concat(zipcode_feature_importance_list, ignore_index=True)
    zipcode_importance_df['zipcode'] = zipcode_importance_df['zipcode'].astype(str).str.rstrip('.0')

    # Aggregate by zipcode, incident type, and traffic inclusion
    zipcode_importance_avg_df = zipcode_importance_df.groupby(['incident_type', 'zipcode', 'with_traffic']).agg({
        'importance': 'mean'
    }).reset_index()
    zipcode_importance_avg_df.rename(columns={'importance': 'average_feature_importance'}, inplace=True)
else:
    zipcode_importance_avg_df = pd.DataFrame(columns=['incident_type', 'zipcode', 'average_feature_importance', 'with_traffic'])

# Create final results DataFrame
if results_table:
    results_df = pd.DataFrame(results_table)

    metrics = ['total_mae', 'r_squared', 'variance']
    for metric in metrics:
        pivot_table = results_df.pivot_table(
            index='future_divisions',
            columns=['past_division', 'with_traffic'],
            values=metric
        )
        # Save pivot table to Excel
        filename = 'results_cross_table_with_without_traffic.xlsx'
        with pd.ExcelWriter(filename) as writer:
            pivot_table.to_excel(writer, sheet_name=f'{metric} Results')

    # Also save the zipcode average feature importance to the same Excel file
    if not zipcode_importance_avg_df.empty:
        with pd.ExcelWriter(filename, mode='a', engine='openpyxl') as writer:
            zipcode_importance_avg_df.to_excel(writer, sheet_name='Zipcode Avg Feature Importance', index=False)
else:
    print("No results to save.")

# Also save the zipcode average feature importance to a separate CSV file
if not zipcode_importance_avg_df.empty:
    zipcode_importance_avg_df.to_csv('zipcode_average_feature_importance.csv', index=False)

# Concatenate all prediction results
if prediction_results:
    prediction_df = pd.concat(prediction_results, ignore_index=True)

    # Calculate residuals
    prediction_df['residual'] = abs(prediction_df['actual'] - prediction_df['predicted'])

    # Convert datetime to date for daily aggregation
    prediction_df['date'] = prediction_df['datetime'].dt.date

    # Aggregate residuals by date and traffic inclusion
    daily_residuals = prediction_df.groupby(['date', 'with_traffic'])['residual'].sum().reset_index()

    # Plotting daily residuals
    plt.figure(figsize=(14, 8))
    sns.scatterplot(data=daily_residuals, x='date', y='residual', hue='with_traffic')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)

    plt.title('Daily Average Residuals (Actual - Predicted) for Testing Set')
    plt.xlabel('Date')
    plt.ylabel('Average Residual')
    plt.grid(True, alpha=0.3)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig('daily_residuals_with_without_traffic.png')
    plt.close()

    # Aggregate predictions by time of day, incident type, and traffic inclusion using mean
    prediction_df['time_of_day_hour'] = prediction_df['time_of_day']
    aggregated_df = prediction_df.groupby(
        ['incident_type', 'time_of_day_hour', 'with_traffic']
    ).agg({
        'actual': 'mean',
        'predicted': 'mean'
    }).reset_index()

    # Plotting actual vs predicted by time of day and incident type
    plt.figure(figsize=(14, 8))
    sns.lineplot(
        data=aggregated_df,
        x='time_of_day_hour',
        y='actual',
        hue='incident_type',
        style='with_traffic',
        markers=True,
        dashes=False,
        palette='tab10',
    )
    sns.lineplot(
        data=aggregated_df,
        x='time_of_day_hour',
        y='predicted',
        hue='incident_type',
        style='with_traffic',
        markers=True,
        dashes=[(2, 2)],
        palette='tab10',
    )
    plt.title('Average Actual vs Predicted Values by Time of Day and Incident Type')
    plt.xlabel('Hour of Day')
    plt.ylabel('Predicted Aggregate Travel Time')
    plt.legend(title='Incident Type and Traffic Inclusion', loc='upper right')
    plt.tight_layout()
    plt.savefig('actual_vs_predicted_by_time_of_day_with_without_traffic.png')
    plt.close()
else:
    print("No prediction results to plot.")

# Create heatmaps for MAE, R_squared, and Variance
if results_table:
    results_df = pd.DataFrame(results_table)
    metrics = ['total_mae', 'r_squared', 'variance']
    for metric in metrics:
        pivot_table = results_df.pivot_table(
            index='future_divisions',
            columns=['past_division', 'with_traffic'],
            values=metric
        )
        if not pivot_table.empty:
            plt.figure(figsize=(12, 8))
            sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='YlOrRd')
            plt.title(f'{metric} Heatmap: Cross-table of Past and Future Divisions')
            plt.xlabel('Past Division and Traffic Inclusion')
            plt.ylabel('Number of Future Divisions')
            plt.tight_layout()
            plt.savefig(f'heatmap_cross_table_{metric}_with_without_traffic.png')
            plt.close()
else:
    print("No results to plot in heatmaps.")

print("Analysis complete. Plots and results have been saved.")