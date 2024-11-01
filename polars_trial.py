import polars as pl
import numpy as np
from sklearn.cluster import KMeans
from datetime import datetime, timedelta
import time
from tqdm import tqdm

print("Starting script execution...")

def load_and_process(file, date_col):
    print(f"Loading and processing {file}...")
    df = pl.read_csv(file, schema_overrides={'STARFIRE_INCIDENT_ID': pl.Utf8})
    print("CSV loaded successfully")

    if 'traffic' in file.lower():
        # For traffic data, use a more flexible date parsing
        df = df.with_columns(pl.col(date_col).str.to_datetime(format="%m/%d/%Y %I:%M:%S %p"))
    else:
        # For incident data, use the original format
        df = df.with_columns(pl.col(date_col).str.to_datetime(format="%Y-%m-%d %H:%M:%S"))

    print(f"Date column '{date_col}' processed")
    return df

df = load_and_process('all_ny_incidents_merged.csv', 'INCIDENT_DATETIME')
print(df.columns)
traffic_df = load_and_process('traffic_all_ny.csv', 'DATA_AS_OF')

# Add INCIDENT_BOROUGH to the columns needed
columns_needed = ['LATITUDE', 'LONGITUDE', 'INCIDENT_DATETIME',
                  'INCIDENT_RESPONSE_SECONDS_QY', 'INCIDENT_TRAVEL_TM_SECONDS_QY',
                  'INCIDENT_CLASSIFICATION', 'INCIDENT_CLASSIFICATION_GROUP',
                  'INCIDENT_BOROUGH']
df = df.select(columns_needed)

print("Sorting dataframes...")
traffic_df = traffic_df.sort('DATA_AS_OF')

print("Extracting coordinates for clustering...")
X = df.select(['LATITUDE', 'LONGITUDE']).to_numpy()

seed = 42
np.random.seed(seed)
m = 60000
past_cluster_sizes = [1,8,16,32,64]
future_cluster_sizes = [1,8,16,32]
traffic_cluster_sizes = [32]

incident_types = df['INCIDENT_CLASSIFICATION_GROUP'].unique().to_list()

past_cluster_labels = {}
future_cluster_labels = {}
traffic_cluster_labels = {}

print("Performing clustering for each incident type...")
for incident_type in tqdm(incident_types, desc="Incident Types"):
    # For past clusters, use all boroughs
    X_type_past = df.filter(pl.col('INCIDENT_CLASSIFICATION_GROUP') == incident_type).select(['LATITUDE', 'LONGITUDE']).to_numpy()

    # For future clusters, filter for Manhattan only before clustering
    manhattan_df = df.filter((pl.col('INCIDENT_CLASSIFICATION_GROUP') == incident_type) &
                             (pl.col('LATITUDE') >= 40.70) & (pl.col('LATITUDE') <= 40.88) &
                             (pl.col('LONGITUDE') >= -74.02) & (pl.col('LONGITUDE') <= -73.92))
    X_type_future = manhattan_df.select(['LATITUDE', 'LONGITUDE']).to_numpy()


    for size in past_cluster_sizes:
        kmeans = KMeans(n_clusters=size, random_state=seed)
        labels = kmeans.fit_predict(X_type_past)
        past_cluster_labels[(incident_type, size)] = labels

    for size in future_cluster_sizes:
        kmeans = KMeans(n_clusters=size, random_state=seed)
        labels = kmeans.fit_predict(X_type_future)
        future_cluster_labels[(incident_type, size)] = labels
print("Extracting average coordinates for traffic data...")
def extract_average_coordinates(link_points):
    coords = []
    for point in link_points.split():
        if point.endswith(',-'):
            point = point[:-2]
        try:
            lat, lon = map(float, point.split(','))
            coords.append([lat, lon])
        except ValueError:
            continue
    return np.mean(coords, axis=0) if coords else np.array([np.nan, np.nan])

X_traffic = np.array([extract_average_coordinates(link) for link in traffic_df['LINK_POINTS']])

print("Performing clustering for traffic data...")
for size in traffic_cluster_sizes:
    kmeans = KMeans(n_clusters=size, random_state=seed)
    labels = kmeans.fit_predict(X_traffic)
    traffic_cluster_labels[size] = labels
    traffic_df = traffic_df.with_columns(pl.Series(name=f'TRAFFIC_CLUSTER_{size}', values=labels))

print("Generating time windows...")
np.random.seed(seed)
past_time_windows_h = {size: np.full(size, 0) for size in past_cluster_sizes}
past_time_windows_m = {size: np.full(size, 120) for size in past_cluster_sizes}
future_time_windows_h = {size: np.full(size, 0) for size in future_cluster_sizes}
future_time_windows_m = {size: np.full(size, 120) for size in future_cluster_sizes}
traffic_time_windows_h = {size: np.full(size, 0) for size in traffic_cluster_sizes}
traffic_time_windows_m = {size: np.full(size, 120) for size in traffic_cluster_sizes}

print("Generating random datetimes...")
start_date = datetime(2021, 1, 1)
end_date = datetime(2023, 12, 31)
time_between_dates = end_date - start_date
days_between_dates = time_between_dates.days
np.random.seed(seed)
random_datetimes = [start_date + timedelta(days=np.random.randint(days_between_dates), hours=np.random.randint(0, 24),
                                           minutes=np.random.randint(0, 60)) for _ in range(m)]

result_df = pl.DataFrame({'DATETIME': random_datetimes})

def get_average_speed(cluster_data, window_start, window_end):
    data_in_window = cluster_data.filter((pl.col('DATA_AS_OF') >= window_start) & (pl.col('DATA_AS_OF') < window_end))
    return data_in_window['SPEED'].mean()

print("Processing traffic clusters...")
for size in tqdm(traffic_cluster_sizes, desc="Traffic Cluster Sizes"):
    for cluster in tqdm(range(size), desc=f"Clusters (size {size})"):
        cluster_data = traffic_df.filter(pl.col(f'TRAFFIC_CLUSTER_{size}') == cluster)
        cluster_time_window_h = int(traffic_time_windows_h[size][cluster])
        cluster_time_window_m = int(traffic_time_windows_m[size][cluster])

        traffic_aggregations = []
        for dt in random_datetimes:
            traffic_window_start = dt - timedelta(hours=cluster_time_window_h) - timedelta(minutes=cluster_time_window_m)
            traffic_window_end = dt
            traffic_aggregation = get_average_speed(cluster_data, traffic_window_start, traffic_window_end)
            traffic_aggregations.append(traffic_aggregation)

        result_df = result_df.with_columns(pl.Series(name=f'traffic_cluster_{size}_{cluster}_timewindow_{cluster_time_window_h}_{cluster_time_window_m}', values=traffic_aggregations))

print("Processing past clusters...")
for incident_type in tqdm(incident_types, desc="Incident Types"):
    df_type = df.filter(pl.col('INCIDENT_CLASSIFICATION_GROUP') == incident_type)

    for size in tqdm(past_cluster_sizes, desc="Past Cluster Sizes"):
        df_type = df_type.with_columns(pl.Series(name=f'PAST_CLUSTER_{size}', values=past_cluster_labels[(incident_type, size)]))

        for cluster in tqdm(range(size), desc=f"Clusters (size {size})"):
            cluster_data = df_type.filter(pl.col(f'PAST_CLUSTER_{size}') == cluster)
            cluster_time_window_h = int(past_time_windows_h[size][cluster])
            cluster_time_window_m = int(past_time_windows_m[size][cluster])

            past_aggregations_count = []
            past_aggregations_mean = []
            for dt in random_datetimes:
                past_window_start = dt - timedelta(hours=cluster_time_window_h) - timedelta(minutes=cluster_time_window_m)
                past_window_end = dt
                past_data = cluster_data.filter((pl.col('INCIDENT_DATETIME') >= past_window_start) & (pl.col('INCIDENT_DATETIME') < past_window_end))
                past_aggregations_count.append(past_data['INCIDENT_TRAVEL_TM_SECONDS_QY'].count())
                past_aggregations_mean.append(past_data['INCIDENT_TRAVEL_TM_SECONDS_QY'].sum())

            result_df = result_df.with_columns(
                pl.Series(name=f'past_cluster_{incident_type}_{size}_{cluster}_timewindow_{cluster_time_window_h}_{cluster_time_window_m}_count_past', values=past_aggregations_count),
                pl.Series(name=f'past_cluster_{incident_type}_{size}_{cluster}_timewindow_{cluster_time_window_h}_{cluster_time_window_m}_mean_past', values=past_aggregations_mean)
            )

print("Processing future clusters...")
for incident_type in tqdm(incident_types, desc="Incident Types"):
    # Filter for Manhattan only for future clusters
    df_type = df.filter((pl.col('INCIDENT_CLASSIFICATION_GROUP') == incident_type) &
                             (pl.col('LATITUDE') >= 40.70) & (pl.col('LATITUDE') <= 40.88) &
                             (pl.col('LONGITUDE') >= -74.02) & (pl.col('LONGITUDE') <= -73.92))

    for size in tqdm(future_cluster_sizes, desc="Future Cluster Sizes"):
        df_type = df_type.with_columns(pl.Series(name=f'FUTURE_CLUSTER_{size}', values=future_cluster_labels[(incident_type, size)]))

        for cluster in tqdm(range(size), desc=f"Clusters (size {size})"):
            cluster_data = df_type.filter(pl.col(f'FUTURE_CLUSTER_{size}') == cluster)
            cluster_time_window_h = int(future_time_windows_h[size][cluster])
            cluster_time_window_m = int(future_time_windows_m[size][cluster])

            future_aggregations = []
            lagged_future_aggregations_count = []
            lagged_future_aggregations_mean = []
            for dt in random_datetimes:
                future_window_start = dt
                future_window_end = dt + timedelta(hours=cluster_time_window_h) + timedelta(minutes=cluster_time_window_m)
                future_data = cluster_data.filter((pl.col('INCIDENT_DATETIME') >= future_window_start) & (pl.col('INCIDENT_DATETIME') < future_window_end))
                future_aggregations.append(future_data['INCIDENT_TRAVEL_TM_SECONDS_QY'].sum())

                lagged_future_window_start = future_window_start - timedelta(hours=24)
                lagged_future_window_end = future_window_end - timedelta(hours=24)
                lagged_future_data = cluster_data.filter((pl.col('INCIDENT_DATETIME') >= lagged_future_window_start) & (pl.col('INCIDENT_DATETIME') < lagged_future_window_end))
                lagged_future_aggregations_count.append(lagged_future_data['INCIDENT_TRAVEL_TM_SECONDS_QY'].count())
                lagged_future_aggregations_mean.append(lagged_future_data['INCIDENT_TRAVEL_TM_SECONDS_QY'].sum())

            result_df = result_df.with_columns(
                pl.Series(name=f'future_cluster_{incident_type}_{size}_{cluster}_timewindow_{cluster_time_window_h}_{cluster_time_window_m}_future', values=future_aggregations),
                pl.Series(name=f'future_cluster_{incident_type}_{size}_{cluster}_timewindow_{cluster_time_window_h}_{cluster_time_window_m}_count_lag24h_past', values=lagged_future_aggregations_count),
                pl.Series(name=f'future_cluster_{incident_type}_{size}_{cluster}_timewindow_{cluster_time_window_h}_{cluster_time_window_m}_mean_lag24_past', values=lagged_future_aggregations_mean)
            )

print("Filling NaN values...")
result_df = result_df.fill_null(0)

print("Saving results to CSV...")
result_df.write_csv('results_with_traffic_clusters_polars.csv')

print("Script execution completed. Results saved to 'results_with_traffic_clusters_polars_single_clusters.csv'")

print("Resulting dataframe columns:")
for col in result_df.columns:
    print(col)