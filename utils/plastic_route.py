"""
EcoVision - Plastic Waste Collection Route Optimization
-------------------------------------------------------

Implements:
- Hotspot clustering using KMeans
- Greedy route optimization (nearest-neighbor)
- Designed for academic & prototype use

Author: EcoVision Project
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from math import radians, cos, sin, asin, sqrt

# ==============================
# DISTANCE FUNCTION (HAVERSINE)
# ==============================

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate great-circle distance between two points (km)
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return 6371 * c


# ==============================
# HOTSPOT CLUSTERING
# ==============================

def cluster_hotspots(df, lat_col="Latitude", lon_col="Longitude", n_clusters=3):
    """
    Cluster plastic waste hotspots spatially
    """
    coords = df[[lat_col, lon_col]].values
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["Cluster"] = kmeans.fit_predict(coords)
    return df, kmeans.cluster_centers_


# ==============================
# ROUTE OPTIMIZATION
# ==============================

def optimize_route(points, start_index=0):
    """
    Simple greedy nearest-neighbor route optimization
    """
    if len(points) == 0:
        return []

    visited = [start_index]
    route = [points[start_index]]

    while len(visited) < len(points):
        last = route[-1]
        min_dist = float("inf")
        next_idx = None

        for i, p in enumerate(points):
            if i in visited:
                continue
            dist = haversine(
                last[0], last[1],
                p[0], p[1]
            )
            if dist < min_dist:
                min_dist = dist
                next_idx = i

        visited.append(next_idx)
        route.append(points[next_idx])

    return route


# ==============================
# FULL PIPELINE
# ==============================

def generate_collection_routes(df,
                               lat_col="Latitude",
                               lon_col="Longitude",
                               n_clusters=3):
    """
    End-to-end route generation:
    - Cluster hotspots
    - Generate optimized route per cluster
    """

    df_clustered, centers = cluster_hotspots(
        df, lat_col, lon_col, n_clusters
    )

    routes = {}

    for cluster_id in sorted(df_clustered["Cluster"].unique()):
        cluster_points = df_clustered[
            df_clustered["Cluster"] == cluster_id
        ][[lat_col, lon_col]].values.tolist()

        route = optimize_route(cluster_points)
        routes[f"Cluster_{cluster_id}"] = route

    return routes


# ==============================
# TEST / DEMO
# ==============================

if __name__ == "__main__":
    # Simulated plastic waste points
    data = {
        "Latitude": [12.97, 12.98, 13.01, 19.07, 19.08, 28.61],
        "Longitude": [77.59, 77.60, 77.58, 72.87, 72.88, 77.21],
        "Waste_Tons": [5, 6, 4, 8, 7, 9]
    }

    df = pd.DataFrame(data)

    routes = generate_collection_routes(df, n_clusters=2)

    print("\n🚛 Optimized Plastic Collection Routes:")
    for cluster, route in routes.items():
        print(cluster, "→", route)
