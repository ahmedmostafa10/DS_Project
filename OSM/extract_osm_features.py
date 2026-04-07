import pandas as pd
import numpy as np
import osmium
from sklearn.neighbors import BallTree
import math

class POIHandler(osmium.SimpleHandler):
    def __init__(self):
        super(POIHandler, self).__init__()
        self.pois = []
        
    def node(self, n):
        tags = dict(n.tags)
        # Check if the node matches any of our target categories
        if 'amenity' in tags and tags['amenity'] in ['school', 'hospital', 'university', 'clinic', 'cafe', 'restaurant']:
            self.pois.append({
                'poi_lon': n.location.lon,
                'poi_lat': n.location.lat,
                'amenity': tags['amenity'],
                'shop': None,
                'public_transport': None
            })
        elif 'shop' in tags and tags['shop'] in ['mall', 'supermarket']:
            self.pois.append({
                'poi_lon': n.location.lon,
                'poi_lat': n.location.lat,
                'amenity': None,
                'shop': tags['shop'],
                'public_transport': None
            })
        elif 'public_transport' in tags and tags['public_transport'] in ['station']:
            self.pois.append({
                'poi_lon': n.location.lon,
                'poi_lat': n.location.lat,
                'amenity': None,
                'shop': None,
                'public_transport': tags['public_transport']
            })
            
    # For ways (polygons) you would normally compute centroids, but libosmium only gives refs to node IDs by default. 
    # To keep it simple & fast without maintaining a gigantic coordinate cache, we'll rely on Point nodes.
    # OpenStreetMap is heavily node-based for POIs anyway (e.g. shops, cafes).

def calculate_osm_features():
    print("Loading property dataset...")
    # Load your latest merged properties file
    try:
        df_props = pd.read_csv("all_properties_merged.csv")
    except FileNotFoundError:
        print("Merged file not found! Make sure it exists in this directory.")
        return

    # Keep a copy of original to merge back if needed, but we'll work with rows that have lat/lon
    original_row_count = len(df_props)
    df_props = df_props[df_props['lat'].notnull() & df_props['lon'].notnull()].copy()
    missing_coords_count = original_row_count - len(df_props)
    
    print(f"Total properties loaded: {original_row_count}")
    if missing_coords_count > 0:
        print(f"⚠️ Skipped {missing_coords_count} properties because they are missing latitude or longitude data.")
        
    if len(df_props) == 0:
        print("No rows with latitude/longitude found!")
        return

    # Convert coordinates to Radians for the Haversine formula (required by BallTree)
    print("Converting property coordinates to radians...")
    df_props['lat_rad'] = np.deg2rad(df_props['lat'])
    df_props['lon_rad'] = np.deg2rad(df_props['lon'])

    # --- 1. Load OSM Data via libosmium ---
    print("\nReading Egypt OSM PBF file using osmium... (This is fast but may take ~30 seconds)")
    try:
        handler = POIHandler()
        handler.apply_file("egypt-latest.osm.pbf")
        
        if not handler.pois:
            print("No POIs found. The OSM filter might be too strict or the file is invalid.")
            return
            
        pois = pd.DataFrame(handler.pois)
    except Exception as e:
        print(f"Failed to load PBF file: {e}\nMake sure 'egypt-latest.osm.pbf' is in this folder!")
        return

    print(f"Loaded {len(pois)} POIs. Extracting their coordinates...")
    
    # Calculate radians for Haversine
    pois['poi_lat_rad'] = np.deg2rad(pois['poi_lat'])
    pois['poi_lon_rad'] = np.deg2rad(pois['poi_lon'])
    
    # Earth radius in kilometers (used to convert Haversine radians back to km)
    EARTH_RADIUS_KM = 6371.0

    # --- 2. Calculate Distances using BallTree (Lightning Fast Spatial Math) ---
    print("\nBuilding Spatial Trees and calculating distances...")
    
    # We will group by the type of amenity/shop to get specific distances
    poi_categories = {
        'school': pois[pois['amenity'] == 'school'],
        'hospital': pois[pois['amenity'].isin(['hospital', 'clinic'])],
        'supermarket': pois[pois['shop'] == 'supermarket'],
        'mall': pois[pois['shop'] == 'mall'],
        'transit_station': pois[pois['public_transport'] == 'station'],
        'cafe_restaurant': pois[pois['amenity'].isin(['cafe', 'restaurant'])]
    }

    # Features to keep track of new columns added
    new_features = []

    for cat_name, cat_df in poi_categories.items():
        if cat_df.empty:
            print(f" - No {cat_name} found in OSM data, skipping...")
            continue
            
        print(f" - Processing nearest {cat_name}...")
        
        # Build the Tree using the radians columns
        tree = BallTree(cat_df[['poi_lat_rad', 'poi_lon_rad']].values, metric='haversine')
        
        # A. Find distance to nearest ONE (k=1)
        dist, ind = tree.query(df_props[['lat_rad', 'lon_rad']].values, k=1)
        
        # Convert distances back to Kilometers
        dist_km = dist.flatten() * EARTH_RADIUS_KM
        
        col_name_nearest = f'dist_nearest_{cat_name}_km'
        df_props[col_name_nearest] = np.round(dist_km, 3)
        new_features.append(col_name_nearest)
        
        # B. Count how many are within a 1.5 KM radius (Density Feature)
        # Using tree.query_radius. r = 1.5km converted to radians
        radius_rad = 1.5 / EARTH_RADIUS_KM
        counts = tree.query_radius(df_props[['lat_rad', 'lon_rad']].values, r=radius_rad, count_only=True)
        
        col_name_density = f'{cat_name}_count_within_1.5km'
        df_props[col_name_density] = counts
        new_features.append(col_name_density)

    # --- 3. Clean up and Save ---
    # Drop the temporary radian calculation columns
    df_props.drop(columns=['lat_rad', 'lon_rad'], inplace=True, errors='ignore')
    
    print("\nSuccessfully calculated the following ML features:")
    for feature in new_features:
        print(f" - {feature}")

    output_filename = "all_properties_with_osm.csv"
    print(f"\nSaving final enriched dataset to {output_filename}...")
    df_props.to_csv(output_filename, index=False, encoding='utf-8')
    print("Done! You're ready to build your Machine Learning model.")

if __name__ == "__main__":
    calculate_osm_features()