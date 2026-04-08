import sqlite3
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import scipy.stats as stats
import logging
from datetime import datetime
import time
import json
import os
import re
import urllib.robotparser
import osmium
from sklearn.neighbors import BallTree
import math
from pydantic import BaseModel, ValidationError, Field, field_validator
from typing import Optional

class DataCollectionPipeline:
    """
    Unified data collection from multiple
    sources: databases, APIs, web.
    """
    def __init__(self, db_path="collected_data.db"):
        """
        Initialize pipeline with database and logging.
        Args:
            db_path (str): Path to SQLite database file.
                           Default: "collected_data.db"
        """
        # ── LOGGING SETUP ──────────────────────────
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("pipeline.log", encoding="utf-8"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # ── DATABASE SETUP ─────────────────────────
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._create_tables()   # Build schema

        # ── HTTP SESSION SETUP ────────────────────
        self.session = requests.Session()
        
        retry_strategy = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        })
        self.logger.info("Pipeline initialized")

    def _create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS api_data (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            source       TEXT NOT NULL,
            data_type    TEXT NOT NULL,
            content      TEXT NOT NULL,
            collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS scraped_data (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            url        TEXT NOT NULL,
            title      TEXT,
            content    TEXT,
            scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_logs (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            source_type      TEXT NOT NULL,
            records_collected INTEGER,
            status           TEXT,
            error_message    TEXT,
            timestamp        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")
        self.conn.commit()

    def collect_from_database(self, query, source_db_path):
        """
        Collect data from another SQLite database.
        """
        self.logger.info(f"Collecting from: {source_db_path}")
        try:
            source_conn = sqlite3.connect(source_db_path)
            df = pd.read_sql_query(query, source_conn)
            source_conn.close()
            self.logger.info(f"Collected {len(df)} records")
            self._log_collection("database", len(df), "success")
            return df
        except Exception as e:
            self.logger.error(f"Database error: {e}")
            self._log_collection("database", 0, "error", str(e))
            return pd.DataFrame()

    def collect_from_api(self, url, params=None):
        """
        Collect data from REST API.
        """
        self.logger.info(f"Collecting from API: {url}")
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO api_data (source, data_type, content) VALUES (?, ?, ?)",
                (url, "json", json.dumps(data))
            )
            self.conn.commit()
            self._log_collection("api", 1, "success")
            return data
        except Exception as e:
            self._log_collection("api", 0, "error", str(e))
            return None

    def collect_from_web(self, url, max_pages=None):
        """
        Scrape data from website using the exact logic from scraper.py
        and save it into the pipeline's scraped_data table.
        """
        self.logger.info(f"Collecting from Web: {url}")
        
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(urllib.parse.urljoin(url, "/robots.txt"))
        try:
            rp.read()
            user_agent = self.session.headers["User-Agent"]
            if not rp.can_fetch(user_agent, url):
                self.logger.error("Scraping is disallowed by robots.txt. Aborting.")
                self._log_collection("web", 0, "error", "Disallowed by robots.txt")
                return {}
            self.logger.info("robots.txt check passed. Proceeding with scraping...")
        except Exception as e:
            self.logger.warning(f"Could not read robots.txt, proceeding anyway. Error: {e}")

        current_page = 1
        total_pages = 1
        total_saved = 0
        
        while current_page <= total_pages:
            page_url = url if current_page == 1 else f"{url}page-{current_page}/"
            self.logger.info(f"Fetching {page_url}... (Page {current_page} of {total_pages})")
            
            try:
                response = self.session.get(page_url, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                script_text = None
                scripts = soup.find_all('script')
                for script in scripts:
                    if script.text and 'window.state =' in script.text:
                        script_text = script.text
                        break
                
                if not script_text:
                    self.logger.error("Could not find the JSON data inside the webpage on this page.")
                    self._log_collection("web", 0, "error", "JSON block not found")
                    break

                match = re.search(r'window\.state\s*=\s*({.*?});', script_text, re.DOTALL)
                if not match:
                    self.logger.error("Regex could not extract JSON from script_text on this page.")
                    self._log_collection("web", 0, "error", "Regex failed")
                    break

                state_json_str = match.group(1)
                
                try:
                    state = json.loads(state_json_str)
                    
                    if current_page == 1:
                        fetched_pages = state.get('algolia', {}).get('content', {}).get('nbPages', 1)
                        total_pages = min(fetched_pages, max_pages) if max_pages is not None else fetched_pages
                        self.logger.info(f"Discovered total pages: {fetched_pages}. Target set to scrape: {total_pages}")
                    
                    listings = state.get('algolia', {}).get('content', {}).get('hits', [])
                    if not listings:
                        self.logger.warning("No listings found on this page. Stopping.")
                        break
                    self.logger.info(f"Found {len(listings)} listings on page {current_page}! Parsing...")
                    
                    cursor = self.conn.cursor()
                    records_inserted = 0

                    for index, listing in enumerate(listings):
                        try:
                            # Geometry
                            geography = listing.get('geography', {})
                            lat = geography.get('lat', None)
                            lon = geography.get('lng', None)

                            # Location mapping based on level
                            locations = listing.get('location', [])
                            loc_full = ", ".join(loc.get('name', '') for loc in locations if loc.get('name', None))
                            city = next((loc.get('name', None) for loc in locations if loc.get('level', -1) == 1), None)
                            town = next((loc.get('name', None) for loc in locations if loc.get('level', -1) == 2), None)
                            district = next((loc.get('name', None) for loc in locations if loc.get('level', -1) == 3), None)
                            subdistrict = next((loc.get('name', None) for loc in locations if loc.get('level', -1) == 4), None)

                            # Categories
                            categories = listing.get('category', [])
                            category = next((cat.get('name', None) for cat in categories if cat.get('level', -1) == 0), None)
                            listing_type = next((cat.get('name', None) for cat in categories if cat.get('level', -1) == 1), None)

                            # URL Details
                            listing_id = listing.get('id', None)
                            external_id = listing.get('externalID', str(listing_id) if listing_id is not None else None)
                            detail_url = f"https://www.bayut.eg/en/property/details-{external_id}.html" if external_id else None

                            # Dates
                            listed_date = None
                            created_at = listing.get('createdAt', None)
                            if created_at:
                                listed_date = datetime.fromtimestamp(created_at).strftime('%Y-%m-%d %H:%M:%S')

                            # Phones
                            phone_data = listing.get('phoneNumber', {})
                            contact_phone = phone_data.get('mobile', None)
                            contact_whatsapp = phone_data.get('whatsapp', None)

                            # Agency & Agent
                            agency = listing.get('agency', {})
                            owner_agent = listing.get('ownerAgent', {})
                            extra_fields = listing.get('extraFields', {})
                            
                            data_payload = {
                                'listing_id': listing_id,
                                'internal_id': listing.get('referenceNumber', None),
                                'category': category,
                                'listing_type': listing_type,
                                'detail_url': detail_url,
                                'property_type': listing_type,
                                'offering_type': listing.get('purpose', None),
                                'completion_status': listing.get('completionStatus', None),
                                'title': listing.get('title', None),
                                'price_egp': listing.get('price', None),
                                'price_period': listing.get('rentFrequency', None),
                                'price_currency': 'EGP',
                                'location_full': loc_full,
                                'city': city,
                                'town': town,
                                'district': district,
                                'subdistrict': subdistrict,
                                'lat': lat,
                                'lon': lon,
                                'bedrooms': listing.get('rooms', None),
                                'bathroom': listing.get('baths', None),
                                'area_value': listing.get('area', None),
                                'area_unit': 'SQM',
                                'furnished': listing.get('furnishingStatus', None),
                                'listing_level': listing.get('product', None),
                                'is_premium': listing.get('product', None) == 'premium',
                                'is_verified': listing.get('isVerified', False),
                                'is_featured': listing.get('product', None) == 'hot', 
                                'is_new_construction': listing.get('completionStatus', None) == 'off_plan',
                                'is_direct_from_developer': extra_fields.get('ownership', None) == 'primary',
                                'is_exclusive': None,
                                'listed_date': listed_date,
                                'images_count': listing.get('photoCount', 0),
                                'has_video': listing.get('videoCount', 0) > 0,
                                'video_url': None,
                                'reference': listing.get('referenceNumber', None),
                                'rera': None,
                                'description': None,
                                'amenities': None,
                                'payment_plan': None,
                                'agent_id': owner_agent.get('externalID', None),
                                'agent_name': owner_agent.get('name', None),
                                'agent_email': None,
                                'agent_is_verified': owner_agent.get('isTruBroker', False),
                                'agent_languages': None,
                                'broker_id': agency.get('id', None),
                                'broker_name': agency.get('name', None),
                                'broker_email': None, 
                                'broker_phone': contact_phone,
                                'contact_phone': contact_phone,
                                'contact_whatsapp': contact_whatsapp,
                                'contact_email': listing.get('hasEmail', False),
                                'scraped_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            }

                            title = listing.get('title', None)
                            cursor.execute(
                                "INSERT INTO scraped_data (url, title, content) VALUES (?,?,?)",
                                (page_url, title, json.dumps(data_payload))
                            )
                            records_inserted += 1
                            total_saved += 1

                        except Exception as entry_e:
                            self.logger.error(f"Error parsing property {index}: {entry_e}")

                    self.conn.commit()
                    self._log_collection("web", records_inserted, "success")

                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to decode the JSON object: {e}")
                    self._log_collection("web", 0, "error", str(e))
                except KeyError as e:
                    self.logger.error(f"The internal JSON structure was not as expected: missing key {e}")

            except requests.exceptions.HTTPError as e:
                self.logger.error(f"HTTP Error: {e}")
                self._log_collection("web", 0, "error", str(e))
                break
            except Exception as e:
                self.logger.error(f"An error occurred: {e}")
                self._log_collection("web", 0, "error", str(e))
                break
            
            current_page += 1
            time.sleep(0.2)

        return {"status": "Bayut scraper completed", "records_saved": total_saved}

    def collect_from_kaggle(self, dataset_identifier, output_fileName="kaggle_data.csv"):
        self.logger.info(f"Kaggle collection started for {dataset_identifier}.")
        try:
            import kagglehub
            import shutil
            
            path = kagglehub.dataset_download(dataset_identifier)
            csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
            if not csv_files:
                self._log_collection("kaggle", 0, "error", "No CSV files found")
                return
            
            total_records = 0
            all_dfs = []
            for csv_file in csv_files:
                df = pd.read_csv(os.path.join(path, csv_file))
                all_dfs.append(df)
                total_records += len(df)
                
            # Merge and save local copy of Kaggle dataset
            if all_dfs:
                final_df = pd.concat(all_dfs, ignore_index=True)
                final_df.to_csv(output_fileName, index=False)
                self.logger.info(f"Kaggle dataset successfully saved to {output_fileName}")
                
            self._log_collection("kaggle", total_records, "success")
            return total_records
        except ImportError:
            self._log_collection("kaggle", 0, "error", "kagglehub not installed")
        except Exception as e:
            self._log_collection("kaggle", 0, "error", str(e))

    def _log_collection(self, source_type, records, status, error_msg=None):
        """Log each collection attempt to pipeline_logs table."""
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO pipeline_logs (source_type, records_collected, status, error_message) VALUES (?, ?, ?, ?)",
            (source_type, records, status, error_msg)
        )
        self.conn.commit()

    def get_collection_stats(self):
        """Returns dict with collection statistics."""
        stats = {}
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM api_data")
        stats["api_records"] = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM scraped_data")
        stats["scraped_records"] = cursor.fetchone()[0]
        stats["logs"] = pd.read_sql_query("""
            SELECT source_type, COUNT(*) as count,
            SUM(CASE WHEN status='success' THEN 1 ELSE 0 END) AS successful
            FROM pipeline_logs
            GROUP BY source_type
        """, self.conn)
        return stats

    def export_all_data(self, output_dir="exports"):
        """Export all collected data to CSV files."""
        os.makedirs(output_dir, exist_ok=True)
        api_df = pd.read_sql_query("SELECT * FROM api_data", self.conn)
        api_df.to_csv(f"{output_dir}/api_data.csv", index=False)
        
        scraped_df = pd.read_sql_query("SELECT * FROM scraped_data", self.conn)
        if not scraped_df.empty and 'content' in scraped_df.columns:
            # The 'content' column stores data as JSON strings. Let's expand them into separate columns.
            content_expanded = pd.json_normalize(scraped_df['content'].apply(lambda x: json.loads(x) if pd.notna(x) else {}))
            # Merge the expanded features back to the original DataFrame and drop the raw 'content' column
            scraped_df = pd.concat([scraped_df.drop(columns=['content']), content_expanded], axis=1)
            
        scraped_df.to_csv(f"{output_dir}/scraped_data.csv", index=False)
        logs_df = pd.read_sql_query("SELECT * FROM pipeline_logs", self.conn)
        logs_df.to_csv(f"{output_dir}/pipeline_logs.csv", index=False)
        self.logger.info(f"Exported data tables into '{output_dir}' directory.")

    def close(self):
        """Close the database connection."""
        self.conn.close()
        self.logger.info("Pipeline closed")

class DataMergerPipeline:
    """
    Handles merging of datasets from different sources.
    """
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def merge_datasets(self, pf_file='propertyfinder.csv', bayut_file='bayut_properties_all.csv', output_filename='all_properties_merged.csv'):
        print("Starting the merge process...")
        
        # 1. Load PropertyFinder data
        if os.path.exists(pf_file):
            df_pf = pd.read_csv(pf_file)
            df_pf['source'] = 'PropertyFinder'
            print(f"Loaded {len(df_pf)} rows from PropertyFinder.")
            
            # Rename columns to match the Bayut standard
            df_pf.rename(columns={
                'bathrooms': 'bathroom',
                'is_direct_from_dev': 'is_direct_from_developer',
                'payment_method': 'payment_plan',
                'agent_is_super': 'agent_is_verified',
                'has_view_360': 'has_video'
            }, inplace=True)
        else:
            print(f"Warning: {pf_file} not found.")
            df_pf = pd.DataFrame()

        # 2. Load Bayut data
        if os.path.exists(bayut_file):
            df_bayut = pd.read_csv(bayut_file)
            df_bayut['source'] = 'Bayut'
            print(f"Loaded {len(df_bayut)} rows from Bayut.")
        else:
            print(f"Warning: {bayut_file} not found. Make sure to run merge_bayut_batches.py first!")
            df_bayut = pd.DataFrame()

        # 3. Combine DataFrames
        if df_pf.empty and df_bayut.empty:
            print("No datasets available to merge. Exiting.")
            return

        df_combined = pd.concat([df_pf, df_bayut], ignore_index=True)
        initial_count = len(df_combined)
        print(f"Combined total rows: {initial_count}")

        # 4. (Skipped) Remove duplicates
        # User requested not to remove duplicated rows
        final_count = len(df_combined)

        # 5. Save the final merged dataset
        os.makedirs(os.path.dirname(output_filename) or '.', exist_ok=True)
        df_combined.to_csv(output_filename, index=False, encoding='utf-8')
        print(f"✅ Successfully saved the final merged dataset with {final_count} rows to '{output_filename}'")


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

class OSMFeatureExtractorPipeline:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def extract_features(self, input_csv, osm_pbf, output_csv):
        print("Loading property dataset...")
        try:
            df_props = pd.read_csv(input_csv)
        except FileNotFoundError:
            print(f"Merged file {input_csv} not found! Make sure it exists.")
            return

        original_row_count = len(df_props)
        # Check if lat/lon columns exist
        if 'lat' not in df_props.columns or 'lon' not in df_props.columns:
            print(f"'lat' or 'lon' column missing in {input_csv}.")
            return
            
        df_props = df_props[df_props['lat'].notnull() & df_props['lon'].notnull()].copy()
        missing_coords_count = original_row_count - len(df_props)
        
        print(f"Total properties loaded: {original_row_count}")
        if missing_coords_count > 0:
            print(f"⚠️ Skipped {missing_coords_count} properties because they are missing latitude or longitude data.")
            
        if len(df_props) == 0:
            print("No rows with latitude/longitude found!")
            return

        print("Converting property coordinates to radians...")
        df_props['lat_rad'] = np.deg2rad(df_props['lat'])
        df_props['lon_rad'] = np.deg2rad(df_props['lon'])

        print(f"\\nReading Egypt OSM PBF file using osmium... (This is fast but may take ~30 seconds)")
        try:
            handler = POIHandler()
            handler.apply_file(osm_pbf)
            
            if not handler.pois:
                print("No POIs found. The OSM filter might be too strict or the file is invalid.")
                return
                
            pois = pd.DataFrame(handler.pois)
        except Exception as e:
            print(f"Failed to load PBF file: {e}\\nMake sure '{osm_pbf}' is in this folder!")
            return

        print(f"Loaded {len(pois)} POIs. Extracting their coordinates...")
        pois['poi_lat_rad'] = np.deg2rad(pois['poi_lat'])
        pois['poi_lon_rad'] = np.deg2rad(pois['poi_lon'])
        
        EARTH_RADIUS_KM = 6371.0

        print(f"\\nBuilding Spatial Trees and calculating distances...")
        poi_categories = {
            'school': pois[pois['amenity'] == 'school'],
            'hospital': pois[pois['amenity'].isin(['hospital', 'clinic'])],
            'supermarket': pois[pois['shop'] == 'supermarket'],
            'mall': pois[pois['shop'] == 'mall'],
            'transit_station': pois[pois['public_transport'] == 'station'],
            'cafe_restaurant': pois[pois['amenity'].isin(['cafe', 'restaurant'])]
        }

        new_features = []

        for cat_name, cat_df in poi_categories.items():
            if cat_df.empty:
                print(f" - No {cat_name} found in OSM data, skipping...")
                continue
                
            print(f" - Processing nearest {cat_name}...")
            tree = BallTree(cat_df[['poi_lat_rad', 'poi_lon_rad']].values, metric='haversine')
            
            dist, ind = tree.query(df_props[['lat_rad', 'lon_rad']].values, k=1)
            dist_km = dist.flatten() * EARTH_RADIUS_KM
            
            col_name_nearest = f'dist_nearest_{cat_name}_km'
            df_props[col_name_nearest] = np.round(dist_km, 3)
            new_features.append(col_name_nearest)
            
            radius_rad = 3.0 / EARTH_RADIUS_KM
            counts = tree.query_radius(df_props[['lat_rad', 'lon_rad']].values, r=radius_rad, count_only=True)
            
            col_name_density = f'{cat_name}_count_within_3km'
            df_props[col_name_density] = counts
            new_features.append(col_name_density)

        df_props.drop(columns=['lat_rad', 'lon_rad'], inplace=True, errors='ignore')
        
        print(f"\\nSuccessfully calculated the following ML features:")
        for feature in new_features:
            print(f" - {feature}")

        print(f"\\nSaving final enriched dataset to {output_csv}...")
        os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
        df_props.to_csv(output_csv, index=False, encoding='utf-8')
        print("✅ OSM Feature Extraction Complete!")

class PropertyRowSchema(BaseModel):
    city: Optional[str] = None
    town: Optional[str] = None
    district: Optional[str] = None
    area_unit: Optional[str] = None
    price_currency: Optional[str] = None
    bedrooms: Optional[float] = None
    bathroom: Optional[float] = None
    area_value: Optional[float] = None

    @field_validator('area_value')
    @classmethod
    def check_area_beds_ratio(cls, v, info):
        bedrooms = info.data.get('bedrooms')
        if v is not None and bedrooms is not None:
            if bedrooms > 2 and v < 25:
                raise ValueError("Illogical area vs bedrooms ratio")
        return v

    @field_validator('city', 'town', 'district', mode='before')
    @classmethod
    def check_capitalization(cls, v):
        if isinstance(v, str) and v != v.title() and v != v.lower() and v != v.upper():
            pass # Pydantic doesn't trivially group this state across rows but we can do type coercion check
        return v

class DataValidationPipeline:
    """
    Handles data quality validation & profiling.
    """
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.report_summary = {}

    def run_validation(self, input_csv, output_report_csv='data_quality_report.csv'):
        self.logger.info("Starting Data Validation...")
        try:
            df = pd.read_csv(input_csv)
            self.logger.info(f"Data loaded successfully! Shape: {df.shape}")
        except FileNotFoundError:
            self.logger.error(f"File {input_csv} not found. Cannot proceed with validation.")
            return
            
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        # 1. Accuracy Validation
        self.logger.info("--- Accuracy Validation ---")
        accuracy_issues = 0
        acc_details = []
        for col in numeric_cols:
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                self.logger.warning(f"Column '{col}' contains {negative_count} negative values.")
                accuracy_issues += negative_count
                acc_details.append(f"{col}: {negative_count} negatives")
                
        # Custom boundary checks
        if 'bedrooms' in df.columns:
            invalid_beds_count = (pd.to_numeric(df['bedrooms'], errors='coerce') > 12).sum()
            if invalid_beds_count > 0:
                self.logger.warning(f"Column 'bedrooms' contains {invalid_beds_count} values > 12.")
                accuracy_issues += invalid_beds_count
                acc_details.append(f"bedrooms: {invalid_beds_count} values > 12")
        
        if 'bathroom' in df.columns:
            invalid_baths_count = (pd.to_numeric(df['bathroom'], errors='coerce') > 12).sum()
            if invalid_baths_count > 0:
                self.logger.warning(f"Column 'bathroom' contains {invalid_baths_count} values > 12.")
                accuracy_issues += invalid_baths_count
                acc_details.append(f"bathroom: {invalid_baths_count} values > 12")
                
        if 'area_value' in df.columns:
            # Example heuristic: area over 100,000 might be an error or different unit
            invalid_area_count = (pd.to_numeric(df['area_value'], errors='coerce') > 100000).sum()
            if invalid_area_count > 0:
                self.logger.warning(f"Column 'area_value' contains {invalid_area_count} values > 100,000.")
                accuracy_issues += invalid_area_count
                acc_details.append(f"area_value: {invalid_area_count} values > 100k")
                
        if 'price_egp' in df.columns:
            # Check for reasonably high ceiling (e.g. > 2 Billion EGP)
            invalid_price_count = (pd.to_numeric(df['price_egp'], errors='coerce') > 2000000000).sum()
            if invalid_price_count > 0:
                self.logger.warning(f"Column 'price_egp' contains {invalid_price_count} extremely high values (> 2B EGP).")
                accuracy_issues += invalid_price_count
                acc_details.append(f"price_egp: {invalid_price_count} extreme highs (> 2B EGP)")

        if accuracy_issues == 0:
            self.logger.info("Basic numeric boundaries look accurate.")
            acc_details.append("No boundary violations detected.")
        
        self.report_summary['Accuracy'] = " | ".join(acc_details)

        # 2. Consistency Validation (With Pydantic)
        self.logger.info("--- Consistency Validation ---")
        consistency_errors = []
        con_details = []
        pydantic_error_count = 0
        
        # We can still do group-level checks first, but row-level schema validation via Pydantic handles mixed types and row logic
        # Run Pydantic check row by row for the schema
        
        # Replace expected_numeric_cols string check with pydantic type-checking natively
        for index, row in df.iterrows():
            # converting row to dict and swapping nans to None
            row_dict = row.replace({np.nan: None}).to_dict()
            try:
                PropertyRowSchema(**row_dict)
            except ValidationError as e:
                pydantic_error_count += 1
                # Log first 5 errors to avoid spamming the log
                if pydantic_error_count <= 5:
                    self.logger.warning(f"Row {index} failed Pydantic validation: {e}")

        if pydantic_error_count > 0:
            self.logger.warning(f"Total rows failing Pydantic consistency validation: {pydantic_error_count}")
            consistency_errors.append("Pydantic Schema/Type/Logic validation failed")
            con_details.append(f"{pydantic_error_count} rows failed Pydantic validation")

        cat_cols_to_check = ['city', 'town', 'district', 'property_type', 'listing_type']
        for col in cat_cols_to_check:
            if col in df.columns:
                lower_vals = df[col].dropna().astype(str).str.lower()
                original_unique = df[col].nunique()
                lower_unique = lower_vals.nunique()
                if original_unique > lower_unique:
                    self.logger.warning(f"[{col}] Found {original_unique - lower_unique} capitalization discrepancies.")
                    consistency_errors.append(f"{col} capitalization mismatch")
                    con_details.append(f"{col}: {original_unique - lower_unique} capitalization discrepancies")
                    
        units_cols = ['area_unit', 'price_currency']
        for col in units_cols:
            if col in df.columns:
                unique_units = df[col].dropna().unique()
                if len(unique_units) > 1:
                    self.logger.warning(f"[{col}] Inconsistent units found: {unique_units}")
                    consistency_errors.append(f"Mixed {col}")
                    con_details.append(f"{col}: Mixed units {list(unique_units)}")

        # Check for mixed data types within the same column holistically
        for col in df.columns:
            col_dropna = df[col].dropna()
            if not col_dropna.empty:
                inferred_type = pd.api.types.infer_dtype(col_dropna)
                if inferred_type.startswith('mixed') and inferred_type != 'mixed-integer-float':
                    self.logger.warning(f"[{col}] Column contains fundamentally mixed data types (inferred: {inferred_type}).")
                    consistency_errors.append(f"Mixed data types in {col}")
                    con_details.append(f"{col}: mixed-type values ({inferred_type})")
                
                if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
                    empty_str_count = col_dropna.astype(str).str.strip().eq('').sum()
                    if empty_str_count > 0:
                        self.logger.warning(f"[{col}] Found {empty_str_count} empty string values posing as valid data instead of NaNs.")
                        consistency_errors.append(f"Empty strings in {col}")
                        con_details.append(f"{col}: {empty_str_count} hidden empty strings")

        if not con_details:
            con_details.append("No consistency issues found")
            
        self.report_summary['Consistency'] = " | ".join(con_details)

        # 3. Completeness Analysis
        self.logger.info("--- Completeness Analysis ---")
        comp_details = []
        missing_data = df.isnull().sum()
        completeness_df = pd.DataFrame({'Missing Values': missing_data, 'Percentage (%)': (missing_data / len(df)) * 100})
        completeness_df = completeness_df[completeness_df['Missing Values'] > 0].sort_values(by='Percentage (%)', ascending=False)
        self.logger.info(f"Missing values found in {len(completeness_df)} columns.")
        
        # Log top 5 columns with missing data
        for idx, missing_row in completeness_df.head(5).iterrows():
            pct = missing_row['Percentage (%)']
            row_cnt = missing_row['Missing Values']
            self.logger.warning(f"Missing Data -> [{idx}]: {row_cnt} rows ({pct:.2f}%)")
            comp_details.append(f"{idx}: {pct:.2f}% missing")

        if not comp_details:
            comp_details.append("No missing data in dataset.")
            
        self.report_summary['Completeness'] = f"{len(completeness_df)} cols have missing -> Top: " + " | ".join(comp_details)

        # 4. Uniqueness Analysis
        self.logger.info("--- Uniqueness Analysis ---")
        uniq_details = []
        
        # Identify logical duplicates based on property characteristics rather than exact row match
        dup_subset = ['title', 'price_egp', 'location_full', 'bedrooms', 'bathroom', 'area_value']
        valid_subset = [col for col in dup_subset if col in df.columns]
        
        if valid_subset:
            duplicates_count = df.duplicated(subset=valid_subset).sum()
            self.logger.info(f"Duplicate property rows (ignoring broker/agency differences) based on {valid_subset}: {duplicates_count}")
            uniq_details.append(f"Sub-level duplicates: {duplicates_count} (subset: {valid_subset})")
        else:
            duplicates_count = df.duplicated().sum()
            self.logger.info(f"Exact duplicate rows: {duplicates_count}")
            uniq_details.append(f"Exact duplicates: {duplicates_count}")
        
        # Log cardinality for categorical columns
        for col in categorical_cols:
            if col in df.columns:
                u_cnt = df[col].nunique()
                self.logger.info(f"Cardinality -> [{col}]: {u_cnt} unique values")
                uniq_details.append(f"{col}: {u_cnt} unique")

        self.report_summary['Uniqueness'] = " | ".join(uniq_details)

        # 5. Outlier Detection using IQR
        self.logger.info("--- Outlier Detection: IQR ---")
        iqr_details = []
        for col in numeric_cols:
            if df[col].nunique() > 10 and not col.lower().endswith('id'):
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))]
                if len(outliers) > 0:
                    self.logger.info(f"IQR Outliers -> [{col}]: {len(outliers)} values outside 1.5*IQR boundaries")
                    iqr_details.append(f"{col}: {len(outliers)} outliers")
        
        if not iqr_details:
            iqr_details.append("No IQR outliers found.")
            
        self.report_summary['IQR Outliers'] = " | ".join(iqr_details)

        # 7. Distribution Profiling (Skewness & Kurtosis) & Relationships (Spearman)
        self.logger.info("--- Distribution Profiling & Relationships ---")
        dist_details = []
        for col in numeric_cols:
            if df[col].nunique() > 10 and not col.lower().endswith('id'):
                col_data = df[col].dropna()
                if not col_data.empty:
                    skewness = col_data.skew()
                    kurt = col_data.kurtosis()
                    self.logger.info(f"Distribution Profile -> [{col}] Skewness: {skewness:.2f}, Kurtosis: {kurt:.2f}")
                    dist_details.append(f"{col}: Skew={skewness:.2f}, Kurt={kurt:.2f}")

        if not dist_details:
            dist_details.append("Insufficient numeric data for distribution profile.")
            
        self.report_summary['Distribution'] = " | ".join(dist_details)
        
        rel_details = []
        if len(numeric_cols) > 1:
            # Using Spearman rank correlation because it is robust to outliers
            corr_matrix = df[numeric_cols].corr(method='spearman')
            self.logger.info("Calculated Spearman Correlation Matrix (robust to outliers).")
            # Log highly correlated feature pairs (> 0.8 or < -0.8)
            for i in range(len(corr_matrix.columns)):
                for j in range(i):
                    val = corr_matrix.iloc[i, j]
                    if abs(val) > 0.8:
                        self.logger.info(f"High Spearman Correlation: [{corr_matrix.columns[i]}] & [{corr_matrix.columns[j]}] = {val:.2f}")
                        rel_details.append(f"{corr_matrix.columns[i]}/{corr_matrix.columns[j]}:{val:.2f}")
        
        if not rel_details:
            if len(numeric_cols) > 1:
                rel_details.append("No strong (>0.8) Spearman correlations found.")
            else:
                rel_details.append("Insufficient numeric variables for correlation.")
                
        self.report_summary['Relationships'] = " | ".join(rel_details)
        
        # --- Generate & Save Distribution & Relationship Charts ---
        plots_out_dir = os.path.join(os.path.dirname(output_report_csv) or '.', 'plots')
        os.makedirs(plots_out_dir, exist_ok=True)
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # 1. Numeric Feature Histograms
            cols_to_plot = [c for c in numeric_cols if df[c].nunique() > 10 and not c.lower().endswith('id')][:4]
            if cols_to_plot:
                # pandas .hist creates multiple subplots automatically, no need to pass a single ax if multiple cols
                df[cols_to_plot].hist(bins=30, figsize=(12, 8), edgecolor='black')
                plt.suptitle('Histograms of Numeric Features')
                plt.tight_layout()
                hist_path = os.path.join(plots_out_dir, 'numeric_distribution.png')
                plt.savefig(hist_path)
                plt.close()
                self.logger.info(f"📊 Numeric distribution histograms saved to {hist_path}")
            
            # 2. Categorical Bar Chart (Class Distribution)
            if len(categorical_cols) > 0:
                # Prefer property_type if it exists, else the first categorical column
                target_cat = 'property_type' if 'property_type' in df.columns else categorical_cols[0]
                plt.figure(figsize=(10, 6))
                df[target_cat].value_counts().head(10).plot(kind='bar', color='coral', edgecolor='black')
                plt.title(f'Class Distribution: {target_cat}')
                plt.ylabel('Frequency')
                plt.xticks(rotation=45)
                plt.tight_layout()
                bar_path = os.path.join(plots_out_dir, f'class_distribution_{target_cat}.png')
                plt.savefig(bar_path)
                plt.close()
                self.logger.info(f"📊 Class distribution bar chart saved to {bar_path}")

            # 3. Spearman Correlation Heatmap
            if len(numeric_cols) > 1:
                plt.figure(figsize=(10, 8))
                # Hiding annotations (annot=False) if too many columns, else it looks cluttered.
                sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
                plt.title('Spearman Correlation Matrix')
                plt.tight_layout()
                heatmap_path = os.path.join(plots_out_dir, 'correlation_heatmap_spearman.png')
                plt.savefig(heatmap_path)
                plt.close()
                self.logger.info(f"📊 Correlation heatmap saved to {heatmap_path}")

        except ImportError:
            self.logger.warning("matplotlib or seaborn not installed. Skipping chart generation.")

        # 8. Description & Target Context
        self.logger.info("--- Description & Context ---")
        target_col = 'price'
        if target_col in df.columns:
            self.logger.info(f"Target Definition: Target defined as '{target_col}'")
            self.report_summary['Target Definition'] = f"Target defined as '{target_col}'."
        else:
            self.logger.info("Target Definition: Target variable undefined.")
            self.report_summary['Target Definition'] = "Target variable undefined."

        engineered_cols = [col for col in df.columns if any(kw in col.lower() for kw in ['osm', 'dist', 'nearest', 'count'])]
        self.logger.info(f"Engineered Features: {len(engineered_cols)} features found {engineered_cols}")
        self.report_summary['Engineered Features'] = f"{len(engineered_cols)} engineered features identified."

        # 9. Output Report
        report_df = pd.DataFrame(list(self.report_summary.items()), columns=['Dimension', 'Findings / Summary'])
        report_df.to_csv(output_report_csv, index=False)
        self.logger.info(f"✅ Data Quality Report exported to {output_report_csv}")

if __name__ == "__main__":
    pipeline = DataCollectionPipeline()
    
    print("\n--- Starting Web Scraper (Bayut) ---")
    # Quick sample: Limited to 1 page
    pipeline.collect_from_web("https://www.bayut.eg/en/egypt/properties-for-sale/", max_pages=1)
    
    print("\n--- Starting Kaggle Collection ---")
    pipeline.collect_from_kaggle("waddahali/real-estate-listings", output_fileName="kaggle_data.csv")
    
    print("\n--- Generating Stats ---")
    stats = pipeline.get_collection_stats()
    print("Collection Stats:")
    print(stats)
    
    print("\n--- Exporting DB Scraped Data ---")
    # Exporting all scraped DB tables directly into the root folder to generate scraped_data.csv
    pipeline.export_all_data(output_dir=".") 
    
    pipeline.close()

    print("\n--- Merging Datasets ---")
    merger = DataMergerPipeline()
    # Merge the existing PropertyFinder data with the newly scraped Bayut data
    merger.merge_datasets(pf_file='data/propertyfinder.csv', bayut_file='scraped_data.csv', output_filename='all_properties_merged.csv')

    # print("\n--- Extracting OSM Features ---")
    # osm_extractor = OSMFeatureExtractorPipeline()
    # osm_extractor.extract_features(
    #     input_csv='all_properties_merged.csv',
    #     osm_pbf='OSM/egypt-latest.osm.pbf',
    #     output_csv='all_properties_with_osm.csv'
    # )
    print("\n--- Running Data Validation ---")
    validator = DataValidationPipeline()
    validator.run_validation(
        input_csv='OSM/all_properties_with_osm.csv',
        output_report_csv='data_quality_report.csv'
    )