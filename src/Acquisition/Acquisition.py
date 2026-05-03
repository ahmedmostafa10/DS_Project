import json
import logging
import os
import re
import time
import urllib.parse
import urllib.robotparser
from datetime import datetime

import numpy as np
import osmium
import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from sklearn.neighbors import BallTree
from urllib3.util.retry import Retry


class DataCollectionPipeline:
    """
    Unified data collection from multiple
    sources: APIs, web.
    """

    def __init__(self):
        """
        Initialize pipeline with logging and memory storage.
        """
        # ── LOGGING SETUP ──────────────────────────
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("pipeline.log", encoding="utf-8"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(self.__class__.__name__)

        # ── DATA STORAGE ─────────────────────────
        self.scraped_data = []
        self.pipeline_logs = []

        # ── HTTP SESSION SETUP ────────────────────
        self.session = requests.Session()

        retry_strategy = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
            }
        )
        self.logger.info("Pipeline initialized")

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

                soup = BeautifulSoup(response.text, "html.parser")

                script_text = None
                scripts = soup.find_all("script")
                for script in scripts:
                    if script.text and "window.state =" in script.text:
                        script_text = script.text
                        break

                if not script_text:
                    self.logger.error(
                        "Could not find the JSON data inside the webpage on this page."
                    )
                    self._log_collection("web", 0, "error", "JSON block not found")
                    break

                match = re.search(r"window\.state\s*=\s*({.*?});", script_text, re.DOTALL)
                if not match:
                    self.logger.error("Regex could not extract JSON from script_text on this page.")
                    self._log_collection("web", 0, "error", "Regex failed")
                    break

                state_json_str = match.group(1)

                try:
                    state = json.loads(state_json_str)

                    if current_page == 1:
                        fetched_pages = (
                            state.get("algolia", {}).get("content", {}).get("nbPages", 1)
                        )
                        total_pages = (
                            min(fetched_pages, max_pages)
                            if max_pages is not None
                            else fetched_pages
                        )
                        self.logger.info(
                            f"Discovered total pages: {fetched_pages}. Target set to scrape: {total_pages}"
                        )

                    listings = state.get("algolia", {}).get("content", {}).get("hits", [])
                    if not listings:
                        self.logger.warning("No listings found on this page. Stopping.")
                        break
                    self.logger.info(
                        f"Found {len(listings)} listings on page {current_page}! Parsing..."
                    )

                    records_inserted = 0

                    for index, listing in enumerate(listings):
                        try:
                            # Geometry
                            geography = listing.get("geography", {})
                            lat = geography.get("lat", None)
                            lon = geography.get("lng", None)

                            # Location mapping based on level
                            locations = listing.get("location", [])
                            loc_full = ", ".join(
                                loc.get("name", "") for loc in locations if loc.get("name", None)
                            )
                            city = next(
                                (
                                    loc.get("name", None)
                                    for loc in locations
                                    if loc.get("level", -1) == 1
                                ),
                                None,
                            )
                            town = next(
                                (
                                    loc.get("name", None)
                                    for loc in locations
                                    if loc.get("level", -1) == 2
                                ),
                                None,
                            )
                            district = next(
                                (
                                    loc.get("name", None)
                                    for loc in locations
                                    if loc.get("level", -1) == 3
                                ),
                                None,
                            )
                            subdistrict = next(
                                (
                                    loc.get("name", None)
                                    for loc in locations
                                    if loc.get("level", -1) == 4
                                ),
                                None,
                            )

                            # Categories
                            categories = listing.get("category", [])
                            category = next(
                                (
                                    cat.get("name", None)
                                    for cat in categories
                                    if cat.get("level", -1) == 0
                                ),
                                None,
                            )
                            listing_type = next(
                                (
                                    cat.get("name", None)
                                    for cat in categories
                                    if cat.get("level", -1) == 1
                                ),
                                None,
                            )

                            # URL Details
                            listing_id = listing.get("id", None)
                            external_id = listing.get(
                                "externalID",
                                str(listing_id) if listing_id is not None else None,
                            )
                            detail_url = (
                                f"https://www.bayut.eg/en/property/details-{external_id}.html"
                                if external_id
                                else None
                            )

                            # Dates
                            listed_date = None
                            created_at = listing.get("createdAt", None)
                            if created_at:
                                listed_date = datetime.fromtimestamp(created_at).strftime(
                                    "%Y-%m-%d %H:%M:%S"
                                )

                            # Phones
                            phone_data = listing.get("phoneNumber", {})
                            contact_phone = phone_data.get("mobile", None)
                            contact_whatsapp = phone_data.get("whatsapp", None)

                            # Agency & Agent
                            agency = listing.get("agency", {})
                            owner_agent = listing.get("ownerAgent", {})
                            extra_fields = listing.get("extraFields", {})

                            data_payload = {
                                "listing_id": listing_id,
                                "internal_id": listing.get("referenceNumber", None),
                                "category": category,
                                "listing_type": listing_type,
                                "detail_url": detail_url,
                                "property_type": listing_type,
                                "offering_type": listing.get("purpose", None),
                                "completion_status": listing.get("completionStatus", None),
                                "title": listing.get("title", None),
                                "price_egp": listing.get("price", None),
                                "price_period": listing.get("rentFrequency", None),
                                "price_currency": "EGP",
                                "location_full": loc_full,
                                "city": city,
                                "town": town,
                                "district": district,
                                "subdistrict": subdistrict,
                                "lat": lat,
                                "lon": lon,
                                "bedrooms": listing.get("rooms", None),
                                "bathroom": listing.get("baths", None),
                                "area_value": listing.get("area", None),
                                "area_unit": "SQM",
                                "furnished": listing.get("furnishingStatus", None),
                                "listing_level": listing.get("product", None),
                                "is_premium": listing.get("product", None) == "premium",
                                "is_verified": listing.get("isVerified", False),
                                "is_featured": listing.get("product", None) == "hot",
                                "is_new_construction": listing.get("completionStatus", None)
                                == "off_plan",
                                "is_direct_from_developer": extra_fields.get("ownership", None)
                                == "primary",
                                "is_exclusive": None,
                                "listed_date": listed_date,
                                "images_count": listing.get("photoCount", 0),
                                "has_video": listing.get("videoCount", 0) > 0,
                                "video_url": None,
                                "reference": listing.get("referenceNumber", None),
                                "rera": None,
                                "description": None,
                                "amenities": None,
                                "payment_plan": None,
                                "agent_id": owner_agent.get("externalID", None),
                                "agent_name": owner_agent.get("name", None),
                                "agent_email": None,
                                "agent_is_verified": owner_agent.get("isTruBroker", False),
                                "agent_languages": None,
                                "broker_id": agency.get("id", None),
                                "broker_name": agency.get("name", None),
                                "broker_email": None,
                                "broker_phone": contact_phone,
                                "contact_phone": contact_phone,
                                "contact_whatsapp": contact_whatsapp,
                                "contact_email": listing.get("hasEmail", False),
                                "scraped_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            }

                            title = listing.get("title", None)
                            self.scraped_data.append(
                                {
                                    "url": page_url,
                                    "title": title,
                                    "content": json.dumps(data_payload),
                                    "scraped_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                }
                            )

                            records_inserted += 1
                            total_saved += 1

                        except Exception as entry_e:
                            self.logger.error(f"Error parsing property {index}: {entry_e}")

                    self._log_collection("web", records_inserted, "success")

                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to decode the JSON object: {e}")
                    self._log_collection("web", 0, "error", str(e))
                except KeyError as e:
                    self.logger.error(
                        f"The internal JSON structure was not as expected: missing key {e}"
                    )

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

    def _log_collection(self, source_type, records, status, error_msg=None):
        """Log each collection attempt to pipeline_logs."""
        self.pipeline_logs.append(
            {
                "source_type": source_type,
                "records_collected": records,
                "status": status,
                "error_message": error_msg,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

    def get_collection_stats(self):
        """Returns dict with collection statistics."""
        stats = {"scraped_records": len(self.scraped_data)}

        logs_df = pd.DataFrame(self.pipeline_logs)
        if not logs_df.empty:
            stats["logs"] = (
                logs_df.groupby("source_type")
                .agg(
                    count=("source_type", "count"),
                    successful=("status", lambda x: (x == "success").sum()),
                )
                .reset_index()
            )
        else:
            stats["logs"] = pd.DataFrame()

        return stats

    def export_all_data(self, output_dir="exports"):
        """Export all collected data to CSV files."""
        os.makedirs(output_dir, exist_ok=True)

        scraped_df = pd.DataFrame(self.scraped_data)
        if not scraped_df.empty and "content" in scraped_df.columns:
            # The 'content' column stores data as JSON strings. Let's expand them into separate columns.
            content_expanded = pd.json_normalize(
                scraped_df["content"].apply(lambda x: json.loads(x) if pd.notna(x) else {})
            )

            # Remove redundant columns so they don't produce '.1' duplicates
            cols_to_drop = [
                col
                for col in ["id", "url", "title", "content", "scraped_at"]
                if col in scraped_df.columns
            ]
            scraped_df = pd.concat(
                [scraped_df.drop(columns=cols_to_drop), content_expanded], axis=1
            )

        scraped_df.to_csv(os.path.join(output_dir, "scraped_data.csv"), index=False)

        logs_df = pd.DataFrame(self.pipeline_logs)
        logs_df.to_csv(os.path.join(output_dir, "pipeline_logs.csv"), index=False)
        self.logger.info(f"Exported data tables into '{output_dir}' directory.")

    def close(self):
        """Close the pipeline."""
        self.logger.info("Pipeline closed")


class DataMergerPipeline:
    """
    Handles merging of datasets from different sources.
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def merge_datasets(
        self,
        pf_file="propertyfinder.csv",
        bayut_file="bayut_properties_all.csv",
        output_filename="all_properties_merged.csv",
    ):
        print("Starting the merge process...")

        # 1. Load PropertyFinder data
        if os.path.exists(pf_file):
            df_pf = pd.read_csv(pf_file)
            df_pf["source"] = "PropertyFinder"
            print(f"Loaded {len(df_pf)} rows from PropertyFinder.")

            # Rename columns to match the Bayut standard
            df_pf.rename(
                columns={
                    "bathrooms": "bathroom",
                    "is_direct_from_dev": "is_direct_from_developer",
                    "payment_method": "payment_plan",
                    "agent_is_super": "agent_is_verified",
                    "has_view_360": "has_video",
                },
                inplace=True,
            )
        else:
            print(f"Warning: {pf_file} not found.")
            df_pf = pd.DataFrame()

        # 2. Load Bayut data
        if os.path.exists(bayut_file):
            df_bayut = pd.read_csv(bayut_file)
            df_bayut["source"] = "Bayut"
            print(f"Loaded {len(df_bayut)} rows from Bayut.")
        else:
            print(
                f"Warning: {bayut_file} not found. Make sure to run merge_bayut_batches.py first!"
            )
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
        os.makedirs(os.path.dirname(output_filename) or ".", exist_ok=True)
        df_combined.to_csv(output_filename, index=False, encoding="utf-8")
        print(
            f"✅ Successfully saved the final merged dataset with {final_count} rows to '{output_filename}'"
        )


class POIHandler(osmium.SimpleHandler):
    def __init__(self):
        super(POIHandler, self).__init__()
        self.pois = []

    def node(self, n):
        tags = dict(n.tags)
        # Check if the node matches any of our target categories
        if "amenity" in tags and tags["amenity"] in [
            "school",
            "hospital",
            "university",
            "clinic",
            "cafe",
            "restaurant",
        ]:
            self.pois.append(
                {
                    "poi_lon": n.location.lon,
                    "poi_lat": n.location.lat,
                    "amenity": tags["amenity"],
                    "shop": None,
                    "public_transport": None,
                }
            )
        elif "shop" in tags and tags["shop"] in ["mall", "supermarket"]:
            self.pois.append(
                {
                    "poi_lon": n.location.lon,
                    "poi_lat": n.location.lat,
                    "amenity": None,
                    "shop": tags["shop"],
                    "public_transport": None,
                }
            )
        elif "public_transport" in tags and tags["public_transport"] in ["station"]:
            self.pois.append(
                {
                    "poi_lon": n.location.lon,
                    "poi_lat": n.location.lat,
                    "amenity": None,
                    "shop": None,
                    "public_transport": tags["public_transport"],
                }
            )


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
        if "lat" not in df_props.columns or "lon" not in df_props.columns:
            print(f"'lat' or 'lon' column missing in {input_csv}.")
            return

        df_props = df_props[df_props["lat"].notnull() & df_props["lon"].notnull()].copy()
        missing_coords_count = original_row_count - len(df_props)

        print(f"Total properties loaded: {original_row_count}")
        if missing_coords_count > 0:
            print(
                f"⚠️ Skipped {missing_coords_count} properties because they are missing latitude or longitude data."
            )

        if len(df_props) == 0:
            print("No rows with latitude/longitude found!")
            return

        print("Converting property coordinates to radians...")
        df_props["lat_rad"] = np.deg2rad(df_props["lat"])
        df_props["lon_rad"] = np.deg2rad(df_props["lon"])

        print(
            "\\nReading Egypt OSM PBF file using osmium... (This is fast but may take ~30 seconds)"
        )
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
        pois["poi_lat_rad"] = np.deg2rad(pois["poi_lat"])
        pois["poi_lon_rad"] = np.deg2rad(pois["poi_lon"])

        EARTH_RADIUS_KM = 6371.0

        print("\\nBuilding Spatial Trees and calculating distances...")
        poi_categories = {
            "school": pois[pois["amenity"] == "school"],
            "hospital": pois[pois["amenity"].isin(["hospital", "clinic"])],
            "supermarket": pois[pois["shop"] == "supermarket"],
            "mall": pois[pois["shop"] == "mall"],
            "transit_station": pois[pois["public_transport"] == "station"],
            "cafe_restaurant": pois[pois["amenity"].isin(["cafe", "restaurant"])],
        }

        new_features = []

        for cat_name, cat_df in poi_categories.items():
            if cat_df.empty:
                print(f" - No {cat_name} found in OSM data, skipping...")
                continue

            print(f" - Processing nearest {cat_name}...")
            tree = BallTree(cat_df[["poi_lat_rad", "poi_lon_rad"]].values, metric="haversine")

            dist, ind = tree.query(df_props[["lat_rad", "lon_rad"]].values, k=1)
            dist_km = dist.flatten() * EARTH_RADIUS_KM

            col_name_nearest = f"dist_nearest_{cat_name}_km"
            df_props[col_name_nearest] = np.round(dist_km, 3)
            new_features.append(col_name_nearest)

            radius_rad = 3.0 / EARTH_RADIUS_KM
            counts = tree.query_radius(
                df_props[["lat_rad", "lon_rad"]].values, r=radius_rad, count_only=True
            )

            col_name_density = f"{cat_name}_count_within_3km"
            df_props[col_name_density] = counts
            new_features.append(col_name_density)

        df_props.drop(columns=["lat_rad", "lon_rad"], inplace=True, errors="ignore")

        print("\\nSuccessfully calculated the following ML features:")
        for feature in new_features:
            print(f" - {feature}")

        print(f"\\nSaving final enriched dataset to {output_csv}...")
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        df_props.to_csv(output_csv, index=False, encoding="utf-8")
        print("✅ OSM Feature Extraction Complete!")
