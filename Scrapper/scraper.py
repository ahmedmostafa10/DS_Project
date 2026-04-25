import json
import re
import datetime
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import pandas as pd
import time
import urllib.robotparser
import logging

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# File handler
file_handler = logging.FileHandler('scraper.log', encoding='utf-8')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def scrape_bayut_egypt():
    base_url = "https://www.bayut.eg/en/egypt/properties-for-sale/"
    
    # Check robots.txt before proceeding
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url("https://www.bayut.eg/robots.txt")
    try:
        rp.read()
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        if not rp.can_fetch(user_agent, base_url):
            logger.error("Scraping is disallowed by robots.txt. Aborting.")
            return
        logger.info("robots.txt check passed. Proceeding with scraping...")
    except Exception as e:
        logger.warning(f"Could not read robots.txt, proceeding anyway. Error: {e}")

    # Headers to mimic a real browser to avoid getting blocked initially
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    }

    # Setup session with retry strategy
    session = requests.Session()
    retry_strategy = Retry(
        total=5,  # Maximum number of retries
        backoff_factor=1,  # Wait 1, 2, 4, 8, 16 seconds between retries
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    properties_data = []
    current_page = 1
    total_pages = 1  # Will be updated after the first request
    batch_num = 1
    total_saved = 0
    
    while current_page <= total_pages:
        url = base_url if current_page == 1 else f"{base_url}page-{current_page}/"
        logger.info(f"Fetching {url}... (Page {current_page} of {total_pages})")
        
        try:
            response = session.get(url, headers=headers, timeout=10)
            response.raise_for_status() # Check if the request was successful
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find the injected JavaScript state object "window.state = {...}"
            script_text = None
            scripts = soup.find_all('script')
            for script in scripts:
                if script.text and 'window.state =' in script.text:
                    script_text = script.text
                    break
            
            if not script_text:
                logger.error("Could not find the JSON data inside the webpage on this page.")
                logger.error("The website structure might have changed again or it might be blocking the request.")
                break

            match = re.search(r'window\.state\s*=\s*({.*?});', script_text, re.DOTALL)
            if not match:
                logger.error("Regex could not extract JSON from script_text on this page.")
                break

            state_json_str = match.group(1)
            
            try:
                state = json.loads(state_json_str)
                
                # Fetch total pages if this is the first page
                if current_page == 1:
                    total_pages = state.get('algolia', {}).get('content', {}).get('nbPages', 1)
                    logger.info(f"Discovered total pages: {total_pages}")
                
                # Navigate to the search hits (the listings)
                listings = state.get('algolia', {}).get('content', {}).get('hits', [])
                if not listings:
                    logger.warning("No listings found on this page. Stopping.")
                    break
                logger.info(f"Found {len(listings)} listings on page {current_page}! Parsing...")
                
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
                            listed_date = datetime.datetime.fromtimestamp(created_at).strftime('%Y-%m-%d %H:%M:%S')

                        # Phones
                        phone_data = listing.get('phoneNumber', {})
                        contact_phone = phone_data.get('mobile', None)
                        contact_whatsapp = phone_data.get('whatsapp', None)

                        # Agency & Agent
                        agency = listing.get('agency', {})
                        owner_agent = listing.get('ownerAgent', {})
                        extra_fields = listing.get('extraFields', {})
                        
                        properties_data.append({
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
                            'scraped_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        })
                    except Exception as entry_e:
                        logger.error(f"Error parsing property {index}: {entry_e}")

            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode the JSON object: {e}")
            except KeyError as e:
                logger.error(f"The internal JSON structure was not as expected: missing key {e}")

        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP Error: {e}")
            logger.error("The website might be blocking the request. Check your internet connection or IP.")
            break
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            break
        
        # Save to Pandas DataFrame in batches of ~1000 rows
        if len(properties_data) >= 1000:
            df = pd.DataFrame(properties_data)
            file_name = f"bayut_properties_batch_{batch_num}.csv"
            df.to_csv(file_name, index=False, encoding='utf-8')
            logger.info(f"--> Saved batch {batch_num}: {len(properties_data)} rows into {file_name}.")
            total_saved += len(properties_data)
            properties_data = []  # Clear memory for the next batch
            batch_num += 1
            
        current_page += 1
        
        # 200 ms Delay as requested
        time.sleep(0.2)

    # After the loop finishes, save any remaining rows
    if properties_data:
        df = pd.DataFrame(properties_data)
        file_name = f"bayut_properties_batch_{batch_num}.csv"
        df.to_csv(file_name, index=False, encoding='utf-8')
        total_saved += len(properties_data)
        logger.info(f"--> Saved final batch {batch_num}: {len(properties_data)} rows into {file_name}.")
        
    logger.info(f"Finished scraping! Total rows saved: {total_saved}")

if __name__ == "__main__":
    scrape_bayut_egypt()
