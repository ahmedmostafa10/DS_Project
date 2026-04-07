import json
import re
import datetime
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def scrape_bayut_egypt():
    base_url = "https://www.bayut.eg/en/egypt/properties-for-sale/"
    
    # Headers to mimic a real browser to avoid getting blocked initially
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    }

    properties_data = []
    current_page = 1
    total_pages = 1  # Will be updated after the first request
    batch_num = 1
    total_saved = 0
    
    while current_page <= total_pages:
        url = base_url if current_page == 1 else f"{base_url}page-{current_page}/"
        print(f"Fetching {url}... (Page {current_page} of {total_pages})")
        
        try:
            response = requests.get(url, headers=headers)
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
                print("Could not find the JSON data inside the webpage on this page.")
                print("The website structure might have changed again or it might be blocking the request.")
                break

            match = re.search(r'window\.state\s*=\s*({.*?});', script_text, re.DOTALL)
            if not match:
                print("Regex could not extract JSON from script_text on this page.")
                break

            state_json_str = match.group(1)
            
            try:
                state = json.loads(state_json_str)
                
                # Fetch total pages if this is the first page
                if current_page == 1:
                    total_pages = state.get('algolia', {}).get('content', {}).get('nbPages', 1)
                    print(f"Discovered total pages: {total_pages}")
                
                # Navigate to the search hits (the listings)
                listings = state.get('algolia', {}).get('content', {}).get('hits', [])
                if not listings:
                    print("No listings found on this page. Stopping.")
                    break
                print(f"Found {len(listings)} listings on page {current_page}! Parsing...")
                
                for index, listing in enumerate(listings):
                    try:
                        # Geometry
                        geography = listing.get('geography', {})
                        lat = geography.get('lat')
                        lon = geography.get('lng')

                        # Location mapping based on level
                        locations = listing.get('location', [])
                        loc_full = ", ".join(loc.get('name', '') for loc in locations if loc.get('name'))
                        city = next((loc.get('name') for loc in locations if loc.get('level') == 1), None)
                        town = next((loc.get('name') for loc in locations if loc.get('level') == 2), None)
                        district = next((loc.get('name') for loc in locations if loc.get('level') == 3), None)
                        subdistrict = next((loc.get('name') for loc in locations if loc.get('level') == 4), None)

                        # Categories
                        categories = listing.get('category', [])
                        category = next((cat.get('name') for cat in categories if cat.get('level') == 0), None)
                        listing_type = next((cat.get('name') for cat in categories if cat.get('level') == 1), None)

                        # URL Details
                        detail_url = "https://www.bayut.eg/en/property/details-" + listing.get('externalID', str(listing.get('id'))) + ".html" if listing.get('externalID') else None

                        # Dates
                        listed_date = None
                        if listing.get('createdAt'):
                            listed_date = datetime.datetime.fromtimestamp(listing.get('createdAt')).strftime('%Y-%m-%d %H:%M:%S')

                        # Phones
                        phone_data = listing.get('phoneNumber', {})
                        contact_phone = phone_data.get('mobile')
                        contact_whatsapp = phone_data.get('whatsapp')

                        # Agency & Agent
                        agency = listing.get('agency', {})
                        owner_agent = listing.get('ownerAgent', {})
                        
                        properties_data.append({
                            'listing_id': listing.get('id'),
                            'internal_id': listing.get('referenceNumber'),
                            'category': category,
                            'listing_type': listing_type,
                            'detail_url': detail_url,
                            'property_type': listing_type,
                            'offering_type': listing.get('purpose'),
                            'completion_status': listing.get('completionStatus'),
                            'title': listing.get('title'),
                            'price_egp': listing.get('price'),
                            'price_period': listing.get('rentFrequency'),
                            'price_currency': 'EGP',
                            'location_full': loc_full,
                            'city': city,
                            'town': town,
                            'district': district,
                            'subdistrict': subdistrict,
                            'lat': lat,
                            'lon': lon,
                            'bedrooms': listing.get('rooms'),
                            'bathroom': listing.get('baths'),
                            'area_value': listing.get('area'),
                            'area_unit': 'SQM',
                            'furnished': listing.get('furnishingStatus'),
                            'listing_level': listing.get('product'),
                            'is_premium': listing.get('product') == 'premium',
                            'is_verified': listing.get('isVerified', False),
                            'is_featured': listing.get('product') == 'hot', 
                            'is_new_construction': listing.get('completionStatus') == 'off_plan',
                            'is_direct_from_developer': listing.get('extraFields', {}).get('ownership') == 'primary',
                            'is_exclusive': None,
                            'listed_date': listed_date,
                            'images_count': listing.get('photoCount'),
                            'has_video': listing.get('videoCount', 0) > 0,
                            'video_url': None,
                            'reference': listing.get('referenceNumber'),
                            'rera': None,
                            'description': None,
                            'amenities': None,
                            'payment_plan': None,
                            'agent_id': owner_agent.get('externalID'),
                            'agent_name': owner_agent.get('name'),
                            'agent_email': None,
                            'agent_is_verified': owner_agent.get('isTruBroker', False),
                            'agent_languages': None,
                            'broker_id': agency.get('id'),
                            'broker_name': agency.get('name'),
                            'broker_email': None, 
                            'broker_phone': contact_phone,
                            'contact_phone': contact_phone,
                            'contact_whatsapp': contact_whatsapp,
                            'contact_email': listing.get('hasEmail', False),
                            'scraped_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        })
                    except Exception as entry_e:
                        print(f"Error parsing property {index}: {entry_e}")

            except json.JSONDecodeError as e:
                print(f"Failed to decode the JSON object: {e}")
            except KeyError as e:
                print(f"The internal JSON structure was not as expected: missing key {e}")

        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error: {e}")
            print("The website might be blocking the request. Check your internet connection or IP.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            break
        
        # Save to Pandas DataFrame in batches of ~1000 rows
        if len(properties_data) >= 1000:
            df = pd.DataFrame(properties_data)
            file_name = f"bayut_properties_batch_{batch_num}.csv"
            df.to_csv(file_name, index=False, encoding='utf-8')
            print(f"--> Saved batch {batch_num}: {len(properties_data)} rows into {file_name}.")
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
        print(f"--> Saved final batch {batch_num}: {len(properties_data)} rows into {file_name}.")
        
    print(f"Finished scraping! Total rows saved: {total_saved}")

if __name__ == "__main__":
    scrape_bayut_egypt()
