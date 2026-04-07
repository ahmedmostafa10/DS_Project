# Real Estate Prediction

This repository contains a data pipeline and analysis workflow for real estate price prediction. The project involves scraping property listings, merging datasets, extracting location-based geographical features using OpenStreetMap, and validating the quality of the gathered data.

## Project Structure

The repository is organized into distinct stages of the data pipeline:

### 1. Scrapper (`/Scrapper`)
Contains scripts and data related to harvesting real estate property listings from online platforms.
- `scraper.py`: The main web scraping script.
- `bayut_properties_all.csv`: Scraped properties dataset from Bayut.

### 2. Data (`/data`)
Stores raw data collected from other sources.
- `propertyfinder.csv`: Property dataset from Property Finder.

### 3. Merger (`/merger`)
Handles the consolidation of various data sources into a single unified dataset.
- `merge_all_datasets.py`: Script to merge data from Bayut, Property Finder, and other sources.
- `all_properties_merged.csv`: The resulting unified dataset.

### 4. OSM Features Extraction (`/OSM`)
Enriches the property data with geographical and neighborhood features using OpenStreetMap (OSM) data.
- `extract_osm_features.py`: Script to extract spatial features (e.g., proximity to amenities, transport) based on property coordinates.
- `egypt-latest.osm.pbf`: Raw OpenStreetMap data file for Egypt.
- `all_properties_with_osm.csv`: The enriched dataset containing both property attributes and geolocated OSM features.

### 5. Validation (`/validator`)
Ensures the integrity and quality of the final working dataset.
- `data_quality_validation.ipynb`: A Jupyter Notebook used for performing initial Exploratory Data Analysis (EDA), handling missing values, and validating data quality metrics.

## Getting Started

1. **Scraping**: Run `Scrapper/scraper.py` to collect the latest property listings.
2. **Merging**: Execute `merger/merge_all_datasets.py` to combine the new scrapes with existing datasets in the `data/` directory.
3. **OSM Enrichment**: Download the latest OpenStreetMap PBF file for your target region (e.g., Egypt) into the `OSM/` folder, then run `OSM/extract_osm_features.py` to generate the enriched dataset.
4. **Validation**: Open `validator/data_quality_validation.ipynb` to verify data consistency before feeding it into machine learning models.
