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

from Acquisition.Acquisition import DataCollectionPipeline, DataMergerPipeline,POIHandler,OSMFeatureExtractorPipeline
from validation.validation import PropertyRowSchema, DataValidationPipeline

if __name__ == "__main__":
    # pipeline = DataCollectionPipeline()
    
    # print("\n--- Starting Web Scraper (Bayut) ---")
    # # Quick sample: Limited to 1 page
    # pipeline.collect_from_web("https://www.bayut.eg/en/egypt/properties-for-sale/", max_pages=1)
    
    # print("\n--- Starting Kaggle Collection ---")
    # pipeline.collect_from_kaggle("waddahali/real-estate-listings", output_fileName="kaggle_data.csv")
    
    # print("\n--- Generating Stats ---")
    # stats = pipeline.get_collection_stats()
    # print("Collection Stats:")
    # print(stats)
    
    # print("\n--- Exporting DB Scraped Data ---")
    # # Exporting all scraped DB tables directly into the root folder to generate scraped_data.csv
    # pipeline.export_all_data(output_dir=".") 
    
    # pipeline.close()

    # print("\n--- Merging Datasets ---")
    # merger = DataMergerPipeline()
    # # Merge the existing PropertyFinder data with the newly scraped Bayut data
    # merger.merge_datasets(pf_file='data/propertyfinder.csv', bayut_file='scraped_data.csv', output_filename='all_properties_merged.csv')

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
        input_csv='OSM/last.csv',
        output_report_csv='data_quality_report.csv'
    )