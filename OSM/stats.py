import pandas as pd
import os


def get_apartment_sales_count(csv_path="all_properties_with_osm.csv"):
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path, low_memory=False)

    prop_types = ["apartment", "apartments"]
    offer_types = ["residential for sale", "for-sale", "for sale", "sale"]

    matches = df[
        df["property_type"].astype(str).str.lower().str.strip().isin(prop_types)
        & df["offering_type"].astype(str).str.lower().str.strip().isin(offer_types)
    ]

    count = len(matches)
    print(
        f"Number of rows with Property Type (Apartment) AND Offering Type (Sale): {count}"
    )

    # Save the matched rows to last.csv
    matches.to_csv("last.csv", index=False)
    print("Saved matching rows to last.csv")

    return count


if __name__ == "__main__":
    get_apartment_sales_count()
