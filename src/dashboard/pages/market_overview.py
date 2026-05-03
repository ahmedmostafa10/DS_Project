from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = PROJECT_ROOT / "data" / "cleaned" / "cleaned_data.csv"

TARGET_COL = "price_category"
TARGET_ORDER = ["Low", "Medium", "High"]
CATEGORY_PALETTE = {
    "Low": "#4C9BE8",
    "Medium": "#F5A623",
    "High": "#E84C4C",
}
POI_COUNT_COLS = [
    "school_count_within_3km",
    "hospital_count_within_3km",
    "supermarket_count_within_3km",
    "mall_count_within_3km",
    "transit_station_count_within_3km",
    "cafe_restaurant_count_within_3km",
]
FURNISHED_ORDER = ["Unfurnished", "Partly furnished", "Furnished", "Missing"]
FURNISHED_LABELS = {
    "no": "Unfurnished",
    "yes": "Furnished",
    "partly": "Partly furnished",
    "missing": "Missing",
}


sns.set_theme(style="whitegrid", font_scale=1.05)


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Cleaned data not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, low_memory=False)
    df[TARGET_COL] = pd.qcut(df["price_egp"], q=3, labels=TARGET_ORDER)
    df[TARGET_COL] = pd.Categorical(df[TARGET_COL], categories=TARGET_ORDER, ordered=True)

    furnished_raw = df["furnished"].fillna("Missing").astype(str).str.strip().str.lower()
    df["furnished_label"] = furnished_raw.map(FURNISHED_LABELS).fillna(
        furnished_raw.str.title()
    )

    return df


def format_egp(value: float) -> str:
    return f"EGP {value:,.0f}"


def make_price_distribution_figure(df: pd.DataFrame) -> plt.Figure:
    data = df["price_egp"].dropna()
    iqr = data.quantile(0.75) - data.quantile(0.25)
    bin_width = 2 * iqr / (len(data) ** (1 / 3)) if iqr > 0 else 1
    n_bins = max(10, int((data.max() - data.min()) / bin_width))

    fig, ax = plt.subplots(figsize=(11, 4.8))
    sns.histplot(
        data,
        bins=n_bins,
        kde=True,
        ax=ax,
        color="#2F80ED",
        edgecolor="white",
        linewidth=0.4,
    )
    ax.axvline(data.mean(), color="#D7263D", linestyle="--", linewidth=1.3, label="Mean")
    ax.axvline(data.median(), color="#1B998B", linestyle="-", linewidth=1.3, label="Median")
    ax.set_title("Price Distribution", fontweight="bold")
    ax.set_xlabel("Price (EGP)")
    ax.set_ylabel("Count")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def make_city_price_heatmap_figure(df: pd.DataFrame, top_n: int = 12) -> plt.Figure:
    city_counts = df["city"].value_counts().head(top_n).index
    city_share = pd.crosstab(
        df[df["city"].isin(city_counts)]["city"],
        df[df["city"].isin(city_counts)][TARGET_COL],
        normalize="index",
    ).reindex(columns=TARGET_ORDER)
    city_share = city_share.sort_values("High", ascending=False)

    fig, ax = plt.subplots(figsize=(9, max(4.5, len(city_share) * 0.42)))
    sns.heatmap(
        city_share,
        annot=True,
        fmt=".0%",
        cmap="YlOrRd",
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": "Share within city"},
    )
    ax.set_title("City vs. Price Tier", fontweight="bold")
    ax.set_xlabel("Price tier")
    ax.set_ylabel("City")
    fig.tight_layout()
    return fig


def make_furnished_price_figure(df: pd.DataFrame) -> plt.Figure:
    furnished_share = pd.crosstab(
        df["furnished_label"],
        df[TARGET_COL],
        normalize="index",
    ).reindex(index=FURNISHED_ORDER).reindex(columns=TARGET_ORDER)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    furnished_share.plot(
        kind="barh",
        stacked=True,
        ax=ax,
        color=[CATEGORY_PALETTE[tier] for tier in TARGET_ORDER],
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_title("Furnished Status vs. Price Tier", fontweight="bold")
    ax.set_xlabel("Share within furnished group")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(title="Price tier", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig


def make_city_gradient_figure(df: pd.DataFrame, top_n: int = 12) -> plt.Figure:
    city_share = pd.crosstab(df["city"], df[TARGET_COL], normalize="index").reindex(
        columns=TARGET_ORDER
    )
    city_share = city_share.assign(
        dominant_tier=city_share.idxmax(axis=1),
        high_share=city_share["High"],
    )
    city_share = city_share.sort_values("high_share", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 5.6))
    colors = city_share["dominant_tier"].map(CATEGORY_PALETTE)
    bars = ax.barh(city_share.index, city_share["high_share"], color=colors, edgecolor="white")

    for bar, (_, row) in zip(bars, city_share.iterrows()):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"L {row['Low']:.0%} | M {row['Medium']:.0%} | H {row['High']:.0%}",
            va="center",
            fontsize=9,
        )

    ax.set_title("City Price Gradient", fontweight="bold")
    ax.set_xlabel("High-tier share")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    fig.tight_layout()
    return fig


def make_poi_figure(df: pd.DataFrame) -> plt.Figure:
    poi_means = df.groupby(TARGET_COL, observed=True)[POI_COUNT_COLS].mean().reindex(TARGET_ORDER)
    short_names = [c.replace("_count_within_3km", "").replace("_", " ") for c in POI_COUNT_COLS]
    poi_means.columns = short_names

    fig, ax = plt.subplots(figsize=(11, 5.2))
    poi_means.T.plot(
        kind="bar",
        ax=ax,
        color=[CATEGORY_PALETTE[tier] for tier in TARGET_ORDER],
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_title("More Amenities = Lower Price", fontweight="bold")
    ax.set_xlabel("POI type")
    ax.set_ylabel("Mean count within 3km")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="Price tier", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig


def show() -> None:
    st.title("Market Overview")
    st.caption("Exploratory findings from the cleaned apartment listings dataset.")

    try:
        df = load_data()
    except FileNotFoundError as error:
        st.error(str(error))
        return

    total_listings = len(df)
    avg_price = df["price_egp"].mean()
    median_area = df["area_value"].median()
    top_city = df["city"].value_counts().idxmax()
    top_city_share = df["city"].value_counts(normalize=True).max()

    metrics = st.columns(4)
    metrics[0].metric("Total listings", f"{total_listings:,}")
    metrics[1].metric("Average price", format_egp(avg_price))
    metrics[2].metric("Median area", f"{median_area:.0f} m²")
    metrics[3].metric("Most common city", top_city, f"{top_city_share:.0%} of rows")

    st.markdown("### Price distribution")
    st.pyplot(make_price_distribution_figure(df), use_container_width=True)
    st.caption("The long right tail confirms that a small set of luxury listings pulls prices upward.")

    left, right = st.columns(2)
    with left:
        st.markdown("### City vs. price tier")
        st.pyplot(make_city_price_heatmap_figure(df), use_container_width=True)
    with right:
        st.markdown("### Furnished vs. price tier")
        st.pyplot(make_furnished_price_figure(df), use_container_width=True)

    st.markdown("### Geographic price gradient")
    st.pyplot(make_city_gradient_figure(df), use_container_width=True)
    st.info(
        "Coastal and resort cities skew toward High tier, while delta cities are overwhelmingly Low tier. "
        "Cairo and Giza sit in the middle because they mix all three segments."
    )

    st.markdown("### POI insight")
    st.pyplot(make_poi_figure(df), use_container_width=True)
    st.success(
        "Across every POI type, Low-tier listings have more amenities within 3km than High-tier listings. "
        "That is the counterintuitive but important pattern the model can exploit."
    )
