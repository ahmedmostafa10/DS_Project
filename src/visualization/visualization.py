from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway, chi2_contingency, shapiro, levene
from statsmodels.stats.multicomp import pairwise_tukeyhsd


# CONFIGURATION & CONSTANTS

DATA_PATH = Path("data/cleaned/cleaned_data.csv")
FIGURES_PATH = Path("reports/figures/EDA")
FIGURES_PATH.mkdir(parents=True, exist_ok=True)

TARGET_COL = "price_category"
TARGET_ORDER = ["Low", "Medium", "High"]

CONTINUOUS_FEATURES = ["price_egp", "area_value"]

ORDINAL_FEATURES = ["bedrooms", "bathroom"]

POI_DISTANCE_COLS = [
    "dist_nearest_school_km",
    "dist_nearest_hospital_km",
    "dist_nearest_supermarket_km",
    "dist_nearest_mall_km",
    "dist_nearest_transit_station_km",
    "dist_nearest_cafe_restaurant_km",
]

POI_COUNT_COLS = [
    "school_count_within_3km",
    "hospital_count_within_3km",
    "supermarket_count_within_3km",
    "mall_count_within_3km",
    "transit_station_count_within_3km",
    "cafe_restaurant_count_within_3km",
]

CATEGORICAL_FEATURES = [
    "city",
    "furnished",
    "completion_status",
    "listing_level",
    "is_premium",
]

# Plotting style
# Color palette
CATEGORY_PALETTE = {
    "Low": "#4C9BE8",  # blue
    "Medium": "#F5A623",  # amber
    "High": "#E84C4C",  # red
}

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "figure.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# Statistical threshold
ALPHA = 0.05

# Additional constants for multivariate analysis
ALL_NUMERIC = CONTINUOUS_FEATURES + POI_DISTANCE_COLS + POI_COUNT_COLS
CORR_COLS = CONTINUOUS_FEATURES + POI_DISTANCE_COLS + POI_COUNT_COLS
PARALLEL_FEATURES = [
    "area_value",
    "bedrooms",
    "bathrooms",
    "dist_nearest_school_km",
    "dist_nearest_hospital_km",
    "dist_nearest_transit_station_km",
    "school_count_within_3km",
    "transit_station_count_within_3km",
]


# DATA LOADING


def load_and_prepare_data(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load cleaned data and create price_category target variable.

    Args:
        data_path: Path to the cleaned CSV file.

    Returns:
        DataFrame with price_category column added.
    """
    df = pd.read_csv(data_path, low_memory=False)

    df[TARGET_COL] = pd.qcut(df["price_egp"], q=3, labels=TARGET_ORDER)

    # Enforce categorical order for target col (safe even if column already existed)
    df[TARGET_COL] = pd.Categorical(
        df[TARGET_COL],
        categories=TARGET_ORDER,
        ordered=True,
    )

    print(f"Shape: {df.shape}")
    print("\nTarget distribution:")
    print(df[TARGET_COL].value_counts(dropna=False).reindex(TARGET_ORDER))
    print(f"\nDtypes summary:\n{df.dtypes.value_counts()}")
    print(df.head(3))

    return df


# UNIVARIATE ANALYSIS


def plot_histogram_kde(
    df: pd.DataFrame,
    columns: list[str],
    n_cols: int = 2,
    figsize: tuple[int, int] = (14, 10),
    save_path: Path | None = None,
) -> None:
    """Plot histograms with KDE overlays for a list of numeric columns.

    Uses Freedman-Diaconis rule for bin width selection (from lecture):
        bin_width = 2 × IQR / n^(1/3)

    Args:
        df: Input DataFrame.
        columns: List of numeric column names to plot.
        n_cols: Number of subplot columns in the grid.
        figsize: Figure size tuple (width, height).
        save_path: If provided, saves the figure to this path.
    """
    n_rows = -(-len(columns) // n_cols)  # ceiling division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for i, col in enumerate(columns):
        ax = axes[i]
        data = df[col].dropna()

        # Freedman-Diaconis bin count
        iqr = data.quantile(0.75) - data.quantile(0.25)
        bin_width = 2 * iqr / (len(data) ** (1 / 3)) if iqr > 0 else 1
        n_bins = max(10, int((data.max() - data.min()) / bin_width))

        sns.histplot(
            data,
            bins=n_bins,
            kde=True,
            ax=ax,
            color="#4C9BE8",
            edgecolor="white",
            linewidth=0.4,
        )
        ax.set_title(f"Distribution of {col}", fontweight="bold")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")

        # Annotate with mean and median
        ax.axvline(
            data.mean(),
            color="red",
            linestyle="--",
            linewidth=1.2,
            label=f"Mean: {data.mean():.1f}",
        )
        ax.axvline(
            data.median(),
            color="green",
            linestyle="-",
            linewidth=1.2,
            label=f"Median: {data.median():.1f}",
        )
        ax.legend(fontsize=8)

    # Hide any unused subplots
    for j in range(len(columns), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(
        "Univariate Distributions — Numeric Features",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_box_violin_grid(
    df: pd.DataFrame,
    columns: list[str],
    figsize: tuple[int, int] = (14, 10),
    save_path: Path | None = None,
) -> None:
    """Plot side-by-side box and violin plots for each numeric column.

    Shows both summary statistics (box) and distribution shape (violin)
    per the lecture: "a fancy version of box plots that also shows the KDE shape."

    Args:
        df: Input DataFrame.
        columns: List of numeric columns to plot.
        figsize: Figure size.
        save_path: Optional save path.
    """
    n_cols_grid = 2
    n_rows = -(-len(columns) // n_cols_grid)
    fig, axes = plt.subplots(n_rows, n_cols_grid, figsize=figsize)
    axes = axes.flatten()

    for i, col in enumerate(columns):
        ax = axes[i]
        data = df[[col]].dropna()

        # Simpler approach: violin with inner box
        sns.violinplot(y=data[col], ax=ax, color="#4C9BE8", inner="box", linewidth=1.2)
        ax.set_title(f"{col} — Violin + Box", fontweight="bold")
        ax.set_ylabel(col)

    for j in range(len(columns), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(
        "Spread & Outlier Visualization — Numeric Features",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_categorical_bar_charts(
    df: pd.DataFrame,
    columns: list[str],
    figsize: tuple[int, int] = (16, 12),
    top_n: int = 15,
    save_path: Path | None = None,
) -> None:
    """Plot bar charts showing relative frequency for categorical features.

    Uses relative frequency (proportion) rather than raw count so that
    different-cardinality features are visually comparable.

    Args:
        df: Input DataFrame.
        columns: Categorical column names to plot.
        figsize: Figure size.
        top_n: For high-cardinality columns, show only the top N categories.
        save_path: Optional save path.
    """
    n_cols_grid = 2
    n_rows = -(-len(columns) // n_cols_grid)
    fig, axes = plt.subplots(n_rows, n_cols_grid, figsize=figsize)
    axes = axes.flatten()

    for i, col in enumerate(columns):
        ax = axes[i]
        freq = df[col].value_counts(normalize=True).head(top_n)

        # Use matplotlib barh with a seaborn colormap to avoid seaborn palette
        # deprecation when no `hue` is provided.
        colors = sns.color_palette("Blues_d", n_colors=len(freq))
        ax.barh(range(len(freq)), freq.values, color=colors)
        ax.set_yticks(range(len(freq)))
        ax.set_yticklabels(freq.index)
        ax.invert_yaxis()
        ax.set_title(f"{col} — Relative Frequency", fontweight="bold")
        ax.set_xlabel("Proportion")
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

        # Annotate bars with percentage
        for bar, val in zip(ax.patches, freq.values):
            ax.text(
                val + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}",
                va="center",
                fontsize=8,
            )

    for j in range(len(columns), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(
        "Categorical Feature Distributions", fontsize=14, fontweight="bold", y=1.01
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_target_distribution(
    df: pd.DataFrame,
    target_col: str,
    order: list[str],
    palette: dict[str, str],
    save_path: Path | None = None,
) -> None:
    """Plot bar chart and pie chart for the target variable distribution side by side.

    Args:
        df: Input DataFrame.
        target_col: Name of the target column.
        order: Ordered list of category labels.
        palette: Dict mapping category label to hex color.
        save_path: Optional save path.
    """
    counts = df[target_col].value_counts().reindex(order)
    colors = [palette[c] for c in order]

    fig, (ax_bar, ax_pie) = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart
    ax_bar.bar(order, counts.values, color=colors, edgecolor="white", linewidth=0.6)
    ax_bar.set_title("Price Category — Count", fontweight="bold")
    ax_bar.set_ylabel("Number of Listings")
    for bar, count in zip(ax_bar.patches, counts.values):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 50,
            f"{count:,}\n({count / len(df):.1%})",
            ha="center",
            fontsize=9,
        )

    # Pie chart
    ax_pie.pie(
        counts.values,
        labels=order,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    ax_pie.set_title("Price Category — Share", fontweight="bold")

    plt.suptitle("Target Variable: price_category", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def compute_univariate_summary(
    df: pd.DataFrame,
    numeric_cols: list[str],
) -> pd.DataFrame:
    """Compute a descriptive statistics summary for numeric features.

    Includes: count, mean, median, std, skewness, kurtosis, min, max.

    Args:
        df: Input DataFrame.
        numeric_cols: List of numeric column names.

    Returns:
        DataFrame with one row per feature and summary statistic columns.
    """
    rows = []
    for col in numeric_cols:
        s = df[col].dropna()
        rows.append(
            {
                "feature": col,
                "count": int(s.count()),
                "mean": s.mean(),
                "median": s.median(),
                "std": s.std(),
                "skewness": s.skew(),
                "kurtosis": s.kurt(),
                "min": s.min(),
                "max": s.max(),
            }
        )
    return pd.DataFrame(rows).set_index("feature").round(3)


# BIVARIATE ANALYSIS: FEATURE vs. TARGET


def plot_continuous_vs_target_boxplots(
    df: pd.DataFrame,
    features: list[str],
    target_col: str,
    order: list[str],
    palette: dict[str, str],
    figsize: tuple[int, int] = (16, 12),
    save_path: Path | None = None,
) -> None:
    """Plot box plots for continuous features grouped by price category.

    This is the primary feature-to-target analysis for continuous variables,
    showing median, IQR, and outliers per price tier.

    Args:
        df: Input DataFrame.
        features: Continuous feature column names.
        target_col: Target column name.
        order: Ordered list of target category labels.
        palette: Dict mapping category to color.
        figsize: Figure size.
        save_path: Optional save path.
    """
    n_cols_grid = 2
    n_rows = -(-len(features) // n_cols_grid)
    fig, axes = plt.subplots(n_rows, n_cols_grid, figsize=figsize)
    axes = axes.flatten()

    for i, feat in enumerate(features):
        ax = axes[i]
        # Use hue=target_col with dodge=False to keep same visual grouping
        sns.boxplot(
            data=df,
            x=target_col,
            y=feat,
            order=order,
            hue=target_col,
            hue_order=order,
            palette=palette,
            dodge=False,
            ax=ax,
            linewidth=1.2,
            flierprops={"marker": "o", "markersize": 2, "alpha": 0.4},
        )
        # Remove duplicated legend created by using hue
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        ax.set_title(f"{feat} by Price Category", fontweight="bold")
        ax.set_xlabel("Price Category")

        # Annotate with median value per group
        for j, cat in enumerate(order):
            median_val = df[df[target_col] == cat][feat].median()
            ax.text(
                j,
                median_val,
                f"{median_val:.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="black",
                fontweight="bold",
            )

    for k in range(len(features), len(axes)):
        axes[k].set_visible(False)

    plt.suptitle(
        "Continuous Features vs Price Category (Box Plots)",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_strip_plots_poi(
    df: pd.DataFrame,
    poi_cols: list[str],
    target_col: str,
    order: list[str],
    palette: dict[str, str],
    figsize: tuple[int, int] = (16, 14),
    save_path: Path | None = None,
) -> None:
    """Plot strip plots for POI distance features grouped by price category.

    Strip plots are ideal for bounded continuous features where you want to
    see the actual point distribution, not just summary statistics.

    Args:
        df: Input DataFrame.
        poi_cols: POI distance column names.
        target_col: Target column name.
        order: Category order.
        palette: Color palette dict.
        figsize: Figure size.
        save_path: Optional save path.
    """
    n_cols_grid = 2
    n_rows = -(-len(poi_cols) // n_cols_grid)
    fig, axes = plt.subplots(n_rows, n_cols_grid, figsize=figsize)
    axes = axes.flatten()

    for i, col in enumerate(poi_cols):
        ax = axes[i]
        # Use a sample for performance — strip plots with 30k points are slow
        sample = df.sample(n=min(3000, len(df)), random_state=42)
        # Color by hue to avoid passing palette without hue (deprecated).
        sns.stripplot(
            data=sample,
            x=target_col,
            y=col,
            order=order,
            hue=target_col,
            hue_order=order,
            palette=palette,
            dodge=False,
            ax=ax,
            alpha=0.3,
            size=2.5,
            jitter=True,
        )
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        # Overlay mean per group as a horizontal line
        for j, cat in enumerate(order):
            mean_val = df[df[target_col] == cat][col].mean()
            ax.hlines(
                mean_val,
                j - 0.3,
                j + 0.3,
                color="black",
                linewidth=2,
                label="mean" if j == 0 else "",
            )
        short_name = col.replace("dist_nearest_", "").replace("_km", "")
        ax.set_title(f"dist to {short_name}", fontweight="bold")
        ax.set_xlabel("Price Category")
        ax.set_ylabel("Distance (km)")

    for k in range(len(poi_cols), len(axes)):
        axes[k].set_visible(False)

    plt.suptitle(
        "POI Distances vs Price Category (Strip Plots)",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_stacked_and_clustered_bars(
    df: pd.DataFrame,
    feature: str,
    target_col: str,
    order: list[str],
    palette: dict[str, str],
    top_n: int = 10,
    save_path: Path | None = None,
) -> None:
    """Plot stacked and clustered bar charts for a categorical feature vs target.

    Stacked: easier to read tier proportions within each category.
    Clustered: easier to compare absolute counts across tiers.

    Args:
        df: Input DataFrame.
        feature: Categorical feature column name.
        target_col: Target column name.
        order: Ordered list of target labels.
        palette: Color palette dict.
        top_n: Show only the top N categories by listing count.
        save_path: Optional save path.
    """
    top_cats = df[feature].value_counts().head(top_n).index
    plot_df = df[df[feature].isin(top_cats)]

    # Build crosstab normalized by row (proportion per feature value)
    ct_norm = pd.crosstab(plot_df[feature], plot_df[target_col], normalize="index")[
        order
    ]
    ct_abs = pd.crosstab(plot_df[feature], plot_df[target_col])[order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # Stacked bar
    ct_norm.plot(
        kind="bar",
        stacked=True,
        ax=ax1,
        color=[palette[c] for c in order],
        edgecolor="white",
        linewidth=0.5,
    )
    ax1.set_title(f"{feature} — Stacked (Proportions)", fontweight="bold")
    ax1.set_xlabel(feature)
    ax1.set_ylabel("Proportion")
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax1.legend(title="Price Category", bbox_to_anchor=(1, 1))
    ax1.tick_params(axis="x", rotation=45)

    # Clustered bar
    ct_abs.plot(
        kind="bar",
        stacked=False,
        ax=ax2,
        color=[palette[c] for c in order],
        edgecolor="white",
        linewidth=0.5,
    )
    ax2.set_title(f"{feature} — Clustered (Counts)", fontweight="bold")
    ax2.set_xlabel(feature)
    ax2.set_ylabel("Count")
    ax2.legend(title="Price Category", bbox_to_anchor=(1, 1))
    ax2.tick_params(axis="x", rotation=45)

    plt.suptitle(f"{feature} vs Price Category", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_crosstab_heatmap(
    df: pd.DataFrame,
    feature: str,
    target_col: str,
    order: list[str],
    normalize: str = "index",
    top_n: int = 15,
    save_path: Path | None = None,
) -> None:
    """Plot a heatmap of a crosstab between a categorical feature and the target.

    Args:
        df: Input DataFrame.
        feature: Categorical feature column name.
        target_col: Target column name.
        order: Target category order.
        normalize: How to normalize: 'index' (by row), 'columns', or None.
        top_n: Show top N categories by count.
        save_path: Optional save path.
    """
    top_cats = df[feature].value_counts().head(top_n).index
    ct = pd.crosstab(
        df[df[feature].isin(top_cats)][feature],
        df[df[feature].isin(top_cats)][target_col],
        normalize=normalize,
    )[order]

    fig, ax = plt.subplots(figsize=(8, max(4, len(ct) * 0.4)))
    sns.heatmap(
        ct,
        annot=True,
        fmt=".1%",
        cmap="YlOrRd",
        ax=ax,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Proportion within row"},
    )
    ax.set_title(f"{feature} × Price Category (row-normalized)", fontweight="bold")
    ax.set_xlabel("Price Category")
    ax.set_ylabel(feature)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


# BIVARIATE ANALYSIS: FEATURE vs. FEATURE


def plot_scatter_with_regression(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: str,
    order: list[str],
    palette: dict[str, str],
    sample_n: int = 5000,
    save_path: Path | None = None,
) -> None:
    """Plot a scatter plot with regression line, colored by price category.

    Uses a sample for performance while preserving proportional representation
    across all three tiers.

    Args:
        df: Input DataFrame.
        x_col: X-axis column name.
        y_col: Y-axis column name.
        hue_col: Column to use for color (target).
        order: Category order for legend.
        palette: Color palette dict.
        sample_n: Max number of points to plot.
        save_path: Optional save path.
    """
    # Stratified sample: equal proportion from each tier
    sample_parts = []
    for _, group in df.groupby(hue_col, observed=True):
        sample_parts.append(
            group.sample(n=min(sample_n // 3, len(group)), random_state=42)
        )
    sample = pd.concat(sample_parts, ignore_index=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(
        data=sample,
        x=x_col,
        y=y_col,
        hue=hue_col,
        hue_order=order,
        palette=palette,
        alpha=0.5,
        s=20,
        ax=ax,
        linewidth=0,
    )

    # Overall regression line (ignoring category)
    sns.regplot(
        data=sample,
        x=x_col,
        y=y_col,
        scatter=False,
        ax=ax,
        color="black",
        line_kws={"linewidth": 1.5, "linestyle": "--"},
        label="OLS trend",
    )

    ax.set_title(f"{x_col} vs {y_col} — colored by Price Category", fontweight="bold")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x / 1e6:.1f}M"))
    ax.legend(title="Price Category")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_hexbin_density(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    gridsize: int = 40,
    save_path: Path | None = None,
) -> None:
    """Plot a hexbin 2D density chart for two continuous variables.

    Hexbin is preferred over scatter when the number of points is large
    and overplotting obscures the density pattern.

    Args:
        df: Input DataFrame.
        x_col: X-axis column name.
        y_col: Y-axis column name.
        gridsize: Resolution of hexagonal bins.
        save_path: Optional save path.
    """
    clean = df[[x_col, y_col]].dropna()

    fig, ax = plt.subplots(figsize=(9, 6))
    hb = ax.hexbin(
        clean[x_col], clean[y_col], gridsize=gridsize, cmap="YlOrRd", mincnt=1
    )
    plt.colorbar(hb, ax=ax, label="Count per bin")
    ax.set_title(f"2D Density: {x_col} vs {y_col}", fontweight="bold")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x / 1e6:.1f}M"))
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_two_categorical_heatmap(
    df: pd.DataFrame,
    row_col: str,
    col_col: str,
    normalize: str = "index",
    save_path: Path | None = None,
) -> None:
    """Plot a heatmap of the crosstab between two categorical features.

    Args:
        df: Input DataFrame.
        row_col: Row variable column name.
        col_col: Column variable column name.
        normalize: Normalization direction: 'index', 'columns', or None.
        save_path: Optional save path.
    """
    ct = pd.crosstab(df[row_col], df[col_col], normalize=normalize)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        ct,
        annot=True,
        fmt=".1%",
        cmap="Blues",
        ax=ax,
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_title(f"{row_col} × {col_col}", fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_ylabel(row_col, labelpad=18)
    ax.tick_params(axis="y", pad=6)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


# MULTIVARIATE ANALYSIS


def plot_spearman_correlation_matrix(
    df: pd.DataFrame,
    columns: list[str],
    figsize: tuple[int, int] = (14, 12),
    save_path: Path | None = None,
) -> pd.DataFrame:
    """Compute and plot the Spearman correlation matrix for numeric features.

    Spearman (rank-based) is used because:
    - price_egp and POI features are skewed
    - It captures monotonic (not just linear) relationships
    - It is robust to the outliers that remain after capping

    Note: This matches the Phase 1 validation methodology.

    Args:
        df: Input DataFrame.
        columns: Numeric columns to include.
        figsize: Figure size.
        save_path: Optional save path.

    Returns:
        Spearman correlation matrix as a DataFrame.
    """
    valid_cols = [c for c in columns if c in df.columns]
    corr = df[valid_cols].corr(method="spearman")

    mask = np.triu(np.ones_like(corr, dtype=bool))  # upper triangle mask

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        square=True,
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Spearman Correlation Matrix — Numeric Features", fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    return corr


def plot_pairplot_structural(
    df: pd.DataFrame,
    features: list[str],
    target_col: str,
    order: list[str],
    palette: dict[str, str],
    sample_n: int = 2000,
    save_path: Path | None = None,
) -> None:
    """Plot a pair plot for structural features colored by price category.

    Uses a stratified sample for performance.

    Args:
        df: Input DataFrame.
        features: Features to include in the pair plot.
        target_col: Hue variable.
        order: Category order.
        palette: Color palette dict.
        sample_n: Max points to include.
        save_path: Optional save path.
    """
    sample_parts = []
    for _, group in df.groupby(target_col, observed=True):
        sample_parts.append(
            group.sample(n=min(sample_n // 3, len(group)), random_state=42)
        )
    sample = pd.concat(sample_parts, ignore_index=True)

    g = sns.pairplot(
        sample[features + [target_col]],
        hue=target_col,
        hue_order=order,
        palette=palette,
        diag_kind="kde",
        plot_kws={"alpha": 0.4, "s": 15},
        diag_kws={"linewidth": 1.5},
    )
    g.fig.suptitle(
        "Pair Plot — Structural Features by Price Category",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    if save_path:
        g.fig.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_parallel_coordinates(
    df: pd.DataFrame,
    features: list[str],
    target_col: str,
    order: list[str],
    palette: dict[str, str],
    sample_n: int = 600,
    save_path: Path | None = None,
) -> None:
    """Plot parallel coordinates for multiple features colored by price category.

    Each listing is a line crossing all feature axes. Bundles of similar-colored
    lines reveal tier-specific patterns across all features simultaneously.

    Features are min-max normalized before plotting so all axes are on [0,1].

    Args:
        df: Input DataFrame.
        features: Features to plot as parallel axes.
        target_col: Color variable.
        order: Category order.
        palette: Color palette dict.
        sample_n: Max points per category.
        save_path: Optional save path.
    """
    from matplotlib.lines import Line2D

    sample_parts = []
    for _, group in df.groupby(target_col, observed=True):
        sample_parts.append(
            group.sample(n=min(sample_n // 3, len(group)), random_state=42)
        )
    sample = pd.concat(sample_parts, ignore_index=True)

    valid_features = [f for f in features if f in df.columns]
    plot_data = sample[valid_features + [target_col]].dropna()

    # Min-max normalize each feature to [0, 1] for comparability
    normalized = plot_data[valid_features].copy()
    for col in valid_features:
        col_min, col_max = normalized[col].min(), normalized[col].max()
        if col_max > col_min:
            normalized[col] = (normalized[col] - col_min) / (col_max - col_min)

    fig, ax = plt.subplots(figsize=(16, 6))
    x_positions = range(len(valid_features))

    for _, row in normalized.iterrows():
        cat = plot_data.loc[row.name, target_col]
        color = palette.get(str(cat), "#999999")
        ax.plot(
            x_positions,
            row[valid_features].values,
            color=color,
            alpha=0.15,
            linewidth=0.8,
        )

    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(valid_features, rotation=45, ha="right")
    ax.set_ylabel("Normalized Value [0–1]")
    ax.set_title(
        "Parallel Coordinates — All Key Features by Price Tier", fontweight="bold"
    )

    legend_elements = [
        Line2D([0], [0], color=palette[c], linewidth=2, label=c) for c in order
    ]
    ax.legend(handles=legend_elements, title="Price Category", bbox_to_anchor=(1, 1))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_bubble_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    size_col: str,
    hue_col: str,
    order: list[str],
    palette: dict[str, str],
    sample_n: int = 800,
    save_path: Path | None = None,
) -> None:
    """Plot a bubble chart: x vs y, bubble size = third variable, color = category.

    Args:
        df: Input DataFrame.
        x_col: X-axis continuous feature.
        y_col: Y-axis continuous feature.
        size_col: Feature controlling bubble size.
        hue_col: Target column for color.
        order: Category order.
        palette: Color palette dict.
        sample_n: Max total points.
        save_path: Optional save path.
    """
    sample_parts = []
    for _, group in df.groupby(hue_col, observed=True):
        sample_parts.append(
            group.sample(n=min(sample_n // 3, len(group)), random_state=42)
        )
    sample = pd.concat(sample_parts, ignore_index=True)

    # Normalize size to [20, 400] for visibility
    s_min, s_max = sample[size_col].min(), sample[size_col].max()
    bubble_sizes = 20 + 380 * (sample[size_col] - s_min) / max(s_max - s_min, 1)

    fig, ax = plt.subplots(figsize=(11, 7))
    for cat in order:
        mask = sample[hue_col] == cat
        ax.scatter(
            sample.loc[mask, x_col],
            sample.loc[mask, y_col],
            s=bubble_sizes[mask],
            color=palette[cat],
            alpha=0.5,
            label=cat,
            edgecolors="white",
            linewidth=0.3,
        )

    ax.set_title(f"{x_col} vs {y_col} | Bubble size = {size_col}", fontweight="bold")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v / 1e6:.1f}M"))
    ax.legend(title="Price Category")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_poi_count_comparison(
    df: pd.DataFrame,
    count_cols: list[str],
    target_col: str,
    order: list[str],
    palette: dict[str, str],
    save_path: Path | None = None,
) -> None:
    """Plot mean POI counts per price tier as a grouped bar chart.

    Args:
        df: Input DataFrame.
        count_cols: POI count column names.
        target_col: Target column.
        order: Category order.
        palette: Color palette.
        save_path: Optional save path.
    """
    means = df.groupby(target_col, observed=True)[count_cols].mean().reindex(order)
    short_names = [
        c.replace("_count_within_3km", "").replace("_", " ") for c in count_cols
    ]
    means.columns = short_names

    ax = means.T.plot(
        kind="bar",
        figsize=(13, 5),
        color=[palette[c] for c in order],
        edgecolor="white",
        linewidth=0.4,
    )
    ax.set_title("Mean POI Counts Within 3km by Price Tier", fontweight="bold")
    ax.set_xlabel("POI Type")
    ax.set_ylabel("Mean Count")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="Price Category")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


# STATISTICAL TESTS


def run_one_way_anova(
    df: pd.DataFrame,
    numeric_col: str,
    group_col: str,
    alpha: float = ALPHA,
    min_group_size: int = 30,
) -> dict:
    """Run one-way ANOVA to test if a numeric variable's mean differs across groups.

    Pre-conditions checked (from lecture):
    1. Normality per group (Shapiro-Wilk) — for groups with n in [20, 2000]
    2. Homogeneity of variance (Levene's test)

    Args:
        df: Input DataFrame.
        numeric_col: Continuous dependent variable.
        group_col: Categorical grouping variable.
        alpha: Significance level (default 0.05).
        min_group_size: Minimum group size to include in test.

    Returns:
        Dict with F-statistic, p-value, and interpretation.
    """
    groups = {
        cat: grp[numeric_col].dropna().values
        for cat, grp in df.groupby(group_col)
        if len(grp) >= min_group_size
    }

    if len(groups) < 2:
        return {"error": "Fewer than 2 qualifying groups"}

    print(f"\n--- One-Way ANOVA: {numeric_col} across {group_col} ---")
    print(f"Groups included (n >= {min_group_size}): {list(groups.keys())}")

    # Step 1: Normality check (Shapiro-Wilk)
    print("\n1. Normality (Shapiro-Wilk):")
    for cat, vals in groups.items():
        sample = vals[:2000] if len(vals) > 2000 else vals
        stat, p = shapiro(sample)
        ok = p > alpha
        print(f"   {cat}: W={stat:.4f}, p={p:.4f} {'✓' if ok else '✗'}")
        # If any group fails normality, it's reported above; no flag retained.

    # Step 2: Homogeneity of variance (Levene's test)
    lev_stat, lev_p = levene(*groups.values())
    variance_ok = lev_p > alpha
    print(
        f"\n2. Levene's Test: stat={lev_stat:.4f}, p={lev_p:.4f} "
        f"{'✓ Equal variances' if variance_ok else '✗ Unequal variances'}"
    )

    # Step 3: ANOVA
    f_stat, p_val = f_oneway(*groups.values())
    reject = p_val < alpha
    print("\n3. One-Way ANOVA:")
    print(f"   F-statistic: {f_stat:.4f}")
    print(f"   p-value:     {p_val:.6f}")
    print(f"   Result: {'REJECT H₀' if reject else 'FAIL TO REJECT H₀'} (α={alpha})")

    # Step 4: Post-hoc Tukey HSD if significant
    if reject:
        print("\n4. Post-Hoc Tukey HSD:")
        all_vals = np.concatenate(list(groups.values()))
        all_labels = np.concatenate([[cat] * len(v) for cat, v in groups.items()])
        tukey = pairwise_tukeyhsd(all_vals, all_labels, alpha=alpha)
        print(tukey.summary())

    return {"F": f_stat, "p": p_val, "reject": reject}


def run_chi_square_independence(
    df: pd.DataFrame,
    row_col: str,
    col_col: str,
    alpha: float = ALPHA,
) -> dict:
    """Run Chi-Square test of independence between two categorical variables.

    Also computes Cramér's V for effect size.

    Args:
        df: Input DataFrame.
        row_col: Row categorical variable.
        col_col: Column categorical variable.
        alpha: Significance level.

    Returns:
        Dict with chi2, p-value, Cramér's V, and interpretation.
    """
    ct = pd.crosstab(df[row_col], df[col_col])
    chi2, p, dof, expected = chi2_contingency(ct)
    reject = p < alpha

    n = ct.values.sum()
    min_dim = min(ct.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else np.nan

    if cramers_v < 0.1:
        effect = "negligible"
    elif cramers_v < 0.3:
        effect = "small"
    elif cramers_v < 0.5:
        effect = "medium"
    else:
        effect = "large"

    print(f"\n--- Chi-Square Test: {row_col} × {col_col} ---")
    print(f"Chi² statistic: {chi2:.4f}")
    print(f"Degrees of freedom: {dof}")
    print(f"p-value: {p:.6f}")
    print(f"Result: {'REJECT H₀' if reject else 'FAIL TO REJECT H₀'} (α={alpha})")
    print(f"Cramér's V: {cramers_v:.4f} → {effect} association")

    # Visualize: bar chart of conditional proportions
    ct_norm = pd.crosstab(df[row_col], df[col_col], normalize="index").reindex(
        columns=TARGET_ORDER, fill_value=0
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    ct_norm.plot(
        kind="bar",
        stacked=False,
        ax=ax,
        color=[CATEGORY_PALETTE.get(c, "#999") for c in TARGET_ORDER],
        edgecolor="white",
    )
    ax.set_title(
        f"{row_col} vs {col_col} — Conditional Proportions\n"
        f"Chi²={chi2:.2f}, p={p:.4f}, Cramér's V={cramers_v:.3f}",
        fontweight="bold",
    )
    ax.set_ylabel("Proportion")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="Price Category")
    plt.tight_layout()
    plt.savefig(
        FIGURES_PATH / f"21_chi2_{row_col}_vs_{col_col}.png",
        bbox_inches="tight",
    )
    plt.close()

    return {"chi2": chi2, "p": p, "cramers_v": cramers_v, "effect": effect}


def rank_features_by_target_correlation(
    df: pd.DataFrame,
    numeric_features: list[str],
    target_col: str,
    order: list[str],
) -> pd.DataFrame:
    """Rank numeric features by Spearman correlation with the encoded target.

    Encodes price_category ordinally (Low=0, Medium=1, High=2) for correlation.

    Args:
        df: Input DataFrame.
        numeric_features: Numeric features to rank.
        target_col: Target column (categorical).
        order: Category order for encoding.

    Returns:
        DataFrame with features ranked by |correlation| descending.
    """
    target_encoded = df[target_col].map({c: i for i, c in enumerate(order)})
    valid_feats = [f for f in numeric_features if f in df.columns]

    rows = []
    for feat in valid_feats:
        corr, p_val = stats.spearmanr(
            df[feat].fillna(df[feat].median()),
            target_encoded,
            nan_policy="omit",
        )
        rows.append(
            {
                "feature": feat,
                "spearman_r": corr,
                "abs_r": abs(corr),
                "p_value": p_val,
                "significant": p_val < ALPHA,
            }
        )

    return (
        pd.DataFrame(rows).sort_values("abs_r", ascending=False).reset_index(drop=True)
    )


# DASHBOARD


def build_eda_dashboard(
    df: pd.DataFrame,
    feature_ranking: pd.DataFrame,
    target_col: str,
    order: list[str],
    palette: dict[str, str],
    poi_count_cols: list[str],
    save_path: Path | None = None,
) -> None:
    """Build and save the updated EDA summary dashboard.

    Final tweaks: increase vertical spacing between rows and title, keep proportions.
    """
    fig = plt.figure(figsize=(28, 35))
    fig.patch.set_facecolor("white")

    # increase vertical spacing between rows
    gs = fig.add_gridspec(
        3, 3, hspace=0.28, wspace=0.28, height_ratios=[1.1, 0.85, 1.15]
    )

    # ROW 1 (now): Area | Histogram | Box Plot
    ax_area = fig.add_subplot(gs[0, 0])
    ax_hist = fig.add_subplot(gs[0, 1])
    ax_box = fig.add_subplot(gs[0, 2])

    # ROW 2 (now): Price tier (left, spans 2 cols) | Feature importance (right, small)
    ax_price_city = fig.add_subplot(gs[1, 0])
    ax_feat = fig.add_subplot(gs[1, 1:3])

    # ROW 3: Furnished (left, spans 2 cols) | Mean POI counts (right)
    ax_furnished = fig.add_subplot(gs[2, 0])
    ax_poi = fig.add_subplot(gs[2, 1:3])

    colors = [palette[c] for c in order]

    # Panel: Area by Price Tier (Violin)
    # Use hue to assign colors per category and remove legend afterwards
    sns.violinplot(
        data=df,
        x=target_col,
        y="area_value",
        order=order,
        hue=target_col,
        hue_order=order,
        palette=palette,
        ax=ax_area,
        inner="box",
        linewidth=1.6,
        cut=0,
        dodge=False,
    )
    if ax_area.get_legend() is not None:
        ax_area.get_legend().remove()
    ax_area.set_title("Area (sqm) by Price Tier", fontweight="bold", fontsize=34)
    ax_area.set_xlabel("")
    ax_area.tick_params(axis="both", labelsize=28)

    # Panel: Price distribution histogram
    data = df["price_egp"].dropna()
    iqr = data.quantile(0.75) - data.quantile(0.25)
    bin_width = 2 * iqr / (len(data) ** (1 / 3)) if iqr > 0 else 1
    n_bins = max(10, int((data.max() - data.min()) / bin_width))
    sns.histplot(
        data,
        bins=n_bins,
        kde=True,
        ax=ax_hist,
        color="#4C9BE8",
        edgecolor="white",
        linewidth=0.6,
    )
    ax_hist.set_title("Distribution of price_egp", fontweight="bold", fontsize=34)
    ax_hist.set_xlabel("price_egp", fontsize=30)
    ax_hist.set_ylabel("Count", fontsize=30)
    ax_hist.axvline(
        data.mean(),
        color="red",
        linestyle="--",
        linewidth=1.8,
        label=f"Mean: {data.mean():.0f}",
    )
    ax_hist.axvline(
        data.median(),
        color="green",
        linestyle="-",
        linewidth=1.8,
        label=f"Median: {data.median():.0f}",
    )
    ax_hist.legend(fontsize=26)
    ax_hist.tick_params(axis="both", labelsize=26)

    # Panel: Price by Price Category (Box Plot)
    sns.boxplot(
        data=df,
        x=target_col,
        y="price_egp",
        order=order,
        hue=target_col,
        hue_order=order,
        palette=palette,
        dodge=False,
        ax=ax_box,
        linewidth=1.6,
        flierprops={"marker": "o", "markersize": 6, "alpha": 0.45},
    )
    if ax_box.get_legend() is not None:
        ax_box.get_legend().remove()
    ax_box.set_title("price_egp by\nPrice Category", fontweight="bold", fontsize=34)
    ax_box.set_xlabel("Price Category", fontsize=30)
    ax_box.tick_params(axis="both", labelsize=28)
    for j, cat in enumerate(order):
        median_val = df[df[target_col] == cat]["price_egp"].median()
        ax_box.text(
            j,
            median_val,
            f"{median_val / 1e6:.1f}M",
            ha="center",
            va="bottom",
            fontsize=24,
            color="black",
            fontweight="bold",
        )

    # Panel: Price Tier by City (left, wider)
    ct_city = pd.crosstab(df["city"], df[target_col], normalize="index")[order]
    ct_city = ct_city.sort_values("High", ascending=True).tail(10)
    ct_city.plot(
        kind="barh",
        stacked=True,
        ax=ax_price_city,
        color=colors,
        edgecolor="white",
        linewidth=0.45,
        legend=False,
    )
    ax_price_city.set_title(
        "Price Tier by City\n(Top 10)", fontweight="bold", fontsize=32
    )
    ax_price_city.set_xlabel("Proportion", fontsize=30)
    ax_price_city.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax_price_city.tick_params(axis="both", labelsize=26)

    # Panel: Feature Importance (right, smaller area)
    feat_ranking_filtered = feature_ranking[feature_ranking["feature"] != "price_egp"]
    top10 = feat_ranking_filtered.head(10)
    bar_colors = ["#E84C4C" if sig else "#AAAAAA" for sig in top10["significant"]]
    ax_feat.barh(
        top10["feature"][::-1],
        top10["abs_r"][::-1],
        color=bar_colors[::-1],
        height=0.28,
    )
    ax_feat.set_title(
        "Feature Importance — |Spearman ρ| (red = p<0.05)",
        fontweight="bold",
        fontsize=20,
    )
    ax_feat.set_xlabel("|Spearman ρ|", fontsize=18)
    ax_feat.axvline(0.1, color="gray", linestyle="--", linewidth=1.3)
    ax_feat.tick_params(axis="y", labelsize=14)
    ax_feat.tick_params(axis="x", labelsize=16)

    # Panel: Furnished vs Price Category (left bottom, wider)
    feat = "furnished"
    top_cats = df[feat].value_counts().head(10).index
    plot_df = df[df[feat].isin(top_cats)]
    ct_norm = pd.crosstab(plot_df[feat], plot_df[target_col], normalize="index")[order]
    ct_norm.plot(
        kind="bar",
        stacked=True,
        ax=ax_furnished,
        color=[palette[c] for c in order],
        edgecolor="white",
        linewidth=0.6,
        legend=False,
        width=0.72,
    )
    ax_furnished.set_title(
        "Furnished vs Price Category (Proportions)", fontweight="bold", fontsize=34
    )
    ax_furnished.set_xlabel("Furnished Status", fontsize=30)
    ax_furnished.set_ylabel("Proportion", fontsize=30)
    ax_furnished.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax_furnished.tick_params(axis="x", rotation=28, labelsize=26)
    ax_furnished.tick_params(axis="y", labelsize=26)

    # Panel: Mean POI Counts (right bottom)
    means = df.groupby(target_col, observed=True)[poi_count_cols].mean().reindex(order)
    short_names = [
        c.replace("_count_within_3km", "").replace("_", " ").title()
        for c in poi_count_cols
    ]
    means.columns = short_names
    means.T.plot(
        kind="bar",
        ax=ax_poi,
        color=[palette[c] for c in order],
        edgecolor="white",
        linewidth=0.9,
        width=0.75,
    )
    ax_poi.set_title(
        "Mean POI Counts\nWithin 3km by Tier", fontweight="bold", fontsize=32
    )
    ax_poi.set_xlabel("POI Type", fontsize=30)
    ax_poi.set_ylabel("Mean Count", fontsize=30)
    ax_poi.tick_params(axis="x", rotation=25, labelsize=26)
    ax_poi.tick_params(axis="y", labelsize=26)
    ax_poi.legend(title="Price Tier", fontsize=22, title_fontsize=22, loc="upper left")
    ax_poi.grid(axis="y", alpha=0.28, linestyle="--")

    # increase top margin so title does not overlap
    fig.subplots_adjust(top=0.88)
    fig.suptitle(
        "Real Estate Price Prediction - EDA Dashboard\n"
        "Team 10 | CMP 2026 | Data Science Project",
        fontsize=36,
        fontweight="bold",
        y=0.98,
    )

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()


# MAIN EXECUTION FUNCTION


def run_eda() -> None:
    """
    Execute all EDA steps in order.
    """

    print("STARTING AUTOMATED EDA\n")

    # Load and prepare data
    print("\n[1] Loading and preparing data...")
    df = load_and_prepare_data()

    # === UNIVARIATE ANALYSIS ===
    print("\n[2] Univariate Analysis: Histograms with KDE...")
    plot_histogram_kde(
        df,
        columns=[*CONTINUOUS_FEATURES, *ORDINAL_FEATURES],
        save_path=FIGURES_PATH / "01_histograms_numeric.png",
    )

    print("\n[3] Univariate Analysis: Box and Violin Plots...")
    plot_box_violin_grid(
        df,
        columns=CONTINUOUS_FEATURES
        + ["dist_nearest_school_km", "dist_nearest_transit_station_km"],
        save_path=FIGURES_PATH / "02_violin_box_numeric.png",
    )

    print("\n[4] Univariate Analysis: Categorical Bar Charts...")
    plot_categorical_bar_charts(
        df,
        columns=CATEGORICAL_FEATURES,
        save_path=FIGURES_PATH / "03_categorical_bars.png",
    )

    print("\n[5] Univariate Analysis: Target Distribution...")
    plot_target_distribution(
        df,
        TARGET_COL,
        TARGET_ORDER,
        CATEGORY_PALETTE,
        save_path=FIGURES_PATH / "04_target_distribution.png",
    )

    print("\n[6] Univariate Analysis: Summary Statistics...")
    summary_table = compute_univariate_summary(
        df, CONTINUOUS_FEATURES + POI_DISTANCE_COLS
    )
    print("Univariate Summary — Numeric Features:")
    print(summary_table)

    # === BIVARIATE ANALYSIS: FEATURE vs TARGET ===
    print("\n[7] Bivariate Analysis: Continuous vs Target (Box Plots)...")
    plot_continuous_vs_target_boxplots(
        df,
        features=[*CONTINUOUS_FEATURES, *ORDINAL_FEATURES],
        target_col=TARGET_COL,
        order=TARGET_ORDER,
        palette=CATEGORY_PALETTE,
        save_path=FIGURES_PATH / "05_boxplots_vs_target.png",
    )

    print("\n[8] Bivariate Analysis: POI Distances vs Target (Strip Plots)...")
    plot_strip_plots_poi(
        df,
        poi_cols=POI_DISTANCE_COLS,
        target_col=TARGET_COL,
        order=TARGET_ORDER,
        palette=CATEGORY_PALETTE,
        save_path=FIGURES_PATH / "07_strip_poi_vs_target.png",
    )

    print("\n[9] Bivariate Analysis: Categorical Features vs Target...")
    for feat, fname in [
        ("city", "08a_city_vs_target"),
        ("furnished", "08b_furnished_vs_target"),
        ("completion_status", "08c_completion_vs_target"),
    ]:
        plot_stacked_and_clustered_bars(
            df,
            feature=feat,
            target_col=TARGET_COL,
            order=TARGET_ORDER,
            palette=CATEGORY_PALETTE,
            save_path=FIGURES_PATH / f"{fname}.png",
        )

    print("\n[10] Bivariate Analysis: City vs Target Heatmap...")
    plot_crosstab_heatmap(
        df,
        feature="city",
        target_col=TARGET_COL,
        order=TARGET_ORDER,
        save_path=FIGURES_PATH / "09_city_target_heatmap.png",
    )

    # === BIVARIATE ANALYSIS: FEATURE vs FEATURE ===
    print("\n[11] Bivariate Analysis: Area vs Price (Scatter + Regression)...")
    plot_scatter_with_regression(
        df,
        x_col="area_value",
        y_col="price_egp",
        hue_col=TARGET_COL,
        order=TARGET_ORDER,
        palette=CATEGORY_PALETTE,
        save_path=FIGURES_PATH / "10_area_vs_price_scatter.png",
    )

    print("\n[12] Bivariate Analysis: Transit Distance vs Price (Hexbin Density)...")
    plot_hexbin_density(
        df,
        x_col="dist_nearest_transit_station_km",
        y_col="price_egp",
        save_path=FIGURES_PATH / "11_hexbin_transit_vs_price.png",
    )

    print("\n[13] Bivariate Analysis: Furnished vs Completion Status (Heatmap)...")
    plot_two_categorical_heatmap(
        df,
        row_col="furnished",
        col_col="completion_status",
        save_path=FIGURES_PATH / "12_furnished_completion_heatmap.png",
    )

    # === MULTIVARIATE ANALYSIS ===
    print("\n[14] Multivariate Analysis: Spearman Correlation Matrix...")
    plot_spearman_correlation_matrix(
        df,
        columns=CORR_COLS,
        save_path=FIGURES_PATH / "13_spearman_correlation_matrix.png",
    )

    print("\n[15] Multivariate Analysis: Pair Plot (Structural Features)...")
    plot_pairplot_structural(
        df,
        features=["price_egp", "area_value", "bedrooms", "bathroom"],
        target_col=TARGET_COL,
        order=TARGET_ORDER,
        palette=CATEGORY_PALETTE,
        save_path=FIGURES_PATH / "14_pairplot_structural.png",
    )

    print("\n[16] Multivariate Analysis: Parallel Coordinates...")
    plot_parallel_coordinates(
        df,
        features=PARALLEL_FEATURES,
        target_col=TARGET_COL,
        order=TARGET_ORDER,
        palette=CATEGORY_PALETTE,
        save_path=FIGURES_PATH / "15_parallel_coordinates.png",
    )

    print("\n[17] Multivariate Analysis: Bubble Chart (Area, Price, Bedrooms)...")
    plot_bubble_chart(
        df,
        x_col="area_value",
        y_col="price_egp",
        size_col="bedrooms",
        hue_col=TARGET_COL,
        order=TARGET_ORDER,
        palette=CATEGORY_PALETTE,
        save_path=FIGURES_PATH / "16_bubble_chart.png",
    )

    print("\n[18] Multivariate Analysis: POI Count Comparison...")
    plot_poi_count_comparison(
        df,
        POI_COUNT_COLS,
        TARGET_COL,
        TARGET_ORDER,
        CATEGORY_PALETTE,
        save_path=FIGURES_PATH / "20_poi_counts_by_tier.png",
    )

    # === STATISTICAL TESTS ===
    print("\n[19] Statistical Tests: One-Way ANOVA (Price across Cities)...")
    run_one_way_anova(df, numeric_col="price_egp", group_col="city")

    print("\n[20] Statistical Tests: Chi-Square Independence Tests...")
    chi2_results = {}
    for row_col, col_col in [
        ("furnished", TARGET_COL),
        ("completion_status", TARGET_COL),
    ]:
        chi2_results[f"{row_col} × {col_col}"] = run_chi_square_independence(
            df, row_col=row_col, col_col=col_col
        )

    chi2_summary = pd.DataFrame(
        [
            {
                "comparison": k,
                "chi2": v["chi2"],
                "p_value": v["p"],
                "cramers_v": v["cramers_v"],
                "effect": v["effect"],
                "significant": v["p"] < ALPHA,
            }
            for k, v in chi2_results.items()
        ]
    )
    print("\nChi-Square Test Summary:")
    print(chi2_summary)

    # === FEATURE RANKING ===
    print("\n[21] Feature Importance: Ranking by Target Correlation...")
    feature_ranking = rank_features_by_target_correlation(
        df, ALL_NUMERIC, TARGET_COL, TARGET_ORDER
    )

    # Plot feature importance
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#E84C4C" if r else "#4C9BE8" for r in feature_ranking["significant"]]
    ax.barh(feature_ranking["feature"], feature_ranking["abs_r"], color=colors)
    ax.set_xlabel("|Spearman r| with price_category")
    ax.set_title(
        "Feature Importance Proxy — Spearman Correlation with Target", fontweight="bold"
    )
    ax.axvline(
        0.1, color="gray", linestyle="--", linewidth=1, label="weak threshold (0.1)"
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / "22_feature_importance_proxy.png", bbox_inches="tight")
    plt.close()

    print("\nFeature Ranking:")
    print(feature_ranking.to_string(index=False))

    # === DASHBOARD ===
    print("\n[22] Building EDA Dashboard...")
    build_eda_dashboard(
        df,
        feature_ranking=feature_ranking,
        target_col=TARGET_COL,
        order=TARGET_ORDER,
        palette=CATEGORY_PALETTE,
        poi_count_cols=POI_COUNT_COLS,
        save_path=FIGURES_PATH / "00_eda_dashboard.png",
    )

    print("\nEDA COMPLETE\n")
    print(f"\nAll figures saved to: {FIGURES_PATH.absolute()}")


if __name__ == "__main__":
    run_eda()
