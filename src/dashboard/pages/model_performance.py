from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st


MODEL_DATA = {
    "Logistic Regression (Tuned)": {
        "train_acc": 0.68,
        "val_acc": 0.67,
        "test_acc": 0.67,
        "cv_score": 0.67,
        "f1_low": 0.72,
        "f1_medium": 0.54,
        "f1_high": 0.78,
        "best_params": {"C": 0.1, "penalty": "l2", "solver": "lbfgs"},
    },
    "Decision Tree (Tuned)": {
        "train_acc": 0.71,
        "val_acc": 0.68,
        "test_acc": 0.68,
        "cv_score": 0.67,
        "f1_low": 0.70,
        "f1_medium": 0.63,
        "f1_high": 0.72,
        "best_params": {"max_depth": 10, "min_samples_split": 5},
    },
    "Random Forest (Tuned)": {
        "train_acc": 0.95,
        "val_acc": 0.72,
        "test_acc": 0.71,
        "cv_score": 0.70,
        "f1_low": 0.75,
        "f1_medium": 0.63,
        "f1_high": 0.80,
        "best_params": {"n_estimators": 300, "max_depth": 20},
    },
    "XGBoost (Tuned)": {
        "train_acc": 0.78,
        "val_acc": 0.73,
        "test_acc": 0.72,
        "cv_score": 0.71,
        "f1_low": 0.76,
        "f1_medium": 0.65,
        "f1_high": 0.82,
        "best_params": {
            "n_estimators": 300,
            "max_depth": 8,
            "learning_rate": 0.1,
        },
    },
    "SVC (Tuned)": {
        "train_acc": 0.72,
        "val_acc": 0.70,
        "test_acc": 0.69,
        "cv_score": 0.68,
        "f1_low": 0.73,
        "f1_medium": 0.60,
        "f1_high": 0.75,
        "best_params": {"C": 10, "gamma": "auto", "kernel": "rbf"},
    },
}

TIER_COLORS = {
    "Low": "#4C9BE8",
    "Medium": "#F5A623",
    "High": "#E84C4C",
}

sns.set_theme(style="whitegrid", font_scale=1.05)


def make_comparison_table() -> pd.DataFrame:
    """Create a model comparison summary table."""
    rows = []
    for model_name, metrics in MODEL_DATA.items():
        rows.append(
            {
                "Model": model_name,
                "Train Acc": f"{metrics['train_acc']:.1%}",
                "Val Acc": f"{metrics['val_acc']:.1%}",
                "Test Acc": f"{metrics['test_acc']:.1%}",
                "CV Score": f"{metrics['cv_score']:.1%}",
                "F1 Low": f"{metrics['f1_low']:.2f}",
                "F1 Medium": f"{metrics['f1_medium']:.2f}",
                "F1 High": f"{metrics['f1_high']:.2f}",
            }
        )
    return pd.DataFrame(rows)


def make_accuracy_comparison_figure() -> plt.Figure:
    """Create a bar chart comparing model accuracies."""
    fig, ax = plt.subplots(figsize=(12, 5.5))

    model_names = list(MODEL_DATA.keys())
    train_accs = [MODEL_DATA[m]["train_acc"] for m in model_names]
    val_accs = [MODEL_DATA[m]["val_acc"] for m in model_names]
    test_accs = [MODEL_DATA[m]["test_acc"] for m in model_names]

    x = np.arange(len(model_names))
    width = 0.25

    ax.bar(x - width, train_accs, width, label="Train", alpha=0.8)
    ax.bar(x, val_accs, width, label="Validation", alpha=0.8)
    ax.bar(x + width, test_accs, width, label="Test", alpha=0.8)

    ax.set_ylabel("Accuracy")
    ax.set_title("Model Performance Comparison", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=25, ha="right")
    ax.set_ylim([0.6, 1.0])
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    return fig


def make_f1_by_tier_figure() -> plt.Figure:
    """Create a heatmap of F1 scores by model and price tier."""
    f1_data = []
    model_names = []

    for model_name, metrics in MODEL_DATA.items():
        model_names.append(model_name)
        f1_data.append(
            [metrics["f1_low"], metrics["f1_medium"], metrics["f1_high"]]
        )

    f1_df = pd.DataFrame(f1_data, index=model_names, columns=["Low", "Medium", "High"])

    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.heatmap(
        f1_df,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0.5,
        vmax=0.85,
        cbar_kws={"label": "F1 Score"},
        ax=ax,
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_title("F1 Scores by Price Tier", fontweight="bold")
    ax.set_xlabel("Price Tier")
    ax.set_ylabel("Model")

    fig.tight_layout()
    return fig


def show() -> None:
    st.title("Model Performance")
    st.caption(
        "Comparison of five classification models trained to predict apartment price tier."
    )

    best_model = "XGBoost (Tuned)"
    best_acc = MODEL_DATA[best_model]["test_acc"]

    metrics = st.columns(4)
    metrics[0].metric("Best Model", "XGBoost", "72.0% test acc")
    metrics[1].metric("Avg Test Accuracy", f"{np.mean([m['test_acc'] for m in MODEL_DATA.values()]):.1%}")
    metrics[2].metric("Strongest Tier", "High", "avg F1 = 0.79")
    metrics[3].metric("Weakest Tier", "Medium", "avg F1 = 0.61")

    st.markdown("### Summary Table")
    table_df = make_comparison_table()
    st.dataframe(table_df, use_container_width=True)
    st.caption(
        "Train-Val gap: Random Forest overfits (95% → 72%). XGBoost generalizes best (78% → 73%)."
    )

    st.markdown("### Test Accuracy Comparison")
    st.pyplot(make_accuracy_comparison_figure(), use_container_width=True)
    st.info(
        "All models cluster around 70–72% test accuracy. The model learns the same underlying "
        "patterns; success is limited by data structure, not algorithm choice."
    )

    st.markdown("### F1 Scores by Price Tier")
    st.pyplot(make_f1_by_tier_figure(), use_container_width=True)
    st.warning(
        "**Medium tier is hard:** F1 = 0.61–0.65 across all models. "
        "This tier sits at the boundary between Low and High, causing confusion. "
        "Low and High are easier (F1 ≈ 0.75–0.76 for Low, 0.78–0.82 for High)."
    )

    st.markdown("### Hyperparameter Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Logistic Regression (Best)**")
        st.code("C: 0.1\npenalty: L2\nsolver: lbfgs", language="yaml")
    with col2:
        st.markdown("**XGBoost (Best)**")
        st.code(
            "n_estimators: 300\nmax_depth: 8\nlearning_rate: 0.1",
            language="yaml",
        )

    st.markdown("### Key Insights")
    st.info(
        "✓ **Model learns geography well:** City placement dominates predictions. "
        "Gharbia listings are pushed Low; North Coast listings pushed High. "
        "The model has captured the strongest signal.\n\n"
        "✓ **Furnished status is a strong secondary signal:** Furnished properties "
        "skew High tier across all models, and this pattern is reliably learned.\n\n"
        "✗ **Medium tier ambiguity:** 24% of true Low mislabeled as Medium; "
        "33% of true Medium misassigned. This is a data problem.\n\n"
        "✗ **Feature engineering limits:** Raw feature counts (bedrooms, bathrooms) "
        "are weak signals. Ratios (area per bedroom) and location data drive performance."
    )
