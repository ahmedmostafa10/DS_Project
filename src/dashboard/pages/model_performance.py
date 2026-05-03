from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

MODEL_DATA = {
    "Logistic Regression": {
        "train_acc": 0.52,
        "val_acc": 0.52,
        "test_acc": 0.52,
        "macro_f1": 0.50,
        "f1_low": 0.50,
        "f1_medium": 0.34,
        "f1_high": 0.54,
        "best_params": {"C": 0.1, "penalty": "l2", "solver": "lbfgs"},
    },
    "Decision Tree": {
        "train_acc": 0.88,
        "val_acc": 0.67,
        "test_acc": 0.67,
        "macro_f1": 0.66,
        "f1_low": 0.70,
        "f1_medium": 0.57,
        "f1_high": 0.73,
        "best_params": {"max_depth": 10, "min_samples_split": 5},
    },
    "Random Forest": {
        "train_acc": 0.94,
        "val_acc": 0.70,
        "test_acc": 0.70,
        "macro_f1": 0.71,
        "f1_low": 0.75,
        "f1_medium": 0.61,
        "f1_high": 0.75,
        "best_params": {"n_estimators": 300, "max_depth": 20},
    },
    "XGBoost": {
        "train_acc": 0.85,
        "val_acc": 0.72,
        "test_acc": 0.73,
        "macro_f1": 0.73,
        "f1_low": 0.77,
        "f1_medium": 0.64,
        "f1_high": 0.78,
        "best_params": {
            "n_estimators": 300,
            "max_depth": 8,
            "learning_rate": 0.1,
        },
    },
    "SVC": {
        "train_acc": 0.42,
        "val_acc": 0.43,
        "test_acc": 0.43,
        "macro_f1": 0.32,
        "f1_low": 0.41,
        "f1_medium": 0.08,
        "f1_high": 0.41,
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
                "Train Acc": f"{metrics['train_acc']:.0%}",
                "Val Acc": f"{metrics['val_acc']:.0%}",
                # "Test Acc": f"{metrics['test_acc']:.0%}",
                "Macro F1": f"{metrics['macro_f1']:.2f}",
                "Medium F1": f"{metrics['f1_medium']:.2f}",
                "High Prec.": f"{metrics['f1_high']:.2f}",
            }
        )
    return pd.DataFrame(rows)


def make_accuracy_comparison_figure() -> plt.Figure:
    """Create a bar chart comparing model accuracies."""
    fig, ax = plt.subplots(figsize=(12, 5.5))

    model_names = list(MODEL_DATA.keys())
    train_accs = [MODEL_DATA[m]["train_acc"] for m in model_names]
    val_accs = [MODEL_DATA[m]["val_acc"] for m in model_names]
    # test_accs = [MODEL_DATA[m]["test_acc"] for m in model_names]

    x = np.arange(len(model_names))
    width = 0.25

    ax.bar(x - width, train_accs, width, label="Train", alpha=0.8)
    ax.bar(x, val_accs, width, label="Validation", alpha=0.8)
    # ax.bar(x + width, test_accs, width, label="Test", alpha=0.8)

    ax.set_ylabel("Accuracy")
    ax.set_title("Model Performance Comparison (Train/Val Accuracy)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=25, ha="right")
    ax.set_ylim([0.3, 1.0])
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
        f1_data.append([metrics["f1_low"], metrics["f1_medium"], metrics["f1_high"]])

    f1_df = pd.DataFrame(f1_data, index=model_names, columns=["Low", "Medium", "High"])

    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.heatmap(
        f1_df,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0.0,
        vmax=0.80,
        cbar_kws={"label": "F1 Score"},
        ax=ax,
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_title("F1 Scores by Price Tier (Test Set)", fontweight="bold")
    ax.set_xlabel("Price Tier")
    ax.set_ylabel("Model")

    fig.tight_layout()
    return fig


def show() -> None:
    st.title("Model Performance")
    st.caption("Comparison of five classification models trained to predict apartment price tier.")

    metrics = st.columns(4)
    metrics[0].metric("Best Model", "XGBoost", "73% test acc")
    metrics[1].metric("XGBoost Overfitting", "12%", "Train: 85%, Val: 72%")
    metrics[2].metric("Best Macro F1", "0.73", "XGBoost test")
    metrics[3].metric("Weakest Tier", "Medium F1", "0.64 (XGBoost)")

    st.markdown("### Summary Table")
    table_df = make_comparison_table()
    st.dataframe(table_df, use_container_width=True)
    st.caption(
        "XGBoost achieves 73% test accuracy with a 12-point overfitting gap (85% train → 72% val → 73% test). "
        "This represents the best balance of performance and generalization."
    )

    st.markdown("### Test Accuracy Comparison")
    st.pyplot(make_accuracy_comparison_figure(), width="stretch")
    st.info(
        "XGBoost (73%) outperforms all other models. Decision Tree and Random Forest suffer from significant overfitting "
        "(88%→67% and 94%→70% respectively). Logistic Regression and SVC underfit (42-52% train/test)."
    )

    st.markdown("### F1 Scores by Price Tier")
    st.pyplot(make_f1_by_tier_figure(), width="stretch")
    st.warning(
        "**Medium tier is the hardest to predict:** F1 ranges from 0.08 (SVC) to 0.64 (XGBoost). "
        "This class sits at the boundary between Low and High, causing systematic confusion. "
        "Low and High are easier (F1 ≈ 0.77–0.78 for both in XGBoost)."
    )

    st.markdown("### Hyperparameter Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Logistic Regression**")
        st.code("C: 0.1\npenalty: L2\nsolver: lbfgs", language="yaml")
    with col2:
        st.markdown("**XGBoost (Best)**")
        st.code(
            "n_estimators: 300\nmax_depth: 8\nlearning_rate: 0.1",
            language="yaml",
        )

    st.markdown("### XGBoost Test Classification Report")
    xgb_results = pd.DataFrame(
        {
            "Price Tier": ["Low (Class 0)", "Medium (Class 1)", "High (Class 2)", "Macro Average"],
            "Precision": [0.79, 0.63, 0.78, 0.73],
            "Recall": [0.75, 0.66, 0.78, 0.73],
            "F1-Score": [0.77, 0.64, 0.78, 0.73],
            "Support": [773, 774, 773, 2320],
        }
    )
    st.dataframe(xgb_results, use_container_width=True)
    st.success(
        "XGBoost achieved 73% test accuracy and 0.73 Macro F1. Training accuracy was 85%, "
        "resulting in a 12-point overfitting gap—the best balance among all tested models."
    )

    st.markdown("### Key Insights")
    st.info(
        "✓ **XGBoost wins:** 73% test accuracy, best Macro F1 (0.73), and most balanced overfitting (12 points).\n\n"
        "✓ **Low & High tiers are predictable:** XGBoost F1 = 0.77–0.78 with strong precision (0.78–0.79).\n\n"
        "✗ **Medium tier is intrinsically hard:** F1 = 0.64 even with best model. Data structure creates natural ambiguity.\n\n"
        "✗ **Other models struggle:** Logistic Regression/SVC underfit; Decision Tree/Random Forest overfit severely."
    )
