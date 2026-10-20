"""
Meemansa_Sharma__ML_Network_Intrusion_Detection.py
===================================================
Streamlit dashboard for the ML Network Intrusion Detection project.

This file is the ONLY frontend file.  It reads saved results, metrics, and
the trained Random Forest pipeline.  It never calls .fit() or retrains anything.

Run with:
    .venv\\Scripts\\python.exe -m streamlit run Meemansa_Sharma__ML_Network_Intrusion_Detection.py
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

# ---------------------------------------------------------------------------
# Page configuration — must be the FIRST Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Network Intrusion Detection — ML Analytics",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Project-relative paths — robust regardless of terminal working directory
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.resolve()
MODELS_DIR  = ROOT / "models"
RESULTS_DIR = ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
DATA_DIR    = ROOT / "data"

RF_MODEL_PATH = MODELS_DIR / "RandomForest.joblib"

# Columns the saved pipeline NEVER sees (excluded before preprocessing)
EXCLUDE_COLS = frozenset({"label", "attack_cat", "id"})
# All raw feature columns the pipeline expects
FEATURE_COLS = [
    "dur", "proto", "service", "state", "spkts", "dpkts", "sbytes",
    "dbytes", "rate", "sttl", "dttl", "sload", "dload", "sloss", "dloss",
    "sinpkt", "dinpkt", "sjit", "djit", "swin", "stcpb", "dtcpb", "dwin",
    "tcprtt", "synack", "ackdat", "smean", "dmean", "trans_depth",
    "response_body_len", "ct_srv_src", "ct_state_ttl", "ct_dst_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm",
    "ct_srv_dst", "is_sm_ips_ports",
]
NUMERIC_FEATURES = [
    "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sttl", "dttl",
    "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit",
    "swin", "stcpb", "dtcpb", "dwin", "tcprtt", "synack", "ackdat", "smean",
    "dmean", "trans_depth", "response_body_len", "ct_srv_src", "ct_state_ttl",
    "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm",
    "ct_srv_dst", "is_sm_ips_ports",
]
CATEGORICAL_FEATURES = ["proto", "service", "state"]

# ---------------------------------------------------------------------------
# Dark cybersecurity theme — injected via HTML/CSS
# ---------------------------------------------------------------------------
THEME_CSS = """
<style>
    /* ---- Root colour palette ---- */
    :root {
        --bg-dark:    #0d1117;
        --bg-surface: #161b22;
        --bg-card:    #1c2230;
        --border:     #30363d;
        --accent:     #1f6feb;
        --accent2:    #3fb950;
        --danger:     #f85149;
        --text:       #e6edf3;
        --muted:      #8b949e;
        --cyan:       #58a6ff;
    }

    /* Main background */
    .stApp { background-color: var(--bg-dark); }
    .block-container { background-color: var(--bg-dark); }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: var(--bg-surface) !important;
        border-right: 1px solid var(--border);
    }

    /* General text */
    html, body, p, li, span, div, label, .stMarkdown {
        color: var(--text) !important;
    }
    h1, h2, h3 { color: var(--cyan) !important; }
    h4, h5, h6 { color: var(--text) !important; }

    /* Metric cards */
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 18px 20px 14px 20px;
        text-align: center;
    }
    .metric-label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--muted);
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: var(--cyan);
        line-height: 1.1;
    }
    .metric-value.good  { color: var(--accent2); }
    .metric-value.warn  { color: #d29922; }
    .metric-value.alert { color: var(--danger); }
    .metric-sub {
        font-size: 11px;
        color: var(--muted);
        margin-top: 4px;
    }

    /* Info / callout boxes */
    .info-box {
        background: var(--bg-card);
        border-left: 3px solid var(--accent);
        border-radius: 4px;
        padding: 12px 16px;
        margin: 10px 0;
        font-size: 13px;
    }
    .warn-box {
        background: var(--bg-card);
        border-left: 3px solid #d29922;
        border-radius: 4px;
        padding: 12px 16px;
        margin: 10px 0;
        font-size: 13px;
    }
    .danger-box {
        background: var(--bg-card);
        border-left: 3px solid var(--danger);
        border-radius: 4px;
        padding: 12px 16px;
        margin: 10px 0;
        font-size: 13px;
    }

    /* DataFrames */
    [data-testid="stDataFrame"] { background: var(--bg-card); }

    /* Divider */
    hr { border-color: var(--border); }

    /* Footer */
    .footer-note {
        text-align: center;
        color: var(--muted);
        font-size: 11px;
        margin-top: 40px;
        padding-top: 12px;
        border-top: 1px solid var(--border);
    }
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict | None:
    """Load a JSON file; return None if absent or invalid (no crash)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        st.warning(f"Could not read {path.name}: {e}")
        return None


def load_csv_file(path: Path) -> pd.DataFrame | None:
    """Load a CSV file; return None if absent (no crash)."""
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return None
    except Exception as e:
        st.warning(f"Could not read {path.name}: {e}")
        return None


def show_figure(path: Path, caption: str = "", use_column_width: bool = True):
    """Display a saved PNG figure if it exists; otherwise show a fallback."""
    if path.exists():
        img = Image.open(path)
        st.image(img, caption=caption, use_container_width=use_column_width)
    else:
        st.info(f"Figure not found: `{path.name}`. Run the analysis scripts to generate it.")


def metric_card(label: str, value: str, sub: str = "", colour_class: str = "") -> str:
    """Return an HTML metric card string."""
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value {colour_class}">{value}</div>'
        f'{sub_html}'
        f'</div>'
    )


def pct(v: float) -> str:
    return f"{v * 100:.1f}%"


# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load_model_comparison_json() -> dict | None:
    return load_json(METRICS_DIR / "model_comparison.json")


@st.cache_data(show_spinner=False)
def _load_model_comparison_csv() -> pd.DataFrame | None:
    return load_csv_file(METRICS_DIR / "model_comparison.csv")


@st.cache_data(show_spinner=False)
def _load_eda_summary() -> dict | None:
    return load_json(METRICS_DIR / "eda_summary.json")


@st.cache_data(show_spinner=False)
def _load_interpretation_summary() -> dict | None:
    return load_json(METRICS_DIR / "interpretation_summary.json")


@st.cache_data(show_spinner=False)
def _load_individual_metrics(model_name: str) -> dict | None:
    return load_json(METRICS_DIR / f"metrics_{model_name}.json")


@st.cache_data(show_spinner=False, max_entries=1)
def _load_dataset_head() -> pd.DataFrame | None:
    """Load only the first 500 rows of training data for the preview table."""
    path = DATA_DIR / "UNSW_NB15_training-set.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, nrows=500)
        return df
    except Exception:
        return None


@st.cache_resource(show_spinner="Loading Random Forest model…")
def _load_rf_pipeline():
    """Load and cache the Random Forest sklearn Pipeline (cached as resource)."""
    if not RF_MODEL_PATH.exists():
        return None, f"Model file not found: {RF_MODEL_PATH}"
    try:
        pipeline = joblib.load(RF_MODEL_PATH)
        return pipeline, None
    except Exception as e:
        return None, f"Failed to load model: {e}"


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

st.sidebar.markdown(
    "<h2 style='color:#58a6ff;margin-bottom:4px;'>🛡 NIDS Dashboard</h2>"
    "<p style='font-size:11px;color:#8b949e;margin-top:0;'>ML-Based Intrusion Detection</p>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

PAGES = {
    "Overview":                 "overview",
    "Model Comparison":         "comparison",
    "Dataset Explorer":         "dataset",
    "Feature Importance & Errors": "features",
    "Prediction Lab":           "predict",
    "Project Information":      "info",
}
page_label = st.sidebar.radio(
    "Navigate",
    list(PAGES.keys()),
    label_visibility="collapsed",
)
page = PAGES[page_label]

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<p style='font-size:11px;color:#8b949e;'>"
    "<b>Dataset:</b> UNSW-NB15<br>"
    "<b>Models:</b> Dummy · LR · Random Forest<br>"
    "<b>Best model:</b> Random Forest<br>"
    "<b>Task:</b> Binary intrusion classification"
    "</p>",
    unsafe_allow_html=True,
)


# ===========================================================================
# PAGE: OVERVIEW
# ===========================================================================
if page == "overview":
    st.markdown("# Machine Learning-Based Network Intrusion Detection")
    st.markdown(
        "<p style='color:#8b949e;font-size:14px;'>"
        "An offline dataset-based cybersecurity analytics prototype using the UNSW-NB15 benchmark. "
        "Classifies network-flow records as <b>Normal</b> or <b>Attack</b> traffic using supervised machine learning."
        "</p>",
        unsafe_allow_html=True,
    )

    eda = _load_eda_summary()
    mc  = _load_model_comparison_json()

    # ---- Dataset stats row ----
    st.markdown("### Dataset at a Glance")
    c1, c2, c3, c4 = st.columns(4)
    if eda:
        train_rows = eda.get("train", {}).get("rows", "—")
        test_rows  = eda.get("test", {}).get("rows", "—")
        train_atk  = eda.get("train", {}).get("class_distribution", {}).get("attack_pct", "—")
        test_atk   = eda.get("test",  {}).get("class_distribution", {}).get("attack_pct", "—")
        c1.markdown(metric_card("Train Rows", f"{train_rows:,}",  "UNSW-NB15 training set"), unsafe_allow_html=True)
        c2.markdown(metric_card("Test Rows",  f"{test_rows:,}",   "UNSW-NB15 testing set"),  unsafe_allow_html=True)
        c3.markdown(metric_card("Attack % (Train)", f"{train_atk}%", "class imbalance"), unsafe_allow_html=True)
        c4.markdown(metric_card("Attack % (Test)",  f"{test_atk}%",  "class imbalance"), unsafe_allow_html=True)
    else:
        st.info("EDA summary not found. Run `notebooks/01_exploratory_data_analysis.ipynb` to generate it.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Best model metrics row ----
    st.markdown("### Best Model Performance — Random Forest (on Test Set)")
    if mc and "RandomForest" in mc:
        rf = mc["RandomForest"]
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.markdown(metric_card("Accuracy",    pct(rf["accuracy"]),        "overall correct",       "good"),  unsafe_allow_html=True)
        m2.markdown(metric_card("ROC-AUC",     f"{rf['roc_auc']:.4f}",     "discrimination power",  "good"),  unsafe_allow_html=True)
        m3.markdown(metric_card("Attack F1",   pct(rf["attack_f1"]),       "attack class F1",       "good"),  unsafe_allow_html=True)
        m4.markdown(metric_card("Attack Recall", pct(rf["attack_recall"]), "attacks detected",      "good"),  unsafe_allow_html=True)
        m5.markdown(metric_card("False Neg.",  str(rf["fn"]),              "undetected attacks",    "warn"),   unsafe_allow_html=True)
    else:
        st.info("Model metrics not found.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Normal vs Attack explanation ----
    col_a, col_b = st.columns(2)
    col_a.markdown(
        '<div class="info-box">'
        '<b>Label 0 — Normal Traffic</b><br>'
        'Benign network connections. The model predicts no intrusion attempt. '
        'A false positive (normal classified as attack) causes unnecessary analyst workload.'
        '</div>',
        unsafe_allow_html=True,
    )
    col_b.markdown(
        '<div class="danger-box">'
        '<b>Label 1 — Attack Traffic</b><br>'
        'Malicious or anomalous network activity (e.g. DoS, Exploits, Reconnaissance). '
        'A false negative (attack classified as normal) is the most dangerous outcome — the intrusion goes undetected.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    # ---- Figures ----
    st.markdown("### Class Distribution")
    show_figure(FIGURES_DIR / "class_distribution_both_splits.png",
                "Class distribution in training and testing splits.")
    st.markdown("### Model Comparison")
    show_figure(FIGURES_DIR / "model_comparison.png",
                "Performance comparison across all three models on the test set.")
    st.markdown("### ROC Curves")
    show_figure(FIGURES_DIR / "roc_curves.png",
                "ROC curves — the area under the curve (AUC) measures discrimination power.")

    st.markdown(
        '<div class="footer-note">Offline demonstration only — not connected to live network traffic.</div>',
        unsafe_allow_html=True,
    )


# ===========================================================================
# PAGE: MODEL COMPARISON
# ===========================================================================
elif page == "comparison":
    st.markdown("# Model Comparison")
    st.markdown(
        "<p style='color:#8b949e;font-size:14px;'>"
        "All metrics are evaluated on the held-out UNSW-NB15 test set (82,332 records). "
        "No model had access to test data during training."
        "</p>",
        unsafe_allow_html=True,
    )

    mc_csv  = _load_model_comparison_csv()
    mc_json = _load_model_comparison_json()

    if mc_csv is not None:
        st.markdown("### Metrics Table")
        display_cols = [
            "model", "accuracy", "attack_recall", "attack_precision",
            "attack_f1", "normal_recall", "normal_precision", "normal_f1",
            "macro_f1", "weighted_f1", "roc_auc", "predict_time_s",
        ]
        # Keep only columns that exist in the CSV (roc_auc may be absent)
        display_cols = list(dict.fromkeys(c for c in display_cols if c in mc_csv.columns))
        # Add roc_auc from JSON if not in CSV
        display_df = mc_csv[display_cols].copy()
        if "roc_auc" not in display_df.columns and mc_json:
            display_df["roc_auc"] = display_df["model"].map(
                {k: v.get("roc_auc", None) for k, v in mc_json.items()}
            )
        # Format numeric columns
        fmt_cols = [c for c in display_df.columns if c != "model"]
        for col in fmt_cols:
            if col == "predict_time_s":
                display_df[col] = display_df[col].map(lambda x: f"{x:.3f}s")
            else:
                display_df[col] = display_df[col].map(lambda x: f"{x:.4f}" if isinstance(x, float) else x)
        display_df.columns = [c.replace("_", " ").title() for c in display_df.columns]
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.error("model_comparison.csv not found in results/metrics/")

    # ---- Metric explanations ----
    st.markdown("### Metric Glossary")
    with st.expander("What do these metrics mean?", expanded=False):
        st.markdown("""
| Metric | What it measures |
|--------|-----------------|
| **Accuracy** | Fraction of all predictions that are correct. Easy to understand but misleading on imbalanced datasets — a model that always predicts "Attack" achieves 55% accuracy here. |
| **Precision** (Attack) | Of all records predicted as attacks, how many actually were attacks? Low precision → many false alarms. |
| **Recall** (Attack) | Of all real attacks, how many did the model catch? Low recall → dangerous missed attacks (false negatives). |
| **F1 Score** | Harmonic mean of precision and recall. Balances both concerns. |
| **Macro F1** | Average F1 across both classes, treating them equally. |
| **ROC-AUC** | Area under the ROC curve. 1.0 = perfect, 0.5 = random guessing. Measures ranking/discrimination power. |
| **Predict Time** | How long the model takes to score the full test set. |
        """)

    st.markdown("---")
    st.markdown("### Why a Baseline Matters")
    st.markdown(
        '<div class="warn-box">'
        'The <b>DummyClassifier</b> always predicts "Attack" (the majority class). '
        'It achieves 55% accuracy and 100% attack recall — not because it is smart, '
        'but because attacks are slightly more common in the test set. '
        'Any useful model must clearly outperform this baseline. '
        'Random Forest achieves 87% accuracy and 97.9% AUC, demonstrating genuine learning.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### Confusion Matrices")
    st.markdown(
        "A confusion matrix shows how many predictions fell into each outcome category. "
        "TN = True Negative (normal correctly identified), FP = False Positive (normal flagged as attack), "
        "FN = False Negative (attack missed), TP = True Positive (attack correctly detected)."
    )

    model_names = ["DummyClassifier", "LogisticRegression", "RandomForest"]
    cm_cols = st.columns(3)
    for i, name in enumerate(model_names):
        m = _load_individual_metrics(name)
        with cm_cols[i]:
            st.markdown(f"**{name}**")
            fig_path = FIGURES_DIR / f"confusion_matrix_{name}.png"
            if fig_path.exists():
                show_figure(fig_path, caption=name)
            elif m and "confusion_matrix" in m:
                # Build a simple matplotlib confusion matrix as fallback
                cm = np.array(m["confusion_matrix"])
                fig, ax = plt.subplots(figsize=(3, 3))
                fig.patch.set_facecolor("#1c2230")
                ax.set_facecolor("#1c2230")
                im = ax.imshow(cm, cmap="Blues")
                for r in range(2):
                    for c in range(2):
                        ax.text(c, r, f"{cm[r, c]:,}", ha="center", va="center",
                                color="white", fontsize=11)
                ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
                ax.set_xticklabels(["Normal", "Attack"], color="white")
                ax.set_yticklabels(["Normal", "Attack"], color="white")
                ax.set_xlabel("Predicted", color="#8b949e")
                ax.set_ylabel("Actual", color="#8b949e")
                ax.tick_params(colors="white")
                for spine in ax.spines.values():
                    spine.set_edgecolor("#30363d")
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            else:
                st.info("No confusion matrix data available.")

            if m:
                st.markdown(
                    f"<small style='color:#8b949e;'>"
                    f"TN={m.get('tn', '?'):,}  FP={m.get('fp', '?'):,}<br>"
                    f"FN={m.get('fn', '?'):,}  TP={m.get('tp', '?'):,}</small>",
                    unsafe_allow_html=True,
                )

    st.markdown(
        '<div class="footer-note">Offline demonstration only — not connected to live network traffic.</div>',
        unsafe_allow_html=True,
    )


# ===========================================================================
# PAGE: DATASET EXPLORER
# ===========================================================================
elif page == "dataset":
    st.markdown("# Dataset Explorer")
    st.markdown(
        "<p style='color:#8b949e;font-size:14px;'>"
        "Exploring the UNSW-NB15 dataset — an official benchmark of synthetic network traffic "
        "generated at the Australian Centre for Cyber Security."
        "</p>",
        unsafe_allow_html=True,
    )

    eda = _load_eda_summary()

    if eda:
        train_info = eda.get("train", {})
        test_info  = eda.get("test",  {})

        # ---- Dimensions ----
        st.markdown("### Dataset Dimensions")
        d1, d2, d3, d4 = st.columns(4)
        d1.markdown(metric_card("Train Rows",    f"{train_info.get('rows', '—'):,}",    "training records"),  unsafe_allow_html=True)
        d2.markdown(metric_card("Train Columns", str(train_info.get("columns", "—")),  "raw feature cols"),  unsafe_allow_html=True)
        d3.markdown(metric_card("Test Rows",     f"{test_info.get('rows', '—'):,}",     "testing records"),   unsafe_allow_html=True)
        d4.markdown(metric_card("Test Columns",  str(test_info.get("columns", "—")),   "raw feature cols"),  unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ---- Feature exclusion note ----
        excluded = eda.get("columns_excluded_from_features", {})
        if excluded:
            st.markdown(
                '<div class="warn-box">'
                '<b>Columns excluded from model features:</b><br>'
                + "<br>".join(f"<code>{col}</code> — {reason}" for col, reason in excluded.items())
                + '</div>',
                unsafe_allow_html=True,
            )

        # ---- Class distribution (both splits) ----
        st.markdown("### Class Distribution")
        tab_train, tab_test = st.tabs(["Training Split", "Testing Split"])
        with tab_train:
            cd = train_info.get("class_distribution", {})
            if cd:
                t1, t2, t3 = st.columns(3)
                t1.markdown(metric_card("Total Records", f"{cd.get('total', '—'):,}"), unsafe_allow_html=True)
                t2.markdown(metric_card("Normal Traffic", f"{cd.get('normal_count', '—'):,}", f"{cd.get('normal_pct', '—')}%", "good"), unsafe_allow_html=True)
                t3.markdown(metric_card("Attack Traffic", f"{cd.get('attack_count', '—'):,}", f"{cd.get('attack_pct', '—')}%", "warn"), unsafe_allow_html=True)
        with tab_test:
            cd = test_info.get("class_distribution", {})
            if cd:
                t1, t2, t3 = st.columns(3)
                t1.markdown(metric_card("Total Records", f"{cd.get('total', '—'):,}"), unsafe_allow_html=True)
                t2.markdown(metric_card("Normal Traffic", f"{cd.get('normal_count', '—'):,}", f"{cd.get('normal_pct', '—')}%", "good"), unsafe_allow_html=True)
                t3.markdown(metric_card("Attack Traffic", f"{cd.get('attack_count', '—'):,}", f"{cd.get('attack_pct', '—')}%", "warn"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        show_figure(FIGURES_DIR / "class_distribution_both_splits.png",
                    "Class imbalance: attacks are over-represented in training (68%) and slightly dominant in testing (55%).")

        # ---- Attack category distribution ----
        st.markdown("### Attack Category Distribution")
        cats_train = train_info.get("attack_categories", [])
        cats_test  = test_info.get("attack_categories",  [])

        cat_tab1, cat_tab2 = st.tabs(["Training — Attack Categories", "Testing — Attack Categories"])
        with cat_tab1:
            if cats_train:
                df_cat = pd.DataFrame(cats_train).rename(columns={"attack_cat": "Category", "count": "Count", "pct": "%"})
                st.dataframe(df_cat, use_container_width=True, hide_index=True)
            show_figure(FIGURES_DIR / "attack_category_distribution_train.png", "Attack category breakdown — training split.")
        with cat_tab2:
            if cats_test:
                df_cat = pd.DataFrame(cats_test).rename(columns={"attack_cat": "Category", "count": "Count", "pct": "%"})
                st.dataframe(df_cat, use_container_width=True, hide_index=True)
            show_figure(FIGURES_DIR / "attack_category_distribution_test.png", "Attack category breakdown — testing split.")

    else:
        st.info("EDA summary not found. Run `notebooks/01_exploratory_data_analysis.ipynb` to generate it.")

    st.markdown("---")

    # ---- EDA figures ----
    st.markdown("### Numeric Feature Distributions")
    show_figure(FIGURES_DIR / "numeric_distributions_train.png",
                "Distribution of key numeric features in the training set.")
    st.markdown("### Categorical Feature Frequencies")
    show_figure(FIGURES_DIR / "categorical_frequencies_train.png",
                "Value frequency distributions for protocol, service, and connection state.")
    st.markdown("### Feature Correlation Heatmap")
    show_figure(FIGURES_DIR / "correlation_heatmap_train.png",
                "Pairwise Pearson correlations between numeric features (training set).")

    st.markdown("---")
    # ---- Data preview ----
    st.markdown("### Data Preview (first 500 rows, training set)")
    st.markdown(
        "<p style='color:#8b949e;font-size:12px;'>"
        "Only the first 500 rows are loaded here to avoid memory issues. "
        "The full dataset is used by the model pipeline."
        "</p>",
        unsafe_allow_html=True,
    )
    df_head = _load_dataset_head()
    if df_head is not None:
        st.dataframe(df_head, use_container_width=True, height=300)
    else:
        st.info("Training CSV not found. Place `UNSW_NB15_training-set.csv` in the `data/` directory.")

    st.markdown(
        '<div class="footer-note">Offline demonstration only — not connected to live network traffic.</div>',
        unsafe_allow_html=True,
    )


# ===========================================================================
# PAGE: FEATURE IMPORTANCE & ERROR ANALYSIS
# ===========================================================================
elif page == "features":
    st.markdown("# Feature Importance & Error Analysis")
    st.markdown(
        "<p style='color:#8b949e;font-size:14px;'>"
        "Using the saved interpretation results from the trained Random Forest model. "
        "No recomputation or model fitting is performed here."
        "</p>",
        unsafe_allow_html=True,
    )

    interp = _load_interpretation_summary()

    # ---- Feature importance ----
    st.markdown("### Random Forest Feature Importance (Top 20)")
    if interp and "feature_importances" in interp:
        fi = interp["feature_importances"]
        top20 = sorted(fi, key=lambda x: x["importance"], reverse=True)[:20]
        fi_df = pd.DataFrame(top20)
        # Clean up feature names (strip prefix like "numeric__")
        fi_df["feature_clean"] = fi_df["feature"].str.replace(r"^[a-z]+__", "", regex=True)

        # Horizontal bar chart with matplotlib (dark theme)
        fig, ax = plt.subplots(figsize=(9, 6))
        fig.patch.set_facecolor("#161b22")
        ax.set_facecolor("#161b22")
        colours = ["#1f6feb"] * len(top20)
        colours[0] = "#58a6ff"  # highlight top feature
        bars = ax.barh(fi_df["feature_clean"][::-1], fi_df["importance"][::-1], color=colours[::-1])
        ax.set_xlabel("Importance Score", color="#8b949e")
        ax.set_title("Top 20 Features — Random Forest (Gini importance)", color="#e6edf3", pad=12)
        ax.tick_params(colors="#e6edf3")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        ax.xaxis.label.set_color("#8b949e")
        ax.tick_params(axis="x", colors="#8b949e")
        ax.tick_params(axis="y", colors="#e6edf3")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Also show the saved figure
        show_figure(FIGURES_DIR / "feature_importance_rf.png",
                    "Saved feature importance plot generated during model training.")

        # Table
        st.markdown("#### Top 10 Features")
        top10_df = fi_df.head(10)[["rank", "feature_clean", "importance"]].rename(
            columns={"rank": "Rank", "feature_clean": "Feature", "importance": "Importance Score"}
        )
        top10_df["Importance Score"] = top10_df["Importance Score"].map(lambda x: f"{x:.5f}")
        st.dataframe(top10_df, use_container_width=True, hide_index=True)
    else:
        show_figure(FIGURES_DIR / "feature_importance_rf.png",
                    "Feature importance from Random Forest.")
        st.info("interpretation_summary.json not found or missing feature_importances key.")

    st.markdown("---")
    # ---- What feature importance means ----
    st.markdown("### What Feature Importance Means")
    st.markdown(
        '<div class="info-box">'
        '<b>Random Forest Gini importance</b> measures how much each feature reduces '
        'impurity (uncertainty) across all decision trees on average. '
        'A higher score means the model split on that feature more often and found it more useful.<br><br>'
        '<b>Limitations:</b> '
        '(1) Importance reflects correlation in the training data, not causal influence. '
        '(2) Correlated features can share importance, making individual scores less reliable. '
        '(3) This analysis applies only to this dataset and may not generalise to real traffic.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    # ---- Error analysis ----
    st.markdown("### Error Analysis — False Positives & False Negatives")
    st.markdown(
        "Examining the characteristics of the model's mistakes helps understand when it struggles."
    )

    show_figure(FIGURES_DIR / "error_distributions_rf.png",
                "Distribution of key feature values for false positive and false negative predictions.")

    if interp and "error_analysis" in interp:
        ea = interp["error_analysis"]
        e1, e2 = st.columns(2)
        e1.markdown(
            f'<div class="warn-box">'
            f'<b>False Positives (FP): {ea.get("total_fp", "?"):,}</b><br>'
            f'Normal traffic incorrectly flagged as attacks. '
            f'These cause unnecessary analyst alerts (alert fatigue).'
            f'</div>',
            unsafe_allow_html=True,
        )
        e2.markdown(
            f'<div class="danger-box">'
            f'<b>False Negatives (FN): {ea.get("total_fn", "?"):,}</b><br>'
            f'Attacks that slipped through undetected. '
            f'These are the most dangerous outcome in a real deployment.'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Protocol breakdown of errors
        fp_proto = ea.get("fp_proto_counts", {})
        fn_proto = ea.get("fn_proto_counts", {})
        if fp_proto or fn_proto:
            st.markdown("#### Error Breakdown by Protocol")
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                st.markdown("**False Positives by Protocol**")
                if fp_proto:
                    df_fp = pd.DataFrame(list(fp_proto.items()), columns=["Protocol", "Count"])
                    df_fp = df_fp.sort_values("Count", ascending=False)
                    st.dataframe(df_fp, use_container_width=True, hide_index=True)
            with p_col2:
                st.markdown("**False Negatives by Protocol**")
                if fn_proto:
                    df_fn = pd.DataFrame(list(fn_proto.items()), columns=["Protocol", "Count"])
                    df_fn = df_fn.sort_values("Count", ascending=False)
                    st.dataframe(df_fn, use_container_width=True, hide_index=True)

    if interp and "limitations" in interp:
        st.markdown("---")
        st.markdown("### Model Limitations (from Analysis)")
        st.markdown(
            f'<div class="warn-box">{interp["limitations"]}</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="footer-note">Offline demonstration only — not connected to live network traffic.</div>',
        unsafe_allow_html=True,
    )


# ===========================================================================
# PAGE: PREDICTION LAB
# ===========================================================================
elif page == "predict":
    st.markdown("# Interactive Prediction Lab")
    st.markdown(
        '<div class="warn-box">'
        'This lab uses the <b>pre-trained Random Forest pipeline</b> saved to disk. '
        'No model training or fitting happens here. '
        'Predictions are for educational purposes only — '
        'this is not a real-time network monitor or production intrusion-prevention system.'
        '</div>',
        unsafe_allow_html=True,
    )

    # Load model (cached)
    pipeline, load_err = _load_rf_pipeline()

    if load_err:
        st.error(f"Could not load model: {load_err}")
        st.stop()

    st.success("Random Forest pipeline loaded successfully.")

    # ---- Input method ----
    st.markdown("---")
    st.markdown("### Choose Input Method")
    input_method = st.radio(
        "Input method",
        ["Upload CSV file", "Use sample data (demo)"],
        label_visibility="collapsed",
        horizontal=True,
    )

    # ---- Sample data builder ----
    def _make_sample_row() -> dict:
        """
        Construct a single plausible sample row matching the UNSW-NB15 feature schema.
        Values are approximate medians/typical values from the dataset — for demonstration only.
        This row is synthetic and should not be used to draw real security conclusions.
        """
        return {
            "dur":             0.0,
            "proto":           "tcp",
            "service":         "-",
            "state":           "FIN",
            "spkts":           2,
            "dpkts":           0,
            "sbytes":          100,
            "dbytes":          0,
            "rate":            0.0,
            "sttl":            64,
            "dttl":            0,
            "sload":           0.0,
            "dload":           0.0,
            "sloss":           0,
            "dloss":           0,
            "sinpkt":          0.0,
            "dinpkt":          0.0,
            "sjit":            0.0,
            "djit":            0.0,
            "swin":            0,
            "stcpb":           0,
            "dtcpb":           0,
            "dwin":            0,
            "tcprtt":          0,
            "synack":          0,
            "ackdat":          0,
            "smean":           50,
            "dmean":           0,
            "trans_depth":     0,
            "response_body_len": 0,
            "ct_srv_src":      1,
            "ct_state_ttl":    2,
            "ct_dst_ltm":      1,
            "ct_src_dport_ltm": 1,
            "ct_dst_sport_ltm": 1,
            "ct_dst_src_ltm":  1,
            "is_ftp_login":    0,
            "ct_ftp_cmd":      0,
            "ct_flw_http_mthd": 0,
            "ct_src_ltm":      1,
            "ct_srv_dst":      1,
            "is_sm_ips_ports": 0,
        }

    input_df = None

    if input_method == "Use sample data (demo)":
        st.markdown(
            '<div class="info-box">'
            'The sample row below uses approximate typical values from the UNSW-NB15 training set. '
            'It is synthetic and for demonstration only. '
            'Do not interpret its prediction as a real security finding.'
            '</div>',
            unsafe_allow_html=True,
        )
        n_sample = st.slider("Number of sample rows", 1, 10, 3)
        sample_rows = [_make_sample_row() for _ in range(n_sample)]
        input_df = pd.DataFrame(sample_rows)
        st.dataframe(input_df, use_container_width=True)

    else:  # CSV upload
        st.markdown("### Upload a CSV File")
        st.markdown(
            f"Your CSV must contain these **{len(FEATURE_COLS)} feature columns** "
            f"(column names are case-sensitive). "
            f"The columns `label`, `attack_cat`, and `id` should NOT be included — "
            f"they are excluded by the pipeline."
        )
        with st.expander("Show required feature columns"):
            st.code(", ".join(FEATURE_COLS))

        uploaded = st.file_uploader(
            "Upload CSV (network flow records)",
            type=["csv"],
            label_visibility="collapsed",
        )

        if uploaded is not None:
            try:
                raw_df = pd.read_csv(uploaded)
            except Exception as e:
                st.error(f"Could not parse CSV: {e}")
                st.stop()

            # Strip whitespace from column names
            raw_df.columns = [c.strip() for c in raw_df.columns]

            if raw_df.empty:
                st.error("The uploaded CSV is empty.")
                st.stop()

            # Remove excluded cols silently (they are not needed)
            for exc_col in EXCLUDE_COLS:
                if exc_col in raw_df.columns:
                    raw_df = raw_df.drop(columns=[exc_col])
                    st.info(f"Column `{exc_col}` was found and removed (it is not a model feature).")

            # Validate required feature columns
            present  = set(raw_df.columns)
            required = set(FEATURE_COLS)
            missing  = required - present
            extra    = present - required

            if missing:
                st.error(
                    f"Your CSV is missing {len(missing)} required feature column(s):\n\n"
                    + ", ".join(sorted(missing))
                    + "\n\nPlease add these columns before uploading."
                )
                st.stop()

            if extra:
                st.warning(
                    f"Your CSV contains {len(extra)} extra column(s) that are not model features "
                    f"and will be ignored:\n`{'`, `'.join(sorted(extra))}`"
                )

            st.success(f"CSV loaded: {len(raw_df):,} rows, {len(raw_df.columns)} columns.")
            st.dataframe(raw_df[FEATURE_COLS].head(10), use_container_width=True)
            input_df = raw_df[FEATURE_COLS].copy()

    # ---- Run predictions ----
    st.markdown("---")
    if input_df is not None and st.button("Run Predictions", type="primary"):
        with st.spinner("Running predictions…"):
            try:
                # The saved pipeline handles all preprocessing internally.
                # It was built with split_features_target → only the 42 feature cols.
                X = input_df[FEATURE_COLS].copy()
                preds = pipeline.predict(X)

                # Check if model supports predict_proba
                has_proba = hasattr(pipeline, "predict_proba")
                if has_proba:
                    try:
                        probas = pipeline.predict_proba(X)
                        # probas[:,0] = P(Normal), probas[:,1] = P(Attack)
                        conf_attack  = probas[:, 1]
                        conf_normal  = probas[:, 0]
                        proba_ok = True
                    except Exception:
                        proba_ok = False
                else:
                    proba_ok = False

                # Build results dataframe
                # Label convention: 0 = Normal (label=0), 1 = Attack (label=1)
                result_df = input_df.copy()
                result_df["Prediction (raw)"] = preds
                result_df["Prediction"] = ["🔴 Attack" if p == 1 else "🟢 Normal" for p in preds]
                if proba_ok:
                    result_df["P(Attack)"]  = conf_attack.round(4)
                    result_df["P(Normal)"]  = conf_normal.round(4)
                    result_df["Confidence"] = [
                        f"{max(ca, cn)*100:.1f}%"
                        for ca, cn in zip(conf_attack, conf_normal)
                    ]

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

        # ---- Summary stats ----
        n_total  = len(preds)
        n_attack = int((preds == 1).sum())
        n_normal = int((preds == 0).sum())

        st.markdown("### Prediction Summary")
        s1, s2, s3 = st.columns(3)
        s1.markdown(metric_card("Total Records", str(n_total)), unsafe_allow_html=True)
        s2.markdown(metric_card("Attack",  str(n_attack), f"{n_attack/n_total*100:.1f}%", "alert" if n_attack > 0 else ""), unsafe_allow_html=True)
        s3.markdown(metric_card("Normal",  str(n_normal), f"{n_normal/n_total*100:.1f}%", "good"), unsafe_allow_html=True)

        st.markdown("### Prediction Results")
        display_result_cols = ["Prediction"] + (["P(Attack)", "P(Normal)", "Confidence"] if proba_ok else []) + FEATURE_COLS[:6]
        display_result_cols = [c for c in display_result_cols if c in result_df.columns]
        st.dataframe(result_df[display_result_cols], use_container_width=True, height=350)

        # ---- Download ----
        csv_bytes = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download All Results as CSV",
            data=csv_bytes,
            file_name="nids_predictions.csv",
            mime="text/csv",
        )

        st.markdown(
            '<div class="warn-box">'
            '<b>Reminder:</b> These predictions are generated by a model trained on the UNSW-NB15 benchmark dataset. '
            'They are for educational demonstration only. '
            'Real intrusion detection requires validated, up-to-date threat intelligence and human expert review.'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="footer-note">Offline demonstration only — not connected to live network traffic.</div>',
        unsafe_allow_html=True,
    )


# ===========================================================================
# PAGE: PROJECT INFORMATION / HELP
# ===========================================================================
elif page == "info":
    st.markdown("# Project Information & Help")

    st.markdown("## What is Network Intrusion Detection?")
    st.markdown("""
Network Intrusion Detection Systems (NIDS) monitor network traffic for signs of malicious activity.
They analyse patterns in packet-level or flow-level data to distinguish normal behaviour from attacks
such as port scans, denial-of-service floods, exploit attempts, and backdoor connections.

This project is a **machine learning-based prototype**: instead of hand-written rules, models learn
patterns from labelled historical data and generalise to new, unseen traffic records.
""")

    st.markdown("---")
    st.markdown("## The UNSW-NB15 Dataset")
    st.markdown("""
UNSW-NB15 was created by the Australian Centre for Cyber Security (ACCS) at UNSW Canberra.
It provides a modern, realistic benchmark for intrusion-detection research with:

- **175,341 training records** and **82,332 testing records** (official predefined split).
- **45 features** per record, including network-flow statistics such as byte counts, timing values, protocol types, and connection states.
- **Binary label** (`label`): 0 = normal, 1 = attack.
- **9 attack categories**: Generic, Exploits, Fuzzers, DoS, Reconnaissance, Analysis, Backdoor, Shellcode, Worms.
- No missing values in either split.

Dataset source: [https://research.unsw.edu.au/projects/unsw-nb15-dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
""")

    st.markdown("---")
    st.markdown("## Models Included")
    st.markdown("""
| Model | Purpose | Notes |
|-------|---------|-------|
| **DummyClassifier** | Baseline | Always predicts the majority class (Attack). Establishes minimum performance bar. |
| **Logistic Regression** | Linear baseline | Interpretable, fast. Requires scaled features. Achieves ~81% accuracy, 0.955 AUC. |
| **Random Forest** | Best model | 100 decision trees, non-linear, robust. Achieves ~87% accuracy, 0.979 AUC. |

All models are trained on the training split only. The test split is used solely for final evaluation.
Preprocessing (scaling, one-hot encoding) is fitted on training data and applied to test data — never the reverse.
""")

    st.markdown("---")
    st.markdown("## How to Use This Dashboard")
    st.markdown("""
**Overview** — High-level project summary, key metrics, and main charts.

**Model Comparison** — Detailed metrics table and confusion matrices for all three models.

**Dataset Explorer** — Class distributions, attack categories, EDA charts, and a data preview.

**Feature Importance & Errors** — Which features the Random Forest relies on most, and where it makes mistakes.

**Prediction Lab** — Upload your own CSV of network-flow records (matching the UNSW-NB15 feature schema) to run batch predictions using the saved Random Forest. You can also test with synthetic sample rows.

**Project Information** — This page.
""")

    st.markdown("---")
    st.markdown("## Metric Definitions (Plain Language)")
    st.markdown("""
| Metric | Simple meaning |
|--------|---------------|
| **Accuracy** | Out of all predictions, how many were right? (Not sufficient alone for imbalanced data.) |
| **Precision (Attack)** | When the model says "Attack", how often is it actually an attack? |
| **Recall (Attack)** | Of all real attacks, what fraction did the model catch? (Missing attacks is dangerous.) |
| **F1 Score** | Balanced score combining precision and recall. |
| **ROC-AUC** | How well can the model rank attacks above normal traffic? 1.0 is perfect; 0.5 is random. |
| **False Positive** | Normal traffic flagged as an attack → wastes analyst time. |
| **False Negative** | An actual attack not detected → the most dangerous error. |
""")

    st.markdown("---")
    st.markdown("## Limitations")
    st.markdown(
        '<div class="warn-box">'
        '<ul>'
        '<li><b>Dataset dependence</b>: Models were trained on UNSW-NB15 synthetic traffic from 2015. '
        'Real-world traffic patterns, protocols, and attack techniques have evolved significantly.</li>'
        '<li><b>No live monitoring</b>: This system reads CSV files. It is not connected to any network interface.</li>'
        '<li><b>False negatives are possible</b>: 642 attacks were missed in the test set evaluation. '
        'Novel attacks not represented in the training data may not be detected.</li>'
        '<li><b>Class imbalance</b>: Attack traffic is over-represented in training (68%), '
        'which may bias the model towards predicting attacks.</li>'
        '<li><b>No guarantee of real-world performance</b>: Strong benchmark results do not automatically '
        'translate to production effectiveness against live, modern threats.</li>'
        '<li><b>Not a replacement for expert review</b>: Any flagged traffic should be verified by a '
        'qualified security analyst before action is taken.</li>'
        '</ul>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("## How Models & Metrics Are Reused (No Retraining)")
    st.markdown("""
All three model pipelines were trained offline using `scripts/run_training.py` and saved as `.joblib` files.
Each saved file contains the **complete sklearn Pipeline** — the fitted preprocessor (scaler statistics,
one-hot encoder category lists) AND the fitted classifier. Loading the file gives a ready-to-predict object.

The dashboard loads these files at startup and uses only:
- `.predict(X)` to get class predictions (0 or 1).
- `.predict_proba(X)` to get confidence scores.

No `.fit()` or `.fit_transform()` is called anywhere in the dashboard.
Metric JSON/CSV files in `results/metrics/` are read directly and displayed as-is.
""")

    st.markdown(
        '<div class="footer-note">'
        'This is an offline, dataset-based machine-learning demonstration. '
        'It is not a live network-monitoring system and does not provide real-time intrusion prevention. '
        'Built for educational purposes using the UNSW-NB15 public benchmark dataset.'
        '</div>',
        unsafe_allow_html=True,
    )
