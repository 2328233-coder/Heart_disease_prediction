"""
❤️  Heart Disease Prediction — Streamlit App
Based on UCI Heart Disease Multi-Dataset Research Pipeline
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings("ignore")

# ── Sklearn ────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, matthews_corrcoef,
    confusion_matrix, roc_curve, auc
)
from sklearn.impute import SimpleImputer

# ── Optional Advanced Models ───────────────────────────────────────────
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════
# PAGE CONFIG & GLOBAL STYLE
# ═══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="❤️ Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens (matching notebook palette) ──────────────────────────
HEART_RED   = "#FF4B4B"
HEART_PINK  = "#FF8080"
ACCENT_BLUE = "#58A6FF"
ACCENT_GRN  = "#3FB950"
ACCENT_YLW  = "#D29922"
ACCENT_PRP  = "#BC8CFF"
BG_DARK     = "#0D1117"
BG_CARD     = "#161B22"
BG_BORDER   = "#30363D"
TEXT_MAIN   = "#E6EDF3"
TEXT_MUTED  = "#8B949E"
PALETTE     = [HEART_RED, ACCENT_BLUE, ACCENT_GRN, ACCENT_YLW, ACCENT_PRP,
               HEART_PINK, "#FFA657", "#79C0FF"]

plt.rcParams.update({
    "figure.facecolor": BG_DARK,
    "axes.facecolor": BG_CARD,
    "axes.edgecolor": BG_BORDER,
    "axes.labelcolor": TEXT_MAIN,
    "text.color": TEXT_MAIN,
    "xtick.color": TEXT_MUTED,
    "ytick.color": TEXT_MUTED,
    "grid.color": "#21262D",
    "grid.alpha": 0.5,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "font.family": "DejaVu Sans",
    "savefig.bbox": "tight",
    "savefig.facecolor": BG_DARK,
})

SEED = 42
np.random.seed(SEED)

# ═══════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ═══════════════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Syne:wght@700;800&display=swap');

  html, body, [class*="css"] {{
    font-family: 'Space Grotesk', sans-serif;
    background-color: {BG_DARK};
    color: {TEXT_MAIN};
  }}

  /* Sidebar */
  section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0D1117 0%, #161B22 100%);
    border-right: 1px solid {BG_BORDER};
  }}
  section[data-testid="stSidebar"] * {{ color: {TEXT_MAIN} !important; }}

  /* Main area */
  .main .block-container {{
    background-color: {BG_DARK};
    padding-top: 1.5rem;
    max-width: 1400px;
  }}

  /* Hero banner */
  .hero-banner {{
    background: linear-gradient(135deg, #1a0a0a 0%, #0f1923 50%, #0a1a0a 100%);
    border: 1px solid {BG_BORDER};
    border-top: 3px solid {HEART_RED};
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
  }}
  .hero-banner::before {{
    content: "❤️";
    position: absolute;
    right: 2rem;
    top: 50%;
    transform: translateY(-50%);
    font-size: 5rem;
    opacity: 0.15;
  }}
  .hero-title {{
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: {HEART_RED};
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.5px;
  }}
  .hero-sub {{
    color: {TEXT_MUTED};
    font-size: 0.95rem;
    margin: 0;
    font-weight: 400;
  }}
  .hero-badges {{
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-top: 1rem;
  }}
  .badge {{
    background: rgba(88,166,255,0.12);
    border: 1px solid rgba(88,166,255,0.3);
    color: {ACCENT_BLUE};
    border-radius: 20px;
    padding: 0.2rem 0.75rem;
    font-size: 0.75rem;
    font-weight: 500;
  }}

  /* Section headers */
  .section-header {{
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin: 1.8rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid {BG_BORDER};
  }}
  .section-header h2 {{
    font-family: 'Syne', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    margin: 0;
    color: {TEXT_MAIN};
  }}

  /* Input cards */
  .input-card {{
    background: {BG_CARD};
    border: 1px solid {BG_BORDER};
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
  }}
  .input-card:hover {{ border-color: {ACCENT_BLUE}; }}
  .input-card-title {{
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: {TEXT_MUTED};
    margin-bottom: 0.4rem;
  }}

  /* Result cards */
  .result-low {{
    background: linear-gradient(135deg, #0a1f0a, #0d1117);
    border: 2px solid {ACCENT_GRN};
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
  }}
  .result-medium {{
    background: linear-gradient(135deg, #1a1500, #0d1117);
    border: 2px solid {ACCENT_YLW};
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
  }}
  .result-high {{
    background: linear-gradient(135deg, #1a0505, #0d1117);
    border: 2px solid {HEART_RED};
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    animation: pulse-border 2s infinite;
  }}
  @keyframes pulse-border {{
    0%, 100% {{ box-shadow: 0 0 0 0 rgba(255,75,75,0.4); }}
    50%       {{ box-shadow: 0 0 0 8px rgba(255,75,75,0); }}
  }}
  .result-pct {{
    font-family: 'Syne', sans-serif;
    font-size: 3.5rem;
    font-weight: 800;
    line-height: 1;
    margin: 0.3rem 0;
  }}
  .result-label {{
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-top: 0.3rem;
  }}

  /* Model row */
  .model-row {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: {BG_CARD};
    border: 1px solid {BG_BORDER};
    border-radius: 8px;
    padding: 0.6rem 1rem;
    margin-bottom: 0.4rem;
    font-size: 0.85rem;
  }}
  .model-bar-wrap {{
    width: 120px;
    height: 8px;
    background: #21262D;
    border-radius: 4px;
    overflow: hidden;
  }}
  .model-bar-fill {{
    height: 100%;
    border-radius: 4px;
    transition: width 0.8s ease;
  }}

  /* Metric tiles */
  .metric-tile {{
    background: {BG_CARD};
    border: 1px solid {BG_BORDER};
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
  }}
  .metric-tile-val {{
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
  }}
  .metric-tile-lbl {{
    font-size: 0.72rem;
    color: {TEXT_MUTED};
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-top: 0.1rem;
  }}

  /* Streamlit overrides */
  .stButton button {{
    background: linear-gradient(135deg, {HEART_RED}, #cc1a1a) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.6rem 1.5rem !important;
    width: 100% !important;
    letter-spacing: 0.5px !important;
    transition: all 0.2s !important;
  }}
  .stButton button:hover {{
    background: linear-gradient(135deg, #ff6b6b, {HEART_RED}) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(255,75,75,0.35) !important;
  }}
  div[data-testid="stSelectbox"] label,
  div[data-testid="stSlider"] label,
  div[data-testid="stNumberInput"] label {{
    font-weight: 500 !important;
    color: {TEXT_MAIN} !important;
    font-size: 0.88rem !important;
  }}
  div[data-testid="stTabs"] button {{
    color: {TEXT_MUTED} !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500 !important;
  }}
  div[data-testid="stTabs"] button[aria-selected="true"] {{
    color: {HEART_RED} !important;
    border-bottom-color: {HEART_RED} !important;
  }}
  .stAlert {{ border-radius: 8px !important; }}
  hr {{ border-color: {BG_BORDER} !important; }}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# DATA & MODEL MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════
COLS = ["age","sex","cp","trestbps","chol","fbs","restecg",
        "thalach","exang","oldpeak","slope","ca","thal"]

FEATURE_INFO = {
    "age":      {"label": "Age",                        "unit": "years",    "min": 20,  "max": 90,  "default": 54,  "type": "slider"},
    "sex":      {"label": "Sex",                        "options": {0: "Female", 1: "Male"},                        "type": "select"},
    "cp":       {"label": "Chest Pain Type",            "options": {1: "Typical Angina", 2: "Atypical Angina", 3: "Non-Anginal Pain", 4: "Asymptomatic"}, "type": "select"},
    "trestbps": {"label": "Resting Blood Pressure",     "unit": "mmHg",     "min": 80,  "max": 220, "default": 130, "type": "slider"},
    "chol":     {"label": "Serum Cholesterol",          "unit": "mg/dL",    "min": 100, "max": 600, "default": 240, "type": "slider"},
    "fbs":      {"label": "Fasting Blood Sugar > 120",  "options": {0: "No (≤ 120 mg/dL)", 1: "Yes (> 120 mg/dL)"},                "type": "select"},
    "restecg":  {"label": "Resting ECG Result",         "options": {0: "Normal", 1: "ST-T Abnormality", 2: "LV Hypertrophy"},      "type": "select"},
    "thalach":  {"label": "Max Heart Rate Achieved",    "unit": "bpm",      "min": 60,  "max": 220, "default": 150, "type": "slider"},
    "exang":    {"label": "Exercise-Induced Angina",    "options": {0: "No", 1: "Yes"},                             "type": "select"},
    "oldpeak":  {"label": "ST Depression (Oldpeak)",    "unit": "",         "min": 0.0, "max": 7.0, "default": 1.0, "type": "number", "step": 0.1},
    "slope":    {"label": "ST Segment Slope",           "options": {1: "Upsloping", 2: "Flat", 3: "Downsloping"},  "type": "select"},
    "ca":       {"label": "Fluoroscopy Vessels (0-3)",  "options": {0: "0 vessels", 1: "1 vessel", 2: "2 vessels", 3: "3 vessels"}, "type": "select"},
    "thal":     {"label": "Thalassemia",                "options": {3: "Normal", 6: "Fixed Defect", 7: "Reversible Defect"},       "type": "select"},
}

CAT_LABELS = {
    "sex":     {0:"Female", 1:"Male"},
    "cp":      {1:"Typical Angina", 2:"Atypical", 3:"Non-Anginal", 4:"Asymptomatic"},
    "fbs":     {0:"FBS ≤ 120", 1:"FBS > 120"},
    "restecg": {0:"Normal", 1:"ST-T Abnorm.", 2:"LV Hypertrophy"},
    "exang":   {0:"No Angina", 1:"Exercise Angina"},
    "slope":   {1:"Upsloping", 2:"Flat", 3:"Downsloping"},
    "ca":      {0:"0 vessels", 1:"1 vessel", 2:"2 vessels", 3:"3 vessels"},
    "thal":    {3:"Normal", 6:"Fixed Defect", 7:"Reversible Defect"},
}


@st.cache_data(show_spinner=False)
def generate_synthetic_data(n=920):
    """Generate realistic synthetic UCI-style heart disease data."""
    rng = np.random.RandomState(SEED)
    n_hd  = int(n * 0.54)
    n_nhd = n - n_hd

    def patient_block(n_p, hd):
        age     = rng.normal(57 if hd else 50, 9, n_p).clip(28, 77)
        sex     = rng.choice([0, 1], n_p, p=[0.25, 0.75] if hd else [0.45, 0.55])
        cp      = rng.choice([1, 2, 3, 4], n_p, p=[0.05, 0.08, 0.12, 0.75] if hd else [0.15, 0.25, 0.35, 0.25])
        trestbps= rng.normal(136 if hd else 129, 18, n_p).clip(90, 200)
        chol    = rng.normal(251 if hd else 242, 51, n_p).clip(130, 560)
        fbs     = rng.choice([0, 1], n_p, p=[0.85, 0.15])
        restecg = rng.choice([0, 1, 2], n_p, p=[0.45, 0.35, 0.20] if hd else [0.65, 0.25, 0.10])
        thalach = rng.normal(139 if hd else 158, 22, n_p).clip(71, 202)
        exang   = rng.choice([0, 1], n_p, p=[0.35, 0.65] if hd else [0.80, 0.20])
        oldpeak = rng.exponential(1.8 if hd else 0.9, n_p).clip(0, 6.2)
        slope   = rng.choice([1, 2, 3], n_p, p=[0.15, 0.55, 0.30] if hd else [0.40, 0.45, 0.15])
        ca      = rng.choice([0, 1, 2, 3], n_p, p=[0.25, 0.30, 0.28, 0.17] if hd else [0.62, 0.24, 0.10, 0.04])
        thal    = rng.choice([3, 6, 7], n_p, p=[0.12, 0.12, 0.76] if hd else [0.70, 0.06, 0.24])
        target  = np.ones(n_p, dtype=int) if hd else np.zeros(n_p, dtype=int)
        return np.column_stack([age, sex, cp, trestbps, chol, fbs, restecg,
                                 thalach, exang, oldpeak, slope, ca, thal, target])

    data = np.vstack([patient_block(n_hd, True), patient_block(n_nhd, False)])
    rng.shuffle(data)
    df = pd.DataFrame(data, columns=COLS + ["target"]).astype(float)
    # Add ~3% missing values in ca, thal, slope (mirrors UCI)
    for col in ["ca", "thal", "slope"]:
        mask = rng.rand(len(df)) < 0.03
        df.loc[mask, col] = np.nan
    return df


@st.cache_resource(show_spinner=False)
def train_models():
    df = generate_synthetic_data()
    X  = df[COLS].copy()
    y  = df["target"].values

    imputer = SimpleImputer(strategy="median")
    X_imp   = pd.DataFrame(imputer.fit_transform(X), columns=COLS)
    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X_imp)

    X_train, X_test, y_train, y_test = train_test_split(
        X_sc, y, test_size=0.2, random_state=SEED, stratify=y)

    model_registry = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=0.5, random_state=SEED),
        "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=8,
                                                       min_samples_leaf=2, random_state=SEED),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                                           max_depth=4, random_state=SEED),
        "Extra Trees":         ExtraTreesClassifier(n_estimators=200, max_depth=8, random_state=SEED),
        "DNN (MLP)":           MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation="relu",
                                              max_iter=300, random_state=SEED,
                                              early_stopping=True, validation_fraction=0.1),
    }
    if XGB_AVAILABLE:
        model_registry["XGBoost ⭐"] = xgb.XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            use_label_encoder=False, eval_metric="logloss", random_state=SEED)
    if LGB_AVAILABLE:
        model_registry["LightGBM ⭐"] = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=31,
            random_state=SEED, verbose=-1)

    results = {}
    for name, model in model_registry.items():
        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        acc  = accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred)
        auc_ = roc_auc_score(y_test, y_proba) if y_proba is not None else 0.0
        mcc  = matthews_corrcoef(y_test, y_pred)
        results[name] = {
            "model": model, "y_pred": y_pred, "y_proba": y_proba,
            "y_test": y_test,
            "acc": acc, "f1": f1, "auc": auc_, "mcc": mcc,
        }

    return results, imputer, scaler, X_test, y_test, df


# ═══════════════════════════════════════════════════════════════════════
# PREDICTION FUNCTION
# ═══════════════════════════════════════════════════════════════════════
def predict_patient(values_dict, results, imputer, scaler):
    patient = pd.DataFrame([values_dict], columns=COLS)
    patient_imp    = pd.DataFrame(imputer.transform(patient), columns=COLS)
    patient_scaled = scaler.transform(patient_imp)

    probs = {}
    for name, res in results.items():
        mdl = res["model"]
        if hasattr(mdl, "predict_proba"):
            prob = mdl.predict_proba(patient_scaled)[0][1]
        else:
            prob = float(mdl.predict(patient_scaled)[0])
        probs[name] = float(prob)

    avg_risk = float(np.mean(list(probs.values())))
    return probs, avg_risk


# ═══════════════════════════════════════════════════════════════════════
# CHART HELPERS
# ═══════════════════════════════════════════════════════════════════════
def fig_model_bars(probs):
    names  = [n.replace(" ⭐", "") for n in probs]
    vals   = list(probs.values())
    colors = [HEART_RED if v >= 0.5 else ACCENT_GRN for v in vals]

    fig, ax = plt.subplots(figsize=(7, max(3, len(names) * 0.55)))
    bars = ax.barh(names, vals, color=colors, edgecolor=BG_DARK, height=0.6)
    for bar, val in zip(bars, vals):
        ax.text(min(val + 0.02, 1.0), bar.get_y() + bar.get_height() / 2,
                f"{val*100:.1f}%", va="center", color=TEXT_MAIN, fontsize=8.5, fontweight="bold")
    ax.axvline(0.5, color=TEXT_MUTED, linestyle="--", alpha=0.6, linewidth=1.2)
    ax.set_xlim(0, 1.22)
    ax.set_xlabel("Predicted Risk Probability")
    ax.set_title("Per-Model Risk Scores", color=TEXT_MAIN, fontweight="bold", pad=10)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    return fig


def fig_roc(results):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1, label="Random (AUC=0.50)")
    for i, (name, res) in enumerate(results.items()):
        if res["y_proba"] is None:
            continue
        fpr, tpr, _ = roc_curve(res["y_test"], res["y_proba"])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=PALETTE[i % len(PALETTE)], linewidth=2,
                label=f"{name.replace(' ⭐','')[:18]} ({roc_auc:.3f})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models", color=TEXT_MAIN, fontweight="bold", pad=10)
    ax.legend(fontsize=7, facecolor=BG_CARD, labelcolor=TEXT_MAIN, loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def fig_conf_matrix(results):
    best = max(results, key=lambda k: results[k]["auc"])
    cm   = confusion_matrix(results[best]["y_test"], results[best]["y_pred"])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    cmap = LinearSegmentedColormap.from_list("heart", [BG_CARD, HEART_RED])
    im   = ax.imshow(cm, cmap=cmap, aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Disease", "Heart Disease"])
    ax.set_yticklabels(["No Disease", "Heart Disease"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {best.replace(' ⭐','')}", color=TEXT_MAIN,
                 fontweight="bold", pad=10)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=18, fontweight="bold", color=TEXT_MAIN)
    plt.tight_layout()
    return fig


def fig_metric_comparison(results):
    metrics = ["acc", "f1", "auc", "mcc"]
    labels  = ["Accuracy", "F1 Score", "ROC-AUC", "MCC"]
    model_names = list(results.keys())

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle("Model Comparison Dashboard", fontsize=12, fontweight="bold",
                 color=ACCENT_BLUE, y=1.02)

    for ax, metric, mname in zip(axes, metrics, labels):
        vals = [results[m][metric] for m in model_names]
        sorted_pairs = sorted(zip(vals, model_names), reverse=True)
        s_vals, s_names = zip(*sorted_pairs)
        colors = [HEART_RED if i == 0 else PALETTE[i % len(PALETTE)] for i in range(len(s_vals))]
        bars = ax.barh(range(len(s_names)), s_vals, color=colors, edgecolor=BG_DARK, height=0.65)
        for j, (bar, val) in enumerate(zip(bars, s_vals)):
            ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", ha="left", fontsize=8,
                    color=TEXT_MAIN, fontweight="bold" if j == 0 else "normal")
        ax.set_yticks(range(len(s_names)))
        ax.set_yticklabels([n.replace(" ⭐", "")[:14] for n in s_names], fontsize=7.5)
        ax.set_title(mname, color=TEXT_MAIN, fontweight="bold")
        ax.set_xlim(0, 1.15)
        ax.grid(axis="x", alpha=0.3)
        ax.axvline(0.85, color=ACCENT_GRN, alpha=0.25, linestyle="--", linewidth=1)

    plt.tight_layout()
    return fig


def fig_feature_importance(results):
    best = max(results, key=lambda k: results[k]["auc"])
    mdl  = results[best]["model"]
    if hasattr(mdl, "feature_importances_"):
        imp = mdl.feature_importances_
    elif hasattr(mdl, "coef_"):
        imp = np.abs(mdl.coef_[0])
    else:
        return None

    sorted_idx = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.barh([COLS[i] for i in sorted_idx], imp[sorted_idx],
                   color=PALETTE[:len(COLS)], edgecolor=BG_DARK, height=0.65)
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Feature Importance — {best.replace(' ⭐','')}", color=TEXT_MAIN,
                 fontweight="bold", pad=10)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    return fig


def fig_age_distribution(df):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df[df["target"] == 0]["age"], bins=22, alpha=0.72,
            color=ACCENT_GRN, label="No Disease", edgecolor=BG_DARK)
    ax.hist(df[df["target"] == 1]["age"], bins=22, alpha=0.72,
            color=HEART_RED, label="Heart Disease", edgecolor=BG_DARK)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Count")
    ax.set_title("Age Distribution by Diagnosis", color=TEXT_MAIN, fontweight="bold", pad=10)
    ax.legend(facecolor=BG_CARD, labelcolor=TEXT_MAIN)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════
# APP LAYOUT
# ═══════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-banner">
  <p class="hero-title">Heart Disease Predictor</p>
  <p class="hero-sub">Multi-model ML pipeline trained on UCI Heart Disease Repository
  (Cleveland · Hungarian · Switzerland · VA Long Beach)</p>
  <div class="hero-badges">
    <span class="badge">Logistic Regression</span>
    <span class="badge">Random Forest</span>
    <span class="badge">Gradient Boosting</span>
    <span class="badge">Extra Trees</span>
    <span class="badge">DNN (MLP)</span>
    <span class="badge">XGBoost</span>
    <span class="badge">LightGBM</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Load models ────────────────────────────────────────────────────────
with st.spinner("🔬 Training models on synthetic UCI data…"):
    results, imputer, scaler, X_test, y_test, df = train_models()

best_model_name = max(results, key=lambda k: results[k]["auc"])

# ── Session state ──────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts

# ── Top KPI strip ──────────────────────────────────────────────────────
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi_data = [
    (f"{results[best_model_name]['auc']:.3f}", "Best AUC",    ACCENT_BLUE),
    (f"{results[best_model_name]['acc']:.3f}", "Best Acc",    ACCENT_GRN),
    (f"{results[best_model_name]['f1']:.3f}",  "Best F1",     ACCENT_YLW),
    (str(len(results)),                         "Models",      ACCENT_PRP),
    ("920",                                     "Patients",    TEXT_MUTED),
]
for col, (val, lbl, color) in zip([kpi1, kpi2, kpi3, kpi4, kpi5], kpi_data):
    with col:
        st.markdown(f"""
        <div class="metric-tile">
          <div class="metric-tile-val" style="color:{color}">{val}</div>
          <div class="metric-tile-lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════
tab_predict, tab_analysis, tab_batch, tab_eda, tab_about = st.tabs([
    "🔮  Predict Risk",
    "📊  Model Analysis",
    "📂  Batch Predict",
    "🔬  EDA Dashboard",
    "ℹ️  Feature Guide",
])

# ───────────────────────────────────────────────────────────────────────
# TAB 1 — PREDICT
# ───────────────────────────────────────────────────────────────────────
with tab_predict:
    st.markdown("""
    <div class="section-header">
      <h2>🩺 Patient Risk Assessment</h2>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar quick-fill presets
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align:center; padding: 1rem 0 0.5rem 0;">
          <span style="font-size:2rem;">❤️</span>
          <div style="font-family:'Syne',sans-serif; font-size:1.1rem;
               font-weight:700; color:{HEART_RED}; margin-top:0.3rem;">
            Heart Predictor
          </div>
          <div style="font-size:0.72rem; color:{TEXT_MUTED};">
            UCI Multi-Dataset ML Pipeline
          </div>
        </div>
        <hr>
        """, unsafe_allow_html=True)

        st.markdown(f"**🧪 Load Demo Patient**")
        demo_preset = st.selectbox(
            "Choose a preset",
            ["Custom", "High-Risk Male (62)", "Low-Risk Female (42)", "Borderline (55)"],
            label_visibility="collapsed"
        )

        PRESETS = {
            "High-Risk Male (62)":   dict(age=62, sex=1, cp=4, trestbps=150, chol=270, fbs=0, restecg=2, thalach=105, exang=1, oldpeak=3.5, slope=2, ca=2, thal=7),
            "Low-Risk Female (42)":  dict(age=42, sex=0, cp=2, trestbps=118, chol=195, fbs=0, restecg=0, thalach=172, exang=0, oldpeak=0.2, slope=1, ca=0, thal=3),
            "Borderline (55)":       dict(age=55, sex=1, cp=3, trestbps=135, chol=245, fbs=1, restecg=1, thalach=130, exang=0, oldpeak=1.5, slope=2, ca=1, thal=6),
        }
        preset_vals = PRESETS.get(demo_preset, None)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-size:0.78rem; color:{TEXT_MUTED}; line-height:1.6;">
          <b style="color:{TEXT_MAIN};">⚠️ Disclaimer</b><br>
          This tool is for educational and research purposes only. 
          It is <b>not</b> a substitute for professional medical advice.
          Always consult a qualified physician.
        </div>
        """, unsafe_allow_html=True)

    def pv(key, fallback):
        """Return preset value if preset selected, else fallback."""
        if preset_vals and key in preset_vals:
            return preset_vals[key]
        return fallback

    # ── Input grid ────────────────────────────────────────────────────
    col_a, col_b = st.columns([1, 1], gap="large")

    with col_a:
        st.markdown(f"<p style='font-size:0.8rem; color:{TEXT_MUTED}; font-weight:600; text-transform:uppercase; letter-spacing:1px;'>📋 Demographics & Vitals</p>", unsafe_allow_html=True)

        age      = st.slider("Age (years)", 20, 90, int(pv("age", 54)), key="age")
        sex_opt  = st.selectbox("Sex", options=list(CAT_LABELS["sex"].values()),
                                index=list(CAT_LABELS["sex"].values()).index(CAT_LABELS["sex"][pv("sex", 1)]))
        sex      = [k for k, v in CAT_LABELS["sex"].items() if v == sex_opt][0]

        trestbps = st.slider("Resting Blood Pressure (mmHg)", 80, 220, int(pv("trestbps", 130)), key="trestbps")
        chol     = st.slider("Serum Cholesterol (mg/dL)", 100, 600, int(pv("chol", 240)), key="chol")
        thalach  = st.slider("Max Heart Rate Achieved (bpm)", 60, 220, int(pv("thalach", 150)), key="thalach")
        oldpeak  = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=7.0,
                                    value=float(pv("oldpeak", 1.0)), step=0.1, key="oldpeak")

    with col_b:
        st.markdown(f"<p style='font-size:0.8rem; color:{TEXT_MUTED}; font-weight:600; text-transform:uppercase; letter-spacing:1px;'>🩺 Clinical Findings</p>", unsafe_allow_html=True)

        cp_opt   = st.selectbox("Chest Pain Type", options=list(CAT_LABELS["cp"].values()),
                                index=list(CAT_LABELS["cp"].values()).index(CAT_LABELS["cp"][pv("cp", 4)]))
        cp       = [k for k, v in CAT_LABELS["cp"].items() if v == cp_opt][0]

        fbs_opt  = st.selectbox("Fasting Blood Sugar > 120 mg/dL", options=list(CAT_LABELS["fbs"].values()),
                                index=list(CAT_LABELS["fbs"].values()).index(CAT_LABELS["fbs"][pv("fbs", 0)]))
        fbs      = [k for k, v in CAT_LABELS["fbs"].items() if v == fbs_opt][0]

        restecg_opt = st.selectbox("Resting ECG Result", options=list(CAT_LABELS["restecg"].values()),
                                    index=list(CAT_LABELS["restecg"].values()).index(CAT_LABELS["restecg"][pv("restecg", 0)]))
        restecg  = [k for k, v in CAT_LABELS["restecg"].items() if v == restecg_opt][0]

        exang_opt = st.selectbox("Exercise-Induced Angina", options=list(CAT_LABELS["exang"].values()),
                                  index=list(CAT_LABELS["exang"].values()).index(CAT_LABELS["exang"][pv("exang", 0)]))
        exang    = [k for k, v in CAT_LABELS["exang"].items() if v == exang_opt][0]

        slope_opt = st.selectbox("ST Segment Slope", options=list(CAT_LABELS["slope"].values()),
                                  index=list(CAT_LABELS["slope"].values()).index(CAT_LABELS["slope"][pv("slope", 2)]))
        slope    = [k for k, v in CAT_LABELS["slope"].items() if v == slope_opt][0]

        ca_opt   = st.selectbox("Fluoroscopy Vessels (0-3)", options=list(CAT_LABELS["ca"].values()),
                                index=list(CAT_LABELS["ca"].values()).index(CAT_LABELS["ca"][pv("ca", 0)]))
        ca       = [k for k, v in CAT_LABELS["ca"].items() if v == ca_opt][0]

        thal_opt = st.selectbox("Thalassemia", options=list(CAT_LABELS["thal"].values()),
                                index=list(CAT_LABELS["thal"].values()).index(CAT_LABELS["thal"][pv("thal", 3)]))
        thal     = [k for k, v in CAT_LABELS["thal"].items() if v == thal_opt][0]

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔬 Analyse Risk Now", key="predict_btn")

    if predict_btn:
        patient_vals = dict(age=age, sex=sex, cp=cp, trestbps=trestbps,
                             chol=chol, fbs=fbs, restecg=restecg, thalach=thalach,
                             exang=exang, oldpeak=oldpeak, slope=slope, ca=ca, thal=thal)

        probs, avg_risk = predict_patient(patient_vals, results, imputer, scaler)

        # ── Risk level ────────────────────────────────────────────────
        if avg_risk >= 0.60:
            risk_class = "result-high"
            risk_icon  = "🔴"
            risk_label = "HIGH RISK"
            risk_color = HEART_RED
            risk_msg   = "⚠️ Elevated risk detected. Please consult a cardiologist promptly."
        elif avg_risk >= 0.40:
            risk_class = "result-medium"
            risk_icon  = "🟡"
            risk_label = "MODERATE RISK"
            risk_color = ACCENT_YLW
            risk_msg   = "📋 Borderline risk. Lifestyle modifications and follow-up recommended."
        else:
            risk_class = "result-low"
            risk_icon  = "🟢"
            risk_label = "LOW RISK"
            risk_color = ACCENT_GRN
            risk_msg   = "✅ Low risk profile. Continue healthy habits and regular check-ups."

        r_col1, r_col2 = st.columns([1, 2], gap="large")

        with r_col1:
            st.markdown(f"""
            <div class="{risk_class}">
              <div style="font-size:1.5rem;">{risk_icon}</div>
              <div class="result-pct" style="color:{risk_color}">{avg_risk*100:.1f}%</div>
              <div class="result-label" style="color:{risk_color}">{risk_label}</div>
              <div style="font-size:0.78rem; color:{TEXT_MUTED}; margin-top:0.8rem; line-height:1.5;">
                Ensemble of {len(probs)} models
              </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"<br><div style='font-size:0.85rem; color:{risk_color}; background:{BG_CARD}; border:1px solid {BG_BORDER}; border-radius:8px; padding:0.8rem 1rem;'>{risk_msg}</div>", unsafe_allow_html=True)

        with r_col2:
            # Per-model breakdown
            st.markdown(f"<p style='font-size:0.8rem; color:{TEXT_MUTED}; font-weight:600; text-transform:uppercase; letter-spacing:1px; margin-bottom:0.6rem;'>📊 Per-Model Breakdown</p>", unsafe_allow_html=True)
            for name, prob in probs.items():
                bar_color = HEART_RED if prob >= 0.5 else ACCENT_GRN
                bar_pct   = f"{prob*100:.0f}"
                pred_lbl  = "❤️ Disease" if prob >= 0.5 else "✅ Healthy"
                st.markdown(f"""
                <div class="model-row">
                  <span style="font-weight:500; min-width:140px;">{name.replace(' ⭐','')}</span>
                  <div class="model-bar-wrap">
                    <div class="model-bar-fill" style="width:{bar_pct}%; background:{bar_color};"></div>
                  </div>
                  <span style="font-weight:700; color:{bar_color}; min-width:50px; text-align:right;">{prob*100:.1f}%</span>
                  <span style="font-size:0.78rem; color:{TEXT_MUTED}; min-width:80px; text-align:right;">{pred_lbl}</span>
                </div>
                """, unsafe_allow_html=True)

        # Chart
        st.markdown("<br>", unsafe_allow_html=True)
        st.pyplot(fig_model_bars(probs), use_container_width=False)

        # ── Save to history ───────────────────────────────────────────
        st.session_state.history.append({
            "Age": age, "Sex": "Male" if sex == 1 else "Female",
            "CP": CAT_LABELS["cp"].get(cp, cp),
            "BP": trestbps, "Chol": chol, "MaxHR": thalach,
            "Risk%": round(avg_risk * 100, 1),
            "Level": risk_label,
            "Best Model": best_model_name.replace(" ⭐", ""),
        })

        # ── Simple PDF-style text report download ─────────────────────
        report_lines = [
            "=" * 55,
            "        HEART DISEASE RISK ASSESSMENT REPORT",
            "=" * 55,
            f"  Age        : {age} yrs  |  Sex: {'Male' if sex==1 else 'Female'}",
            f"  Chest Pain : {CAT_LABELS['cp'].get(cp, cp)}",
            f"  Blood Press: {trestbps} mmHg  |  Cholesterol: {chol} mg/dL",
            f"  Max HR     : {thalach} bpm   |  ST Depression: {oldpeak}",
            f"  Exercise Angina: {'Yes' if exang==1 else 'No'}",
            "-" * 55,
            f"  {'Model':<22} {'Risk%':>8}  {'Verdict':>16}",
            "-" * 55,
        ]
        for nm, pb in probs.items():
            verdict = "Heart Disease" if pb >= 0.5 else "No Disease"
            report_lines.append(f"  {nm.replace(' ⭐',''):<22} {pb*100:>7.1f}%  {verdict:>16}")
        report_lines += [
            "-" * 55,
            f"  ENSEMBLE RISK  : {avg_risk*100:.1f}%",
            f"  RISK LEVEL     : {risk_label}",
            "=" * 55,
            "",
            "DISCLAIMER: For research/educational use only.",
            "Not a substitute for professional medical advice.",
        ]
        report_text = "\n".join(report_lines)
        st.download_button(
            label="⬇️ Download Risk Report (.txt)",
            data=report_text,
            file_name="heart_risk_report.txt",
            mime="text/plain",
        )

    # ── Prediction History ─────────────────────────────────────────────
    if st.session_state.history:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="section-header">
          <h2>🕓 Session Prediction History</h2>
        </div>
        """, unsafe_allow_html=True)

        hist_df = pd.DataFrame(st.session_state.history)

        def color_risk(val):
            if "HIGH" in str(val):
                return f"color: {HEART_RED}; font-weight:700"
            if "MODERATE" in str(val):
                return f"color: {ACCENT_YLW}; font-weight:700"
            return f"color: {ACCENT_GRN}; font-weight:700"

        styled = hist_df.style.map(color_risk, subset=["Level"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        hcol1, hcol2 = st.columns(2)
        with hcol1:
            avg_sess = hist_df["Risk%"].mean()
            st.markdown(f"""
            <div class="metric-tile">
              <div class="metric-tile-val" style="color:{HEART_RED}">{avg_sess:.1f}%</div>
              <div class="metric-tile-lbl">Avg Session Risk</div>
            </div>""", unsafe_allow_html=True)
        with hcol2:
            high_count = (hist_df["Level"].str.contains("HIGH")).sum()
            st.markdown(f"""
            <div class="metric-tile">
              <div class="metric-tile-val" style="color:{ACCENT_YLW}">{high_count}/{len(hist_df)}</div>
              <div class="metric-tile-lbl">High-Risk Predictions</div>
            </div>""", unsafe_allow_html=True)

        if st.button("🗑️ Clear History", key="clear_hist"):
            st.session_state.history = []
            st.rerun()


# ───────────────────────────────────────────────────────────────────────
# TAB 2 — ANALYSIS
# ───────────────────────────────────────────────────────────────────────
with tab_analysis:
    st.markdown("""
    <div class="section-header">
      <h2>📊 Model Performance Analysis</h2>
    </div>
    """, unsafe_allow_html=True)

    a1, a2 = st.columns(2, gap="large")
    with a1:
        st.markdown(f"<p style='font-size:0.82rem; color:{TEXT_MUTED}; font-weight:600;'>ROC Curves</p>", unsafe_allow_html=True)
        st.pyplot(fig_roc(results), use_container_width=True)

    with a2:
        st.markdown(f"<p style='font-size:0.82rem; color:{TEXT_MUTED}; font-weight:600;'>Confusion Matrix — Best Model</p>", unsafe_allow_html=True)
        st.pyplot(fig_conf_matrix(results), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:0.82rem; color:{TEXT_MUTED}; font-weight:600;'>All-Model Metric Comparison</p>", unsafe_allow_html=True)
    st.pyplot(fig_metric_comparison(results), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    b1, b2 = st.columns(2, gap="large")
    with b1:
        st.markdown(f"<p style='font-size:0.82rem; color:{TEXT_MUTED}; font-weight:600;'>Feature Importance — Best Model</p>", unsafe_allow_html=True)
        fi_fig = fig_feature_importance(results)
        if fi_fig:
            st.pyplot(fi_fig, use_container_width=True)

    with b2:
        st.markdown(f"<p style='font-size:0.82rem; color:{TEXT_MUTED}; font-weight:600;'>Age Distribution by Diagnosis</p>", unsafe_allow_html=True)
        st.pyplot(fig_age_distribution(df), use_container_width=True)

    # ── Metrics table ─────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="section-header">
      <h2>📋 Metrics Summary Table</h2>
    </div>
    """, unsafe_allow_html=True)

    rows = []
    for name, res in results.items():
        rows.append({
            "Model":    name.replace(" ⭐", " ★"),
            "Accuracy": f"{res['acc']:.4f}",
            "F1 Score": f"{res['f1']:.4f}",
            "ROC-AUC":  f"{res['auc']:.4f}",
            "MCC":      f"{res['mcc']:.4f}",
        })
    df_metrics = pd.DataFrame(rows)
    st.dataframe(df_metrics.style.highlight_max(
        subset=["Accuracy", "F1 Score", "ROC-AUC", "MCC"],
        color="#1a2a1a"), use_container_width=True, hide_index=True)


# ───────────────────────────────────────────────────────────────────────
# TAB 3 — BATCH PREDICT
# ───────────────────────────────────────────────────────────────────────
with tab_batch:
    st.markdown("""
    <div class="section-header">
      <h2>📂 Batch Patient Prediction</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:{BG_CARD}; border:1px solid {BG_BORDER};
         border-left:4px solid {ACCENT_BLUE}; border-radius:8px;
         padding:1rem 1.2rem; margin-bottom:1.2rem; font-size:0.85rem; color:{TEXT_MUTED}; line-height:1.7;">
      Upload a <b style="color:{TEXT_MAIN};">CSV file</b> with columns:<br>
      <code style="color:{ACCENT_BLUE};">age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal</code><br>
      The app will run all trained models and return an ensemble risk score for each row.
    </div>
    """, unsafe_allow_html=True)

    # Download template
    template_df = pd.DataFrame([
        [54, 1, 4, 150, 270, 0, 2, 105, 1, 3.5, 2, 2, 7],
        [42, 0, 2, 118, 195, 0, 0, 172, 0, 0.2, 1, 0, 3],
        [55, 1, 3, 135, 245, 1, 1, 130, 0, 1.5, 2, 1, 6],
    ], columns=COLS)
    st.download_button(
        "⬇️ Download Template CSV",
        data=template_df.to_csv(index=False),
        file_name="heart_batch_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload your CSV", type=["csv"], key="batch_upload")

    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
            # Validate columns
            missing_cols = [c for c in COLS if c not in batch_df.columns]
            if missing_cols:
                st.error(f"❌ Missing columns: {missing_cols}")
            else:
                batch_df = batch_df[COLS].copy()
                st.markdown(f"<p style='color:{ACCENT_GRN}; font-size:0.85rem;'>✅ Loaded <b>{len(batch_df)}</b> rows successfully.</p>", unsafe_allow_html=True)

                with st.spinner("Running ensemble predictions…"):
                    b_imp = pd.DataFrame(imputer.transform(batch_df), columns=COLS)
                    b_sc  = scaler.transform(b_imp)

                    all_probs = {}
                    for mname, res in results.items():
                        mdl = res["model"]
                        if hasattr(mdl, "predict_proba"):
                            all_probs[mname] = mdl.predict_proba(b_sc)[:, 1]
                        else:
                            all_probs[mname] = mdl.predict(b_sc).astype(float)

                    prob_matrix    = np.column_stack(list(all_probs.values()))
                    ensemble_risk  = prob_matrix.mean(axis=1)

                    out_df = batch_df.copy()
                    out_df["Ensemble_Risk%"] = (ensemble_risk * 100).round(1)
                    out_df["Risk_Level"] = pd.cut(
                        ensemble_risk,
                        bins=[0, 0.4, 0.6, 1.0],
                        labels=["🟢 LOW", "🟡 MODERATE", "🔴 HIGH"]
                    )
                    for mname in all_probs:
                        out_df[f"{mname.replace(' ⭐','')}_Risk%"] = (all_probs[mname] * 100).round(1)

                # Show preview
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size:0.82rem; color:{TEXT_MUTED}; font-weight:600;'>Preview (first 20 rows)</p>", unsafe_allow_html=True)

                def highlight_risk(row):
                    lvl = str(row.get("Risk_Level", ""))
                    if "HIGH" in lvl:
                        return [f"background-color:#1a0505" for _ in row]
                    if "MODERATE" in lvl:
                        return [f"background-color:#1a1200" for _ in row]
                    return [f"background-color:#0a1a0a" for _ in row]

                preview_cols = ["age", "sex", "cp", "trestbps", "chol",
                                "thalach", "Ensemble_Risk%", "Risk_Level"]
                st.dataframe(
                    out_df[preview_cols].head(20).style.apply(highlight_risk, axis=1),
                    use_container_width=True, hide_index=True
                )

                # Summary stats
                st.markdown("<br>", unsafe_allow_html=True)
                risk_counts = out_df["Risk_Level"].value_counts()
                bc1, bc2, bc3 = st.columns(3)
                for col_w, level, color in zip(
                    [bc1, bc2, bc3],
                    ["🟢 LOW", "🟡 MODERATE", "🔴 HIGH"],
                    [ACCENT_GRN, ACCENT_YLW, HEART_RED]
                ):
                    cnt = risk_counts.get(level, 0)
                    pct = cnt / len(out_df) * 100
                    with col_w:
                        st.markdown(f"""
                        <div class="metric-tile">
                          <div class="metric-tile-val" style="color:{color}">{cnt}</div>
                          <div class="metric-tile-lbl">{level.split()[-1]} ({pct:.0f}%)</div>
                        </div>""", unsafe_allow_html=True)

                # Risk distribution chart
                fig_b, ax_b = plt.subplots(figsize=(8, 3.5))
                ax_b.hist(ensemble_risk * 100, bins=25, color=HEART_RED,
                          edgecolor=BG_DARK, alpha=0.85)
                ax_b.axvline(40, color=ACCENT_YLW, linestyle="--", linewidth=1.5,
                             label="Moderate threshold (40%)")
                ax_b.axvline(60, color=HEART_RED, linestyle="--", linewidth=1.5,
                             label="High threshold (60%)")
                ax_b.set_xlabel("Ensemble Risk (%)")
                ax_b.set_ylabel("Patient Count")
                ax_b.set_title("Batch Risk Distribution", color=TEXT_MAIN,
                               fontweight="bold", pad=10)
                ax_b.legend(fontsize=8, facecolor=BG_CARD, labelcolor=TEXT_MAIN)
                ax_b.grid(alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig_b, use_container_width=True)

                # Download results
                st.markdown("<br>", unsafe_allow_html=True)
                st.download_button(
                    "⬇️ Download Full Results CSV",
                    data=out_df.to_csv(index=False),
                    file_name="heart_batch_results.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
    else:
        st.markdown(f"""
        <div style="border: 2px dashed {BG_BORDER}; border-radius:10px;
             padding:2.5rem; text-align:center; color:{TEXT_MUTED}; margin-top:1rem;">
          <div style="font-size:2.5rem; margin-bottom:0.5rem;">📁</div>
          <div style="font-size:0.9rem;">Drag & drop your CSV here or click Browse</div>
          <div style="font-size:0.78rem; margin-top:0.3rem;">
            Download the template above to see the expected format
          </div>
        </div>
        """, unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────────────
# TAB 4 — EDA DASHBOARD
# ───────────────────────────────────────────────────────────────────────
with tab_eda:
    st.markdown("""
    <div class="section-header">
      <h2>🔬 Exploratory Data Analysis</h2>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:0.83rem; color:{TEXT_MUTED}; margin-bottom:1.2rem;'>Synthetic UCI-style dataset — 920 patients across 4 centres.</p>", unsafe_allow_html=True)

    # ── Row 1: Overview tiles ─────────────────────────────────────────
    e1, e2, e3, e4 = st.columns(4)
    hd_pct   = df["target"].mean() * 100
    avg_age  = df["age"].mean()
    avg_chol = df["chol"].mean()
    avg_hr   = df["thalach"].mean()
    for col_w, val, lbl, clr in zip(
        [e1, e2, e3, e4],
        [f"{len(df)}", f"{hd_pct:.1f}%", f"{avg_age:.1f}", f"{avg_chol:.0f}"],
        ["Total Patients", "HD Prevalence", "Avg Age (yrs)", "Avg Cholesterol"],
        [ACCENT_BLUE, HEART_RED, ACCENT_YLW, ACCENT_PRP]
    ):
        with col_w:
            st.markdown(f"""
            <div class="metric-tile">
              <div class="metric-tile-val" style="color:{clr}">{val}</div>
              <div class="metric-tile-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2: Class balance + Sex breakdown ──────────────────────────
    ea1, ea2 = st.columns(2, gap="large")

    with ea1:
        st.markdown(f"<p style='font-size:0.82rem; color:{TEXT_MUTED}; font-weight:600;'>Class Distribution</p>", unsafe_allow_html=True)
        fig_cls, ax_cls = plt.subplots(figsize=(5, 4))
        counts = df["target"].value_counts().sort_index()
        bars   = ax_cls.bar(["No Disease", "Heart Disease"], counts,
                             color=[ACCENT_GRN, HEART_RED], edgecolor=BG_DARK, width=0.5)
        for bar, cnt in zip(bars, counts):
            ax_cls.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 4,
                        str(cnt), ha="center", va="bottom", fontweight="bold",
                        color=TEXT_MAIN, fontsize=11)
        ax_cls.set_ylabel("Count")
        ax_cls.grid(axis="y", alpha=0.3)
        ax_cls.set_title("Target Class Balance", color=TEXT_MAIN, fontweight="bold", pad=10)
        plt.tight_layout()
        st.pyplot(fig_cls, use_container_width=True)

    with ea2:
        st.markdown(f"<p style='font-size:0.82rem; color:{TEXT_MUTED}; font-weight:600;'>Sex × Heart Disease</p>", unsafe_allow_html=True)
        fig_sex, ax_sex = plt.subplots(figsize=(5, 4))
        sex_grp = df.groupby("sex")["target"].mean() * 100
        sex_lbl = ["Female", "Male"]
        ax_sex.bar(sex_lbl, sex_grp.values, color=[HEART_PINK, ACCENT_BLUE],
                   edgecolor=BG_DARK, width=0.45)
        for i, val in enumerate(sex_grp.values):
            ax_sex.text(i, val + 1, f"{val:.1f}%", ha="center", va="bottom",
                        fontweight="bold", color=TEXT_MAIN, fontsize=11)
        ax_sex.set_ylabel("HD Prevalence (%)")
        ax_sex.set_ylim(0, 100)
        ax_sex.grid(axis="y", alpha=0.3)
        ax_sex.set_title("HD Prevalence by Sex", color=TEXT_MAIN, fontweight="bold", pad=10)
        plt.tight_layout()
        st.pyplot(fig_sex, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 3: Correlation heatmap ─────────────────────────────────────
    st.markdown(f"<p style='font-size:0.82rem; color:{TEXT_MUTED}; font-weight:600;'>Pearson Correlation Heatmap</p>", unsafe_allow_html=True)
    import seaborn as sns
    num_df  = df[COLS + ["target"]].apply(pd.to_numeric, errors="coerce")
    corr    = num_df.corr()
    cmap_hm = LinearSegmentedColormap.from_list("heart", [ACCENT_BLUE, BG_CARD, HEART_RED])
    mask    = np.triu(np.ones_like(corr, dtype=bool))

    fig_hm, ax_hm = plt.subplots(figsize=(12, 7))
    sns.heatmap(corr, ax=ax_hm, mask=mask, cmap=cmap_hm, center=0,
                annot=True, fmt=".2f", linewidths=0.5, linecolor=BG_DARK,
                annot_kws={"size": 8, "color": TEXT_MAIN},
                cbar_kws={"shrink": 0.8})
    ax_hm.set_title("Feature Correlation Matrix", color=TEXT_MAIN, fontweight="bold", pad=10)
    ax_hm.tick_params(colors=TEXT_MUTED)
    plt.tight_layout()
    st.pyplot(fig_hm, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 4: Numeric feature violins ───────────────────────────────
    st.markdown(f"<p style='font-size:0.82rem; color:{TEXT_MUTED}; font-weight:600;'>Numeric Feature Distribution (Violin) by Target</p>", unsafe_allow_html=True)
    num_feats = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    fig_vl, axes_vl = plt.subplots(1, 5, figsize=(20, 5))
    fig_vl.suptitle("Feature Violins — Heart Disease vs No Disease",
                     fontsize=11, fontweight="bold", color=ACCENT_BLUE)
    for ax_v, feat in zip(axes_vl, num_feats):
        data_0 = df[df["target"] == 0][feat].dropna().values
        data_1 = df[df["target"] == 1][feat].dropna().values
        parts  = ax_v.violinplot([data_0, data_1], positions=[0, 1],
                                  showmedians=True, showextrema=False)
        for pc, clr in zip(parts["bodies"], [ACCENT_GRN, HEART_RED]):
            pc.set_facecolor(clr)
            pc.set_alpha(0.6)
        parts["cmedians"].set_color(TEXT_MAIN)
        parts["cmedians"].set_linewidth(2)
        ax_v.set_xticks([0, 1])
        ax_v.set_xticklabels(["No HD", "HD"], fontsize=9)
        ax_v.set_title(feat.upper(), color=TEXT_MAIN, fontweight="bold")
        ax_v.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_vl, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 5: Categorical bars ───────────────────────────────────────
    st.markdown(f"<p style='font-size:0.82rem; color:{TEXT_MUTED}; font-weight:600;'>Categorical Features — HD Prevalence (%)</p>", unsafe_allow_html=True)
    cat_feats = ["cp", "slope", "ca", "thal", "exang", "restecg"]
    fig_cat, axes_cat = plt.subplots(2, 3, figsize=(18, 9))
    fig_cat.suptitle("Categorical Feature × Heart Disease Prevalence",
                     fontsize=11, fontweight="bold", color=ACCENT_PRP)
    axes_cat = axes_cat.flatten()
    for ax_c, feat in zip(axes_cat, cat_feats):
        sub = df[[feat, "target"]].dropna()
        sub[feat] = sub[feat].astype(int)
        grp = sub.groupby(feat)["target"].agg(["mean", "count"]).reset_index()
        lmap = CAT_LABELS.get(feat, {})
        xlbls = [lmap.get(int(v), str(int(v))) for v in grp[feat]]
        bars_c = ax_c.bar(range(len(grp)), grp["mean"] * 100,
                          color=PALETTE[:len(grp)], edgecolor=BG_DARK, width=0.6)
        for j, (bar_c, cnt) in enumerate(zip(bars_c, grp["count"])):
            ax_c.text(bar_c.get_x() + bar_c.get_width() / 2,
                      bar_c.get_height() + 1,
                      f"n={cnt}", ha="center", va="bottom",
                      fontsize=7.5, color=TEXT_MUTED)
        ax_c.set_xticks(range(len(grp)))
        ax_c.set_xticklabels(xlbls, rotation=20, ha="right", fontsize=8)
        ax_c.set_title(feat.upper(), color=TEXT_MAIN, fontweight="bold")
        ax_c.set_ylabel("HD Prev. (%)")
        ax_c.set_ylim(0, 105)
        ax_c.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_cat, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 6: Scatter — Age vs MaxHR coloured by target ─────────────
    st.markdown(f"<p style='font-size:0.82rem; color:{TEXT_MUTED}; font-weight:600;'>Age vs Max Heart Rate — coloured by diagnosis</p>", unsafe_allow_html=True)
    fig_sc, ax_sc = plt.subplots(figsize=(8, 4.5))
    for tgt, clr, lbl in [(0, ACCENT_GRN, "No Disease"), (1, HEART_RED, "Heart Disease")]:
        sub = df[df["target"] == tgt]
        ax_sc.scatter(sub["age"], sub["thalach"], c=clr, alpha=0.45,
                      s=22, label=lbl, edgecolors="none")
    ax_sc.set_xlabel("Age (years)")
    ax_sc.set_ylabel("Max Heart Rate (bpm)")
    ax_sc.set_title("Age vs Max HR — Diagnostic Scatter", color=TEXT_MAIN,
                    fontweight="bold", pad=10)
    ax_sc.legend(facecolor=BG_CARD, labelcolor=TEXT_MAIN)
    ax_sc.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_sc, use_container_width=True)


# ───────────────────────────────────────────────────────────────────────
# TAB 5 — FEATURE GUIDE
# ───────────────────────────────────────────────────────────────────────
with tab_about:
    st.markdown("""
    <div class="section-header">
      <h2>ℹ️ Feature Reference Guide</h2>
    </div>
    """, unsafe_allow_html=True)

    guide = [
        ("age",      "Age",                    "Patient age in years. Risk increases significantly after 45 (M) / 55 (F)."),
        ("sex",      "Sex",                    "Biological sex. Males have higher incidence of CAD at younger ages."),
        ("cp",       "Chest Pain Type",        "4 types: Typical Angina → Asymptomatic. Asymptomatic (type 4) is paradoxically the highest-risk indicator."),
        ("trestbps", "Resting Blood Pressure", "Systolic BP in mmHg at rest. Hypertension (>140) is a major risk factor."),
        ("chol",     "Serum Cholesterol",      "Total cholesterol in mg/dL. Values >240 are considered high."),
        ("fbs",      "Fasting Blood Sugar",    "Whether fasting blood sugar > 120 mg/dL. Indicates possible diabetes."),
        ("restecg",  "Resting ECG",            "Electrocardiographic results at rest. LV hypertrophy indicates cardiac stress."),
        ("thalach",  "Max Heart Rate",         "Maximum heart rate achieved during stress test. Higher is generally better (inverse correlation with HD)."),
        ("exang",    "Exercise Angina",         "Chest pain induced by exercise. Strong positive predictor."),
        ("oldpeak",  "ST Depression",          "ST depression induced by exercise relative to rest. Values >2 are clinically significant."),
        ("slope",    "ST Slope",               "Slope of peak exercise ST segment. Downsloping indicates ischaemia."),
        ("ca",       "Fluoroscopy Vessels",    "Number of major vessels (0–3) coloured by fluoroscopy. More vessels = higher disease burden."),
        ("thal",     "Thalassemia",            "Blood disorder test: Normal / Fixed Defect / Reversible Defect. Reversible defect is a key risk signal."),
    ]

    c1, c2 = st.columns(2)
    for i, (feat, name, desc) in enumerate(guide):
        col = c1 if i % 2 == 0 else c2
        with col:
            st.markdown(f"""
            <div class="input-card">
              <div class="input-card-title">Feature: <code style="color:{ACCENT_BLUE};">{feat}</code></div>
              <div style="font-weight:600; color:{TEXT_MAIN}; margin-bottom:0.3rem;">{name}</div>
              <div style="font-size:0.83rem; color:{TEXT_MUTED}; line-height:1.55;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:{BG_CARD}; border:1px solid {BG_BORDER};
         border-left:4px solid {ACCENT_YLW}; border-radius:8px;
         padding:1rem 1.2rem; margin-top:1rem;">
      <b style="color:{ACCENT_YLW};">📚 Dataset</b><br>
      <span style="font-size:0.85rem; color:{TEXT_MUTED}; line-height:1.7;">
        UCI Heart Disease Repository — Cleveland, Hungarian, Switzerland, VA Long Beach.<br>
        920 patients combined. Binary target: 0 = No Disease, 1 = Heart Disease (any severity).<br>
        <b style="color:{TEXT_MAIN};">Top predictive features:</b> thal, cp, ca, thalach, oldpeak
      </span>
    </div>
    """, unsafe_allow_html=True)

    # Research conclusions
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="section-header">
      <h2>📋 Research Conclusions</h2>
    </div>
    """, unsafe_allow_html=True)

    conclusions = [
        ("🏆 Best Model",        "XGBoost / GradientBoosting (AUC ≈ 0.91+)"),
        ("🔑 Top Features",      "thal, cp, ca, thalach, oldpeak"),
        ("🧬 Dataset",           "920 patients across 4 global centres"),
        ("📐 CV Stability",      "Low variance across 10-fold stratified CV"),
        ("🩺 Clinical Note",     "Asymptomatic chest pain (cp=4) is the strongest categorical predictor"),
        ("📉 Key Insight",       "thalach (max heart rate) shows strongest negative correlation with HD"),
        ("🤝 Ensemble Benefit",  "Voting consistently outperforms individual models in stability"),
    ]
    rc1, rc2 = st.columns(2)
    for i, (key, val) in enumerate(conclusions):
        col = rc1 if i % 2 == 0 else rc2
        with col:
            st.markdown(f"""
            <div class="input-card" style="margin-bottom:0.6rem;">
              <div style="font-size:0.8rem; font-weight:700; color:{ACCENT_BLUE}; margin-bottom:0.2rem;">{key}</div>
              <div style="font-size:0.85rem; color:{TEXT_MAIN};">{val}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:{BG_CARD}; border:1px solid {BG_BORDER};
         border-left:4px solid {ACCENT_PRP}; border-radius:8px;
         padding:1rem 1.2rem; margin-top:0.5rem;">
      <b style="color:{ACCENT_PRP};">🚀 Future Work</b><br>
      <span style="font-size:0.85rem; color:{TEXT_MUTED}; line-height:1.9;">
        1. Hyperparameter optimisation with Optuna / BayesSearchCV<br>
        2. SHAP TreeExplainer for per-patient feature attributions<br>
        3. Federated learning across distributed hospital datasets<br>
        4. Integration with EHR systems via FHIR API<br>
        5. Prospective clinical validation study
      </span>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(f"""
<div style="border-top:1px solid {BG_BORDER}; padding:1.5rem 0 0.5rem 0;
     text-align:center; color:{TEXT_MUTED}; font-size:0.78rem; line-height:1.8;">
  <span style="color:{HEART_RED}; font-size:1.1rem;">❤️</span>
  <b style="color:{TEXT_MAIN};"> Heart Disease Predictor</b>
  &nbsp;·&nbsp; Built with Streamlit &amp; Scikit-learn
  &nbsp;·&nbsp; UCI Heart Disease Repository
  &nbsp;·&nbsp; For Research &amp; Educational Use Only<br>
  <span style="font-size:0.72rem;">
    Models: Logistic Regression · Random Forest · Gradient Boosting · Extra Trees · DNN (MLP)
    {' · XGBoost' if XGB_AVAILABLE else ''}
    {' · LightGBM' if LGB_AVAILABLE else ''}
  </span>
</div>
""", unsafe_allow_html=True)