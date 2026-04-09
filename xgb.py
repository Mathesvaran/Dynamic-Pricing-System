"""
=============================================================================
  Dynamic Pricing Model — XGBoost (Per Category × Sub-Category)
=============================================================================
  Features  : base_price, competitor_price, demand, rating, reviews,
              stock, discount, season, day_of_week
  Target    : price
  Rules     :
    • Each (category, sub_category) pair is trained independently.
    • Predicted price is CAPPED at base_price × 1.5  (50 % premium cap).
    • Metrics reported per segment: RMSE, MAE, R²
=============================================================================
"""

# ---------------------------------------------------------------------------
# 0.  Imports
# ---------------------------------------------------------------------------
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1.  Configuration
# ---------------------------------------------------------------------------
DATA_PATH  = os.path.join(os.path.dirname(__file__), "dynamic_pricing_dataset_v2.csv")

PRICE_CAP_MULTIPLIER = 1.25   # maximum allowed price = base_price × 1.50
TEST_SIZE            = 0.20
RANDOM_STATE         = 42

# XGBoost hyper-parameters (shared across all segments)
XGB_PARAMS = dict(
    n_estimators       = 500,
    learning_rate      = 0.05,
    max_depth          = 6,
    subsample          = 0.8,
    colsample_bytree   = 0.8,
    min_child_weight   = 3,
    reg_alpha          = 0.1,
    reg_lambda         = 1.0,
    random_state       = RANDOM_STATE,
    n_jobs             = -1,
    early_stopping_rounds = 30,
    eval_metric        = "rmse",
)

# Color palette for charts
SEGMENT_PALETTE = sns.color_palette("tab20", 20)

# ---------------------------------------------------------------------------
# 2.  Load & basic checks
# ---------------------------------------------------------------------------
print("=" * 70)
print("  Dynamic Pricing — XGBoost (Per Segment)")
print("=" * 70)

df = pd.read_csv(DATA_PATH)
print(f"\n[DATA]  Loaded {len(df):,} rows × {df.shape[1]} columns")
print(f"        Columns : {list(df.columns)}")
print(f"\n[DATA]  Category distribution:")
print(df.groupby(["category", "sub_category"]).size().to_string())

# ---------------------------------------------------------------------------
# 3.  Feature engineering
# ---------------------------------------------------------------------------
def encode_season(s):
    return {"Normal": 0, "Festive": 1}.get(s, 0)

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
day_map   = {d: i for i, d in enumerate(DAY_ORDER)}

def prepare_features(data: pd.DataFrame) -> pd.DataFrame:
    d = data.copy()
    d["season_enc"]      = d["season"].map(encode_season)
    d["day_enc"]         = d["day_of_week"].map(day_map)
    d["is_weekend"]      = d["day_enc"].isin([5, 6]).astype(int)
    d["price_vs_comp"]   = d["base_price"] / (d["competitor_price"] + 1e-6)
    d["demand_norm"]     = d["demand"]  / (d["demand"].max()  + 1e-6)
    d["stock_norm"]      = d["stock"]   / (d["stock"].max()   + 1e-6)
    d["discount_ratio"]  = d["discount"] / 100.0
    d["reviews_log"]     = np.log1p(d["reviews"])
    return d

FEATURE_COLS = [
    "base_price",
    "competitor_price",
    "demand",
    "rating",
    "reviews_log",
    "stock",
    "discount_ratio",
    "season_enc",
    "day_enc",
    "is_weekend",
    "price_vs_comp",
    "demand_norm",
    "stock_norm",
]

df = prepare_features(df)

# ---------------------------------------------------------------------------
# 4.  Per-segment training
# ---------------------------------------------------------------------------
segments  = df.groupby(["category", "sub_category"])
seg_keys  = sorted(segments.groups.keys())
results   = []     # list of dicts — one per segment
all_models = {}    # {(cat, sub): model}

print("\n" + "=" * 70)
print("  Training XGBoost per (Category × Sub-Category) segment")
print("=" * 70)

for (cat, sub) in seg_keys:
    seg_df = segments.get_group((cat, sub)).copy()
    n      = len(seg_df)
    label  = f"{cat} | {sub}"

    if n < 20:
        print(f"\n  [SKIP]  {label}  —  only {n} rows, skipping.")
        continue

    print(f"\n  [{cat}] [{sub}]  —  {n} samples")

    X = seg_df[FEATURE_COLS].values
    y = seg_df["price"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # ---- raw predictions ----
    y_pred_raw = model.predict(X_test)

    # ---- apply price ceiling (base_price × 1.5) ----
    base_prices_test = X_test[:, FEATURE_COLS.index("base_price")]
    price_caps       = base_prices_test * PRICE_CAP_MULTIPLIER
    y_pred           = np.minimum(y_pred_raw, price_caps)

    # ---- apply psychological pricing (round to nearest 50 minus 1) ----
    y_pred           = np.round(y_pred / 50.0) * 50 - 1

    # ---- metrics ----
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    print(f"         RMSE : {rmse:,.2f}   |   MAE : {mae:,.2f}   |   R² : {r2:.4f}")

    results.append({
        "category"    : cat,
        "sub_category": sub,
        "segment"     : label,
        "n_samples"   : n,
        "rmse"        : rmse,
        "mae"         : mae,
        "r2"          : r2,
        "y_test"      : y_test,
        "y_pred"      : y_pred,
        "model"       : model,
    })
    all_models[(cat, sub)] = model

results_df = pd.DataFrame([{k: v for k, v in r.items()
                              if k not in ("y_test", "y_pred", "model")}
                             for r in results])

# ---------------------------------------------------------------------------
# 5.  Global summary table
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  METRICS SUMMARY")
print("=" * 70)
print(results_df[["segment", "n_samples", "rmse", "mae", "r2"]].to_string(index=False))

# ---------------------------------------------------------------------------
# 6.  Visualisations
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("tab10")

# ---- 6a.  Metrics bar chart (RMSE / MAE / R²) ----------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("XGBoost Dynamic Pricing — Metrics per Segment", fontsize=15, fontweight="bold")

segs   = results_df["segment"].tolist()
colors = SEGMENT_PALETTE[:len(segs)]

for ax, metric, title in zip(axes,
                              ["rmse", "mae", "r2"],
                              ["RMSE (lower is better)",
                               "MAE  (lower is better)",
                               "R²   (higher is better)"]):
    bars = ax.barh(segs, results_df[metric], color=colors, edgecolor="white", height=0.6)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(metric.upper())
    for bar, val in zip(bars, results_df[metric]):
        ax.text(bar.get_width() + bar.get_width() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:,.2f}", va="center", fontsize=8)

plt.tight_layout()
plt.show()
print(f"\n  [SAVED]  01_metrics_per_segment.png")

# ---------------------------------------------------------------------------
# 7.  Save metrics CSV
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 8.  Demo — predict optimal price for a new product
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  DEMO — Predict optimal price for sample inputs")
print("=" * 70)

sample_inputs = [
    {
        "category": "Camera", "sub_category": "Premium",
        "base_price": 175511, "competitor_price": 180000,
        "demand": 450, "rating": 4.5, "reviews": 800,
        "stock": 200, "discount": 5.0,
        "season": "Festive", "day_of_week": "Saturday",
    },
    {
        "category": "Mobile Phone", "sub_category": "Budget",
        "base_price": 11858, "competitor_price": 12000,
        "demand": 400, "rating": 3.5, "reviews": 2500,
        "stock": 400, "discount": 8.0,
        "season": "Normal", "day_of_week": "Wednesday",
    },
    {
        "category": "Laptop", "sub_category": "Mid",
        "base_price": 78680, "competitor_price": 80000,
        "demand": 350, "rating": 4.1, "reviews": 1000,
        "stock": 300, "discount": 6.0,
        "season": "Festive", "day_of_week": "Monday",
    },
    {
        "category": "Earphones", "sub_category": "Mid",
        "base_price": 9817, "competitor_price": 10200,
        "demand": 300, "rating": 3.9, "reviews": 1200,
        "stock": 400, "discount": 4.0,
        "season": "Normal", "day_of_week": "Friday",
    },
]

for inp in sample_inputs:
    cat, sub = inp["category"], inp["sub_category"]

    if (cat, sub) not in all_models:
        print(f"  [NO MODEL]  {cat} | {sub}")
        continue

    row = pd.DataFrame([inp])
    row = prepare_features(row)
    X_new = row[FEATURE_COLS].values

    model_seg = all_models[(cat, sub)]
    raw_pred  = model_seg.predict(X_new)[0]

    cap       = inp["base_price"] * PRICE_CAP_MULTIPLIER
    final     = min(raw_pred, cap)

    # ---- apply psychological pricing (round to nearest 50 minus 1) ----
    final     = round(final / 50.0) * 50 - 1

    print(f"\n  Segment       : {cat} | {sub}")
    print(f"  Base Price    : ₹{inp['base_price']:,.0f}")
    print(f"  Comp. Price   : ₹{inp['competitor_price']:,.0f}")
    print(f"  Season        : {inp['season']}   |   Day : {inp['day_of_week']}")
    print(f"  Demand        : {inp['demand']}   |   Stock : {inp['stock']}")
    print(f"  Raw Prediction : ₹{raw_pred:,.2f}")
    print(f"  Price Cap     : ₹{cap:,.2f}  (base × {PRICE_CAP_MULTIPLIER})")
    print(f"  ✅ Optimal Price : ₹{final:,.2f}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("=" * 70)
