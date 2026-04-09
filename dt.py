# ---------------------------------------------------------------------------
# 0.  Imports
# ---------------------------------------------------------------------------
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1.  Configuration
# ---------------------------------------------------------------------------
DATA_PATH  = os.path.join(os.path.dirname(__file__), "dynamic_pricing_dataset_v2.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "dt_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PRICE_CAP_MULTIPLIER = 1.25
TEST_SIZE            = 0.20
RANDOM_STATE         = 42

DT_PARAMS = dict(
    random_state = RANDOM_STATE,
)

SEGMENT_PALETTE = sns.color_palette("tab20", 20)

# ---------------------------------------------------------------------------
# 2.  Load & basic checks
# ---------------------------------------------------------------------------
print("=" * 70)
print("  Dynamic Pricing — Decision Tree (Per Segment)")
print("=" * 70)

df = pd.read_csv(DATA_PATH)
print(f"\n[DATA]  Loaded {len(df):,} rows × {df.shape[1]} columns")

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
results   = []
all_models = {}

print("\n" + "=" * 70)
print("  Training Decision Tree per (Category × Sub-Category) segment")
print("=" * 70)

for (cat, sub) in seg_keys:
    seg_df = segments.get_group((cat, sub)).copy()
    n      = len(seg_df)
    label  = f"{cat} | {sub}"

    if n < 20:
        continue

    print(f"\n  [{cat}] [{sub}]  —  {n} samples")

    X = seg_df[FEATURE_COLS].values
    y = seg_df["price"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Simple dynamic depth like XGB
    depth = 3 if n < 100 else (4 if n < 300 else 6)
    min_cw = 5 if n < 100 else 3

    model = DecisionTreeRegressor(**DT_PARAMS, max_depth=depth, min_samples_leaf=min_cw)
    model.fit(X_train, y_train)

    y_pred_raw = model.predict(X_test)

    # ---- apply price ceiling (base_price × 1.25 / 1.5) ----
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

results_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("y_test", "y_pred", "model")} for r in results])

# ---------------------------------------------------------------------------
# 5.  Visualisations & Saves
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("tab10")

# 6a. Metrics Chart
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Decision Tree Dynamic Pricing — Metrics", fontsize=15, fontweight="bold")
segs   = results_df["segment"].tolist()
colors = SEGMENT_PALETTE[:len(segs)]

for ax, metric, title in zip(axes,
                              ["rmse", "mae", "r2"],
                              ["RMSE", "MAE", "R²"]):
    bars = ax.barh(segs, results_df[metric], color=colors, edgecolor="white", height=0.6)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(metric.upper())

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_metrics_dt.png"), dpi=150, bbox_inches="tight")
plt.close(fig) # Avoid blocking execution
print(f"\n  [SAVED]  01_metrics_dt.png")

metrics_path = os.path.join(OUTPUT_DIR, "dt_metrics_summary.csv")
results_df[["category", "sub_category", "n_samples", "rmse", "mae", "r2"]].to_csv(metrics_path, index=False)
print(f"  [SAVED]  dt_metrics_summary.csv")

# ---------------------------------------------------------------------------
# 6.  Demo
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  DEMO — Predict optimal price for sample inputs (Decision Tree)")
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
]

for inp in sample_inputs:
    cat, sub = inp["category"], inp["sub_category"]
    
    if (cat, sub) not in all_models:
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
    print(f"  Raw Prediction : ₹{raw_pred:,.2f}")
    print(f"  ✅ Optimal Price : ₹{final:,.2f}")

print("\n" + "=" * 70)
