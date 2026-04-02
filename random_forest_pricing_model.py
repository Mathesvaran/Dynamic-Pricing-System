import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os


def load_and_preprocess(train_path, test_path):
    print(f"Loading datasets from {train_path} and {test_path}...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Drop product_id as it's just an identifier
    train_df.drop(columns=['product_id'], inplace=True, errors='ignore')
    test_df.drop(columns=['product_id'], inplace=True, errors='ignore')

    # One-hot encode categorical features (e.g., 'category')
    train_df = pd.get_dummies(train_df, columns=['category'], drop_first=True)
    test_df = pd.get_dummies(test_df, columns=['category'], drop_first=True)

    # Align train and test DataFrames to ensure identical columns
    train_df, test_df = train_df.align(test_df, join='inner', axis=1)

    # Separate features and target variable
    target_col = 'optimal_price'
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    print(f"  Training samples: {len(X_train)} | Test samples: {len(X_test)}")
    print(f"  Features: {list(X_train.columns)}")

    return X_train, y_train, X_test, y_test


def train_and_evaluate(X_train, y_train, X_test, y_test):
    print("\nTraining Random Forest Regressor...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,          # Grow full trees (let the forest decide)
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1                # Use all CPU cores for speed
    )

    model.fit(X_train, y_train)
    print("Model training complete.")

    print("\nEvaluating model on test dataset...")
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n--- Model Evaluation Metrics ---")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Error (MAE):      {mae:.2f}")
    print(f"R-squared (R2):                 {r2:.4f}")

    return model, y_pred


def generate_visualizations(model, X_train, y_test, y_pred, output_dir):
    print(f"\nGenerating and saving visualizations to: {os.path.abspath(output_dir)}")
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Feature Importance Bar Chart ──────────────────────────────────────
    importances = model.feature_importances_
    features = X_train.columns
    indices = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(indices)))
    ax.barh(range(len(indices)), importances[indices], color=colors, align='center')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([features[i] for i in indices], fontsize=10)
    ax.set_xlabel('Relative Importance', fontsize=12)
    ax.set_title('Feature Importance — Random Forest\n(Optimal Price Prediction)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path1 = os.path.join(output_dir, 'rf_feature_importance.png')
    plt.savefig(path1, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path1}")

    # ── 2. Actual vs. Predicted Scatter Plot ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_test, y_pred, alpha=0.35, color='steelblue', edgecolors='none', label='Predictions')
    max_val = max(float(y_test.max()), float(y_pred.max()))
    min_val = min(float(y_test.min()), float(y_pred.min()))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, label='Perfect Prediction')
    ax.set_xlabel('Actual Optimal Price', fontsize=12)
    ax.set_ylabel('Predicted Optimal Price', fontsize=12)
    ax.set_title('Actual vs Predicted Optimal Price — Random Forest', fontsize=13, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    path2 = os.path.join(output_dir, 'rf_actual_vs_predicted.png')
    plt.savefig(path2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path2}")

    # ── 3. Residuals Distribution Plot ───────────────────────────────────────
    residuals = np.array(y_test) - y_pred
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(residuals, kde=True, bins=40, color='steelblue', ax=ax)
    ax.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Zero Residual')
    ax.set_xlabel('Residual (Actual − Predicted)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Residuals Distribution — Random Forest', fontsize=13, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    path3 = os.path.join(output_dir, 'rf_residuals_distribution.png')
    plt.savefig(path3, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path3}")


def predict_random_samples(model, feature_cols):
    print("\n--- Sample Predictions Using Random Inputs (from formula ranges) ---")

    samples = []
    num_samples = 3

    for _ in range(num_samples):
        cost_price       = np.random.uniform(200, 2000)
        current_price    = cost_price * (1 + np.random.uniform(0.1, 0.5))
        competitor_price = current_price * (1 + np.random.uniform(-0.05, 0.05))
        views            = np.random.uniform(400, 1500) * np.random.uniform(0.8, 1.2)
        conversion_rate  = min(max(0.1 * (competitor_price / current_price), 0.01), 0.3)
        price_factor     = current_price / competitor_price
        units_sold       = views * conversion_rate * (1 / price_factor) * np.random.uniform(0.8, 1.2)
        stock_available  = units_sold * np.random.uniform(1.5, 3)
        month            = np.random.randint(1, 13)

        raw = {
            'cost_price':       cost_price,
            'current_price':    current_price,
            'competitor_price': competitor_price,
            'views':            views,
            'units_sold':       units_sold,
            'conversion_rate':  conversion_rate,
            'stock_available':  stock_available,
            'month':            month,
        }

        # Build model input row (fill missing one-hot columns with 0)
        row = {col: raw.get(col, 0) for col in feature_cols}
        samples.append((raw, row))

    inputs_df = pd.DataFrame([r for _, r in samples])
    preds = model.predict(inputs_df)

    for i, ((raw, _), pred) in enumerate(zip(samples, preds), start=1):
        print(f"\nSample {i}:")
        for k, v in raw.items():
            print(f"  {k:<22}: {round(v, 2)}")
        print(f"  >>> Predicted Optimal Price : {pred:.2f}")


if __name__ == "__main__":
    train_file = "dynamic_pricing_train.csv"
    test_file  = "dynamic_pricing_test.csv"

    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print("Error: Could not find training or testing CSV files in the current directory.")
    else:
        # 1. Load & preprocess
        X_train, y_train, X_test, y_test = load_and_preprocess(train_file, test_file)

        # 2. Train & evaluate
        model, y_pred = train_and_evaluate(X_train, y_train, X_test, y_test)

        # 3. Visualizations (saved to current folder)
        generate_visualizations(model, X_train, y_test, y_pred, output_dir=".")

        # 4. Predictions on synthetic random inputs
        predict_random_samples(model, X_train.columns)

        print("\nProcess completed successfully!")
