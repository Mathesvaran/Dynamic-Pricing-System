import pandas as pd
import numpy as np
import xgboost as xgb
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
    # Use pandas get_dummies and align columns to ensure train and test have the same features
    train_df = pd.get_dummies(train_df, columns=['category'], drop_first=True)
    test_df = pd.get_dummies(test_df, columns=['category'], drop_first=True)
    
    # Align the train and test DataFrames to have the same columns
    train_df, test_df = train_df.align(test_df, join='inner', axis=1)
    
    # Separate features and target
    target_col = 'optimal_price'
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    
    return X_train, y_train, X_test, y_test

def train_and_evaluate(X_train, y_train, X_test, y_test):
    print("\nTraining XGBoost Regressor...")
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    print("Evaluating model on test dataset...")
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
    print(f"\nGenerating and saving visualizations to {output_dir}...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Feature Importance Plot
    plt.figure(figsize=(10, 6))
    importance = model.feature_importances_
    features = X_train.columns
    
    # Sort feature importances
    indices = np.argsort(importance)
    
    plt.barh(range(len(indices)), importance[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.title('Feature Importance for Optimal Price Prediction')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300)
    plt.close()
    
    # 2. Actual vs Predicted Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.3, color='blue')
    
    # Add diagonal line
    max_val = max(y_test.max(), y_pred.max())
    min_val = min(y_test.min(), y_pred.min())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('Actual Optimal Price')
    plt.ylabel('Predicted Optimal Price')
    plt.title('Actual vs Predicted Optimal Price')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'actual_vs_predicted.png'), dpi=300)
    plt.close()
    
    print("Visualizations saved: 'feature_importance.png' and 'actual_vs_predicted.png'")

def predict_random_samples(model, feature_cols):
    print("\nGenerating sample predictions using random inputs...")
    
    samples = []
    num_samples = 3
    
    # We need to construct random features matching the model's expected columns
    for i in range(num_samples):
        # Generate random inputs based on formulas
        cost_price = np.random.uniform(200, 2000)
        current_price = cost_price * (1 + np.random.uniform(0.1, 0.5))
        competitor_price = current_price * (1 + np.random.uniform(-0.05, 0.05))
        views = np.random.uniform(400, 1500) * np.random.uniform(0.8, 1.2)
        conversion_rate = min(max(0.1 * (competitor_price / current_price), 0.01), 0.3)
        price_factor = current_price / competitor_price
        units_sold = views * conversion_rate * (1 / price_factor) * np.random.uniform(0.8, 1.2)
        stock_available = units_sold * np.random.uniform(1.5, 3)
        month = np.random.randint(1, 13)
        
        # Default categorical logic: assuming simple representation since category is one-hot encoded
        # We will set all category dummy variables to 0 (base category)
        sample_dict = {
            'cost_price': cost_price,
            'current_price': current_price,
            'competitor_price': competitor_price,
            'views': views,
            'units_sold': units_sold,
            'conversion_rate': conversion_rate,
            'stock_available': stock_available,
            'month': month
        }
        
        # Populate expected columns
        model_input = {}
        for col in feature_cols:
            if col in sample_dict:
                model_input[col] = sample_dict[col]
            else:
                model_input[col] = 0 # Default for one-hot encoded categorical columns missing here
                
        samples.append(model_input)
    
    sample_df = pd.DataFrame(samples)
    predictions = model.predict(sample_df)
    
    for i in range(num_samples):
        print(f"\nSample {i+1} inputs:")
        for k, v in samples[i].items():
            if samples[i][k] != 0: # Print only non-zero to keep it clean, assuming base category
                print(f"  {k}: {round(v, 2)}")
        print(f"  >>> Predicted Optimal Price: {predictions[i]:.2f}")


if __name__ == "__main__":
    train_file = "dynamic_pricing_train.csv"
    test_file = "dynamic_pricing_test.csv"
    
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print("Error: Could not find training or testing CSV files in the current directory.")
    else:
        # Load and preprocess
        X_train, y_train, X_test, y_test = load_and_preprocess(train_file, test_file)
        
        # Train and evaluate
        model, y_pred = train_and_evaluate(X_train, y_train, X_test, y_test)
        
        # Visualizations (save to current folder)
        generate_visualizations(model, X_train, y_test, y_pred, output_dir=".")
        
        # Predict on random samples
        predict_random_samples(model, X_train.columns)
        
        print("\nProcess completed successfully!")
