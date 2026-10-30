"""
OrderBookMachineLearning.py

Use Machine Learning to predict short-term stock price movement using 
Order Book Imbalances (L1, L2, L3) and Money Flow (MF) metrics.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


def load_and_preprocess_data(df: pd.DataFrame):
    """
    Normalizes features to range [-1, 1] and splits into train/test sets.
    """
    X = df.drop(columns=['target'])
    y = df['target']

    # Scale features to [-1, 1]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)


def main():
    print("--- Order Book Machine Learning Pipeline ---")
    
    # 1. Create dummy/sample data for testing 
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'L1_OBI': np.random.uniform(-1, 1, 1000),
        'L2_OBI': np.random.uniform(-1, 1, 1000),
        'L3_OBI': np.random.uniform(-1, 1, 1000),
        'Money_Flow': np.random.uniform(-1, 1, 1000),
        'target': np.random.choice(['u', 'd'], 1000)  # Fixed: changed np.choice to np.random.choice
    })

    # 2. Preprocess
    X_train, X_test, y_train, y_test = load_and_preprocess_data(sample_data)

    # 3. Train Model (Random Forest Classifier)
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 4. Evaluate
    y_pred = model.predict(X_test)
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()