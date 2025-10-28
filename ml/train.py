import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

def train_model(data_path='ml/labeled_btc_data.csv'):
    """
    Trains a LightGBM model on the labeled data and saves it to a file.
    """
    df = pd.read_csv(data_path, index_col='time')

    X = df.drop(columns=['label', 'symbol'])
    y = df['label']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Initialize and train the LightGBM model
    model = lgb.LGBMClassifier(objective='multiclass', num_class=3, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Model Evaluation Report:")
    print(classification_report(y_test, y_pred))

    # Save the trained model
    joblib.dump(model, 'ml/lgbm_model.pkl')
    print("Model trained and saved to 'ml/lgbm_model.pkl'.")

if __name__ == '__main__':
    try:
        train_model()
    except FileNotFoundError:
        print("Labeled data not found. Please run 'ml/data_labeling.py' first.")
    except Exception as e:
        print(f"An error occurred during model training: {e}")
