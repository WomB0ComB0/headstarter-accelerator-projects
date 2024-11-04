import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import pickle

def evaluate_model(model, X_train, X_test, y_train, y_test, filename):
    """
    Evaluate a model and save it to disk
    """
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"{model.__class__.__name__} Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, predictions)}")
    print("--------------")
    
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    
    print(f"Model saved as {filename}")
    return model

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features for model training/prediction
    """
    # Add any feature engineering steps here
    return df