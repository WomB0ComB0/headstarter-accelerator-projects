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

    with open(filename, "wb") as file:
        pickle.dump(model, file)

    print(f"Model saved as {filename}")
    return model


def prepare_features(input_data: pd.DataFrame) -> pd.DataFrame:
    """Prepare features to match the model's expected format"""

    # Create timestamp from hour
    input_data["trans_timestamp"] = input_data["hour"]
    input_data = input_data.drop("hour", axis=1)

    # Encode categorical variables
    category_mapping = {
        "grocery_pos": 0,
        "shopping_pos": 1,
        "entertainment": 2,
        "food_dining": 3,
        "health_fitness": 4,
    }

    gender_mapping = {"M": 0, "F": 1}

    job_mapping = {
        "tech": 0,
        "service": 1,
        "professional": 2,
        "student": 3,
        "retired": 4,
        "other": 5,
    }

    # Apply mappings
    input_data["category"] = input_data["category"].map(category_mapping)
    input_data["gender"] = input_data["gender"].map(gender_mapping)
    input_data["job"] = input_data["job"].map(job_mapping)

    # Ensure all numeric columns are float
    numeric_columns = ["amt", "age", "distance", "city_pop", "trans_timestamp"]
    input_data[numeric_columns] = input_data[numeric_columns].astype(float)

    # Reorder columns to match training order exactly
    correct_order = [
        "age",
        "amt",
        "category",
        "city_pop",
        "distance",
        "gender",
        "job",
        "trans_timestamp",
    ]

    # Create a new DataFrame with the correct order
    ordered_df = pd.DataFrame(columns=correct_order)
    for col in correct_order:
        ordered_df[col] = input_data[col]

    return ordered_df
