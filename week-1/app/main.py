import streamlit as st
import pandas as pd
import pickle
import numpy as np
import sklearn
import os
from openai import OpenAI
import utils

st.set_page_config(layout="wide")

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=st.secrets["GROQ_API_KEY"]
)


def load_model(file_name):
    try:
        with open(file_name, "rb") as file:
            return pickle.load(file, fix_imports=True, encoding="latin1", errors="strict")
    except FileNotFoundError:
        st.error(f"Model file not found: {file_name}")
        return None


# Display current working directory and list files
st.write(f"Current working directory: {os.getcwd()}")
st.write("Files in current directory:")
st.write(os.listdir())

# Check if 'models' directory exists
if not os.path.exists("models"):
    st.error("'models' directory not found. Creating it now.")
    os.makedirs("models")

# List files in 'models' directory
st.write("Files in 'models' directory:")
st.write(os.listdir("models"))

# Load models
model_files = [
    "xgb_model.pkl", "voting_model.pkl", "nb_model.pkl", "rf_model.pkl",
    "dt_model.pkl", "svc_model.pkl", "knn_model.pkl", "xgb_model_smote.pkl",
    "xgb_model_improved.pkl"
]

models = {}
for model_file in model_files:
    model_name = model_file.split('.')[0]
    models[model_name] = load_model(f"models/{model_file}")

# Check if all models are loaded
if all(model is not None for model in models.values()):
    st.success("All models loaded successfully!")
else:
    st.error("Some models failed to load. Please check the model files and their locations.")
    st.stop()

print(f"Current scikit-learn version: {sklearn.__version__}")


def prepare_input(
    credit_score,
    location,
    gender,
    age,
    tenure,
    balance,
    num_products,
    has_credit_card,
    is_active_member,
    estimated_salary,
):
    input_dict = {
        "CreditScore": credit_score,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": int(has_credit_card),
        "IsActiveMember": int(is_active_member),
        "EstimatedSalary": estimated_salary,
        "Geography_France": 1 if location == "France" else 0,
        "Geography_Germany": 1 if location == "Germany" else 0,
        "Geography_Spain": 1 if location == "Spain" else 0,
        "Gender_Male": 1 if gender == "Male" else 0,
        "Gender_Female": 1 if gender == "Female" else 0,
    }

    input_df = pd.DataFrame([input_dict])
    return input_df, input_dict


def make_predictions(input_df, input_dict):
    probabilities = {
        "XGBoost": models["xgb_model"].predict_proba(input_df)[:, 1][0],
        "RandomForest": models["rf_model"].predict_proba(input_df)[:, 1][0],
        "KNN": models["knn_model"].predict_proba(input_df)[:, 1][0],
    }

    avg_probability = np.mean(list(probabilities.values()))

    col1, col2 = st.columns(2)
    
    with col1:
        fig = utils.create_gauge_chart(probabilities)
        st.plotly_chart(fig, use_container_width=True)
        st.write(
            f"The customer has a {round(avg_probability * 100, 1)}% probability of churning."
        )

    with col2:
        fig_probs = utils.create_probability_chart(probabilities)
        st.plotly_chart(fig_probs, use_container_width=True)

    return avg_probability


def generate_email(probability, input_dict, explanation, surname):
    prompt = f"""You are a manager at HS Bank. You are responsible for 
ensuring customers stay with the bank and are incentivized with 
various offers.

You noticed a customer named {surname} has a {round(probability * 100, 1)}% probability of churning.

Here is the customer's information:
{input_dict}

Here is some explanation as to why the customer might be at risk 
of churning:
{explanation}

Generate an email to the customer based on their information,
asking them to stay if they are at risk of churning, or offering them 
incentives so that they become more loyal to the bank.

Make sure to list out a set of incentives to stay based on their 
information, in bullet point format. Don't ever mention the 
probability of churning, or the machine learning model to the 
customer.
"""

    raw_response = client.chat.completions.create(
        model="llama-3.2-3b-preview",
        messages=[{"role": "user", "content": prompt}],
    )

    print("\n\nEMAIL PROMPT", prompt)

    return raw_response.choices[0].message.content


def explain_prediction(probability, input_dict, surname):
    # Perform the DataFrame operations separately
    churned_stats = (
        df[df["Exited"] == 1].describe().to_string()
    )  # Convert DataFrame to string
    non_churned_stats = df[df["Exited"] == 0].describe().to_string()

    # Create the prompt string
    prompt = f"""
  You are an expert data scientist at a bank, where you specialize in interpreting and explaining predictions of machine learning models.

  Your machine learning model has predicted that a customer named {surname} has a {round(probability * 100, 1)}% probability of churning, based on the information provided below.

  Here is the customer's information:
  {input_dict}

  Here are the machine learning model's top 10 most important features for predicting churn:

  Feature | Importance
  ----------------------------
  NumOfProducts         | 0.323888
  IsActiveMember        | 0.164146
  Age                   | 0.109550
  Geography_Germany     | 0.091373
  Balance               | 0.052786
  Geography_France      | 0.046463
  Gender_Female         | 0.045283
  Geography_Spain       | 0.038655
  CreditScore           | 0.035005
  EstimatedSalary       | 0.032655
  HasCrCard             | 0.031940
  Tenure                | 0.030054
  Gender_Male           | 0.000000

  Here are summary statistics for churned customers:
  {churned_stats}

  Here are summary statistics for non-churned customers:
  {non_churned_stats}

  - If the customer has over a 40% risk of churning, generate a 3 sentence explanation of why they are at risk of churning.
  - If the customer has less than a 40% risk of churning, generate a 3 sentence explanation of why they might not be at risk of churning.
  - Your explanation should be based on the customer's information, the summary statistics of churned and non-churned customers, and the feature importances provided.

  Don't mention the probability of churning, or the machine learning model, or say anything like "Based on the machine learning model's prediction and top 10 most important features", just explain the prediction.
  """

    print("EXPLANATION PROMPT", prompt)

    # Call the OpenAI API to generate the response
    raw_response = client.chat.completions.create(
        model="llama-3.2-3b-preview",
        messages=[{"role": "user", "content": prompt}],
    )

    return raw_response.choices[0].message.content


st.title("Customer Churn Prediction")

# Load the dataset
try:
    df = pd.read_csv("churn.csv")
    st.success("Dataset loaded successfully!")
except FileNotFoundError:
    st.error("Dataset file 'churn.csv' not found. Please make sure it's in the correct location.")
    st.write("Files in current directory:")
    st.write(os.listdir())
    st.stop()

customers = [f"{row['CustomerId']} - {row['Surname']}" for _, row in df.iterrows()]

selected_customer_option = st.selectbox("Select Customer", customers)

if selected_customer_option:

    selected_customer_id = int(selected_customer_option.split(" - ")[0])

    print("Selected Customer ID", selected_customer_id)

    selected_surname = selected_customer_option.split(" - ")[1]

    print("Surname", selected_surname)

    selected_customer = df.loc[df["CustomerId"] == selected_customer_id].iloc[0]

    print("Selected Customer", selected_customer)

    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=850,
            value=int(selected_customer["CreditScore"]),
            step=1,
        )

        location = st.selectbox(
            "Location",
            ["Spain", "France", "Germany"],
            index=["Spain", "France", "Germany"].index(selected_customer["Geography"]),
        )

        gender = st.radio(
            "Gender",
            ["Male", "Female"],
            index=0 if selected_customer["Gender"] == "Male" else 1,
        )

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=int(selected_customer["Age"]),
        )

        tenure = st.number_input(
            "Tenure (years)",
            min_value=0,
            max_value=50,
            value=int(selected_customer["Tenure"]),
        )

    with col2:
        balance = st.number_input(
            "Balance", min_value=0.0, value=float(selected_customer["Balance"])
        )

        num_products = st.number_input(
            "Number of Products",
            min_value=1,
            max_value=10,
            value=int(selected_customer["NumOfProducts"]),
        )

        has_credit_card = st.checkbox(
            "Has Credit Card", value=bool(selected_customer["HasCrCard"])
        )

        is_active_member = st.checkbox(
            "Is Active Member", value=bool(selected_customer["IsActiveMember"])
        )

        estimated_salary = st.number_input(
            "Estimated Salary",
            min_value=0.0,
            value=float(selected_customer["EstimatedSalary"]),
        )

    input_df, input_dict = prepare_input(
        credit_score,
        location,
        gender,
        age,
        tenure,
        balance,
        num_products,
        has_credit_card,
        is_active_member,
        estimated_salary,
    )

    probability = make_predictions(input_df, input_dict)

    explanation = explain_prediction(probability, input_dict, selected_surname)

    st.markdown("---")

    st.subheader("Explanation of Prediction")

    st.markdown(explanation)

    email = generate_email(
        probability, input_dict, explanation, selected_surname
    )

    st.markdown("---")

    st.subheader("Personalized Email")

    st.markdown(email)
