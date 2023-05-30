import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

def load_pickles(model_pickle_path, label_encoder_pickle_path):
    with open(model_pickle_path, "rb") as model_pickle_opener:
        model = pickle.load(model_pickle_opener)
    
    with open(label_encoder_pickle_path, "rb") as label_encoder_opener:
        label_encoder_dict = pickle.load(label_encoder_opener)
        
    return model, label_encoder_dict


def pre_process_data(df, label_encoder_dict):
    df_out = df.copy()
    df_out.replace(" ", 0, inplace=True)
    df_out.loc[:, 'TotalCharges'] = pd.to_numeric(df_out.loc[:, 'TotalCharges'])
    if 'customerID' in df_out.columns:
        df_out.drop('customerID', axis=1, inplace=True)
    for column, le in label_encoder_dict.items():
        df_out.loc[:, column] = le.transform(df_out.loc[:, column])
        
    return df_out


def make_predictions(test_data):
    model_pickle_path = "./models/model.pkl"
    label_encoder_pickle_path = "./models/label_encoders.pkl"
    
    model, label_encoder_dict = load_pickles(model_pickle_path, label_encoder_pickle_path)
    
    data_processed = pre_process_data(test_data, label_encoder_dict)
    if 'Churn' in data_processed.columns:
        data_processed = data_processed.drop(columns=['Churn'])
    prediction = model.predict(data_processed)
    return prediction

if __name__ == "__main__":
    st.title("Customer churn prediction")
    data = pd.read_csv("./data/single_row_to_check.csv")

    gender = st.selectbox(
        "Select customer's gender:",
        ["Female", "Male"]
    ) 
    senior_citizen = st.selectbox(
        "Is customer a senior citizen?",
        ["No", "Yes"]
    )
    partner = st.selectbox(
        "Is customer married?",
        ["No", "Yes"]
    )
    dependents = st.selectbox(
        "Does customer have dependents?",
        ["No", "Yes"]
    )
    tenure = st.select_slider(  
        "How many months has customer been with the company?",
        options=list(range(0, 73))
    )
    phone_service = st.selectbox(
        "Does customer have phone service?",
        ["No", "Yes"]
    )
    multiple_lines = st.selectbox(
        "Does customer have multiple lines?",
        ["No", "No phone service", "Yes"]
    )
    internet_service = st.selectbox(
        "What type of internet service does customer have?",
        ["DSL", "Fiber optic", "No"]
    )
    online_security = st.selectbox(
        "Does customer have online security?",
        ["No", "No internet service", "Yes"]
    )
    online_backup = st.selectbox(
        "Does customer have online backup?",
        ["No", "No internet service", "Yes"]
    )
    device_protection = st.selectbox(
        "Does customer have device protection?",
        ["No", "No internet service", "Yes"]
    )
    tech_support = st.selectbox(
        "Does customer have tech support?",
        ["No", "No internet service", "Yes"]
    )
    streaming_tv = st.selectbox(
        "Does customer have streaming TV?",
        ["No", "No internet service", "Yes"]
    )
    streaming_movies = st.selectbox(
        "Does customer have streaming movies?",
        ["No", "No internet service", "Yes"]
    )
    contract = st.selectbox(
        "What type of contract does customer have?",
        ["Month-to-month", "One year", "Two year"]
    )
    paperless_billing = st.selectbox(
        "Does customer have paperless billing?",
        ["No", "Yes"]   
    )
    payment_method = st.selectbox(
        "What type of payment method does customer use?",
        ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"]
    )
    monthly_charges = st.select_slider(
        "What is customer's monthly charge?",
        options=list(range(0, 120))
    )
    total_charges = st.select_slider(
        "What is customer's total charge?",
        options=list(range(0, 9000))
    )
if st.button("Predict Churn"):
        prediction = make_predictions(data)[0]
        prediction = "will churn" if prediction == 1 else "will not churn"
        st.text(f"Customer prediction: {prediction}")