import streamlit as st
import pandas as pd
import joblib

# Load the trained model
# Make sure 'random_forest_model.joblib' is in the same directory as app.py
model = joblib.load('random_forest_model.joblib')

def create_header():
    st.title("Customer Churn Prediction App")
    st.header("Predicting if a Customer will Churn")
    st.subheader("Enter customer details to predict churn.")



tab1, tab2 = st.tabs(["Churn Prediction", "Project Overview"])
with tab1:

    create_header()


    st.write("---")
    st.write("### Enter Customer Information:")

    # Define the features your model was trained on
    # These should match the order and type of features in X_train
    feature_names = [
        'Account length',
        'International plan',
        'Voice mail plan',
        'Number vmail messages',
        'Total day minutes',
        'Total day calls',
        'Total day charge',
        'Total eve minutes',
        'Total eve calls',
        'Total eve charge',
        'Total night minutes',
        'Total night calls',
        'Total night charge',
        'Total intl minutes',
        'Total intl calls',
        'Total intl charge',
        'Customer service calls'
    ]

    # Create input widgets for each feature
    input_data = {}

    # For categorical features ('International plan', 'Voice mail plan'), use selectbox or radio buttons.
    # Remember these were LabelEncoded (No=0, Yes=1)
    input_data['International plan'] = st.selectbox('International Plan', ['No', 'Yes'])
    input_data['Voice mail plan'] = st.selectbox('Voice Mail Plan', ['No', 'Yes'])

    # Convert 'Yes'/'No' to 1/0 for the model
    input_data['International plan'] = 1 if input_data['International plan'] == 'Yes' else 0
    input_data['Voice mail plan'] = 1 if input_data['Voice mail plan'] == 'Yes' else 0

    # For numerical features, use number_input
    input_data['Account length'] = st.number_input('Account Length (days)', min_value=0, max_value=250, value=100)
    input_data['Number vmail messages'] = st.number_input('Number Voice Mail Messages', min_value=0, max_value=50, value=10)
    input_data['Total day minutes'] = st.number_input('Total Day Minutes', min_value=0.0, max_value=360.0, value=180.0)
    input_data['Total day calls'] = st.number_input('Total Day Calls', min_value=0, max_value=170, value=100)
    input_data['Total day charge'] = st.number_input('Total Day Charge', min_value=0.0, max_value=60.0, value=30.0)
    input_data['Total eve minutes'] = st.number_input('Total Evening Minutes', min_value=0.0, max_value=370.0, value=200.0)
    input_data['Total eve calls'] = st.number_input('Total Evening Calls', min_value=0, max_value=170, value=100)
    input_data['Total eve charge'] = st.number_input('Total Evening Charge', min_value=0.0, max_value=35.0, value=17.0)
    input_data['Total night minutes'] = st.number_input('Total Night Minutes', min_value=0.0, max_value=400.0, value=200.0)
    input_data['Total night calls'] = st.number_input('Total Night Calls', min_value=0, max_value=180, value=100)
    input_data['Total night charge'] = st.number_input('Total Night Charge', min_value=0.0, max_value=20.0, value=9.0)
    input_data['Total intl minutes'] = st.number_input('Total International Minutes', min_value=0.0, max_value=25.0, value=10.0)
    input_data['Total intl calls'] = st.number_input('Total International Calls', min_value=0, max_value=20, value=4)
    input_data['Total intl charge'] = st.number_input('Total International Charge', min_value=0.0, max_value=6.0, value=2.70)
    input_data['Customer service calls'] = st.number_input('Customer Service Calls', min_value=0, max_value=10, value=1)


    # Convert input data to a DataFrame, ensuring column order matches training data
    input_df = pd.DataFrame([input_data], columns=feature_names)

    if st.button('Predict Churn'):
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        st.write("---")
        st.write("### Prediction Results:")

        if prediction[0] == 1:
            st.error("This customer is likely to CHURN.")
        else:
            st.success("This customer is likely NOT to churn.")

        st.write(f"Churn Probability: {prediction_proba[0][1]:.2f}")
        st.write(f"No Churn Probability: {prediction_proba[0][0]:.2f}")

with tab2:
    st.header("Project Overview")
    with open("Paper draft.pdf", "rb") as f: 
        pdf_bytes = f.read()

    st.pdf(pdf_bytes)