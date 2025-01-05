import streamlit as st
import pandas as pd
import joblib

# Load the saved model and scaler
model = joblib.load('rf_model.joblib')
scaler = joblib.load('scaler.joblib')

# Set page title
st.title('Bank Customer Churn Prediction')

# Add description
st.write('Enter customer information to predict churn probability')

# Create input fields
col1, col2 = st.columns(2)

with col1:
    credit_score = st.slider('Credit Score', 300, 850, 600)
    age = st.slider('Age', 18, 100, 35)
    tenure = st.slider('Tenure (years)', 0, 10, 5)
    balance = st.number_input('Balance', 0.0, 250000.0, 50000.0)
    num_of_products = st.slider('Number of Products', 1, 4, 1)

with col2:
    has_credit_card = st.checkbox('Has Credit Card')
    is_active_member = st.checkbox('Is Active Member')
    estimated_salary = st.number_input('Estimated Salary', 0.0, 250000.0, 50000.0)
    geography = st.selectbox('Geography', ['France', 'Germany', 'Spain'])
    gender = st.selectbox('Gender', ['Male', 'Female'])

if st.button('Predict Churn Probability'):
    # Prepare input data
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [int(has_credit_card)],
        'IsActiveMember': [int(is_active_member)],
        'EstimatedSalary': [estimated_salary],
        'Geography_Germany': [1 if geography == 'Germany' else 0],
        'Geography_Spain': [1 if geography == 'Spain' else 0],
        'Gender_Male': [1 if gender == 'Male' else 0]
    })

    # Scale the input data
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict_proba(input_scaled)[0][1]

    # Display results
    st.write('---')
    st.header('Prediction Results')
    st.write(f'Churn Probability: {prediction:.2%}')
    
    if prediction > 0.5:
        st.error('⚠️ High Risk: This customer is likely to churn!')
    else:
        st.success('✅ Low Risk: This customer is likely to stay!')
