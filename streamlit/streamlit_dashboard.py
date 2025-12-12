import streamlit as st
import pandas as pd
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


PROJECT_ROOT_PATH3 = Path("C:/Users/LENOVO/Documents/DAandAI-CodeInstitute/Capstone/capstone-airline-passenger-satisfaction")
CURRENT_DIR = Path(__file__).parent

# --- 1. CONFIGURATION AND SETUP ---
st.set_page_config(
    page_title="Airline Passenger Satisfaction Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Define Absolute File Paths ---
MODEL_PATH = CURRENT_DIR.parent / 'data/cleaned_data/final_satisfaction_pipeline.pkl'
DATA_PATH = CURRENT_DIR.parent / 'data/cleaned_data/train_engineered.csv'

# Load the trained model pipeline
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error("Error: 'final_satisfaction_pipeline.pkl' not found. Please ensure the model file is in the ..\data\cleaned_data")
    st.stop()

# Load the engineered data for visualizations and feature reference
try:
    df_train = pd.read_csv(DATA_PATH)
    df_train['satisfaction'] = df_train['satisfaction'].astype(str)
    # Filter the DataFrame to only include features used by the model
    # (Excluding the target 'satisfaction')
    df_train_features = df_train.drop(columns=['satisfaction'])
except FileNotFoundError:
    st.error("Error: 'train_engineered.csv' not found. Please ensure the data file is in the ..\data\cleaned_data.")
    st.stop()


# --- 2. HEADER AND INTRODUCTION ---
st.title("✈️ Airline Passenger Satisfaction Analysis")
st.markdown("### Predicting Customer Satisfaction using Machine Learning")
st.markdown("---")

# --- 3. SIDEBAR: PREDICTION INTERFACE ---
st.sidebar.header("🎯 Predict Passenger Satisfaction")

with st.sidebar.form("prediction_form"):
    st.subheader("Passenger Profile")
    
    # 3.1 DEMOGRAPHICS
    gender = st.selectbox('Gender', ['Female', 'Male'])
    age = st.slider('Age', 7, 85, 40)
    customer_type = st.selectbox('Customer Type', ['Loyal Customer', 'disloyal Customer'])
    
    # 3.2 FLIGHT DETAILS
    type_of_travel = st.selectbox('Type of Travel', ['Business travel', 'Personal Travel'])
    flight_class = st.selectbox('Class', ['Business', 'Eco', 'Eco Plus'])
    flight_distance = st.slider('Flight Distance (miles)', 50, 5000, 1500)
    
    # 3.3 DELAYS (Use input numbers, they are scaled/encoded by the pipeline)
    dep_delay = st.number_input('Departure Delay (Minutes)', min_value=0, max_value=500, value=0)
    arr_delay = st.number_input('Arrival Delay (Minutes)', min_value=0, max_value=500, value=0)
    
    st.subheader("Service Ratings (0 = N/A or Poor, 5 = Excellent)")
    
    # 3.4 SERVICE RATINGS (0-5 scale)
    seat_comfort = st.slider('Seat Comfort', 0, 5, 4)
    online_boarding = st.slider('Online Boarding', 0, 5, 4)
    inflight_service = st.slider('Inflight Service', 0, 5, 4)
    inflight_entertainment = st.slider('Inflight Entertainment', 0, 5, 4)
    inflight_wifi_service = st.slider('Inflight Wifi Service', 0, 5, 3)
    cleanliness = st.slider('Cleanliness', 0, 5, 4)
    
    submitted = st.form_submit_button("Predict Satisfaction")

# --- 4. PREDICTION LOGIC ---
if submitted:
    # --- Define Unexposed Service Placeholders (Constants) ---
    # These must be set because the model was trained with them.
    dep_arr_time_convenient = 3 
    ease_of_online_booking = 3
    gate_location = 3
    food_and_drink = 3
    on_board_service = 3
    leg_room_service = 3
    baggage_handling = 3
    checkin_service = 3
    # 4.1 Create input DataFrame matching model training features
    
    # Get a copy of a single row from the training data structure
    input_df = df_train_features.iloc[0:1].copy()
    
    # --- Update with user inputs (Variables from sidebar assigned directly to columns) ---
    input_df['Gender'] = gender
    input_df['Customer_Type'] = customer_type
    input_df['Age'] = age
    input_df['Type_of_Travel'] = type_of_travel
    input_df['Class'] = flight_class
    input_df['Flight_Distance'] = flight_distance
    
    # Service Ratings (using slider variables)
    input_df['Inflight_wifi_service'] = inflight_wifi_service
    input_df['Online_Boarding'] = online_boarding # CORRECTED ASSIGNMENT
    input_df['Seat_Comfort'] = seat_comfort
    input_df['Inflight_Entertainment'] = inflight_entertainment
    input_df['Cleanliness'] = cleanliness
    input_df['Inflight_Service'] = inflight_service
    
    # Delay Metrics
    input_df['Departure_Delay_in_Minutes'] = dep_delay
    input_df['Arrival_Delay_in_Minutes'] = arr_delay

    # --- Set unexposed service ratings to their default placeholders (defined in Section 3) ---
    # These must be set because the model was trained with them.
    input_df['Departure/Arrival time convenient'] = dep_arr_time_convenient
    input_df['Ease of Online booking'] = ease_of_online_booking
    input_df['Gate location'] = gate_location
    input_df['Food and drink'] = food_and_drink
    input_df['On-board service'] = on_board_service
    input_df['Leg room service'] = leg_room_service
    input_df['Baggage handling'] = baggage_handling
    input_df['Checkin service'] = checkin_service

    # --- 4.2 Re-engineer features (MUST MATCH FEATURE ENGINEERING NOTEBOOK) ---
    
    # Recalculate Total_Service_Score and Average_Service_Score
    service_cols = ['Inflight_wifi_service', 'Online_Boarding', 'Seat_Comfort', 'Cleanliness',  
                'Inflight_Entertainment', 'Checkin_Service', 'Food_and_Drink', 'Inflight_Service',
                'Ease_of_Online_booking', 'Gate_Location', 'Leg_Room_Service', 'Baggage_Handling']
    
    input_df['Total_Service_Score'] = input_df[service_cols].sum(axis=1)
    input_df['Average_Service_Score'] = input_df[service_cols].mean(axis=1)
    input_df['High_Service_Flag'] = np.where(input_df['Average_Service_Score'] >= 4, 1, 0)
    
    # Recalculate Delay Categories
    bins = [-1, 0, 15, 60, np.inf]
    labels = ['On-time', 'Minor', 'Moderate', 'Severe']
    # Ensure all inputs are numerical before cutting
    input_df['Departure_Delay_in_Minutes'] = pd.to_numeric(input_df['Departure_Delay_in_Minutes'])
    input_df['Arrival_Delay_in_Minutes'] = pd.to_numeric(input_df['Arrival_Delay_in_Minutes'])
    
    input_df['Departure_Delay_Category'] = pd.cut(input_df['Departure_Delay_in_Minutes'], bins=bins, labels=labels).astype(object)
    input_df['Arrival_Delay_Category'] = pd.cut(input_df['Arrival_Delay_in_Minutes'], bins=bins, labels=labels).astype(object)
    
    
    # 4.3 Run Prediction
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[:, 1]

    st.header("Prediction Result:")
    if prediction[0] == 1:
        st.success(f"**Satisfied!** (Confidence: {probability[0]*100:.2f}%)")
        st.balloons()
    else:
        st.warning(f"**Neutral or Dissatisfied** (Confidence: {(1-probability[0])*100:.2f}%)")

# --- 5. MAIN CONTENT: EDA & MODEL INSIGHTS ---

st.header("Key Analytical Insights")

# 5.1 Hypothesis Conclusion 1: Travel Type
st.subheader("H1 & H4: Travel Type and Class Impact")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Insight:** Business travelers are overwhelmingly satisfied.")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x='Type_of_Travel', hue='satisfaction', data=df_train, palette='viridis', ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_title("Satisfaction by Type of Travel")
    st.pyplot(fig)

with col2:
    st.markdown("**Insight:** Business Class drives satisfaction, while Economy shows high dissatisfaction.")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x='Class', hue='satisfaction', data=df_train, palette='magma', ax=ax)
    ax.set_title("Satisfaction by Class")
    st.pyplot(fig)
    
# 5.2 Hypothesis Conclusion 2: Delays vs. Service Score
st.subheader("H3 & H5: Service Quality vs. Delays")

col3, col4 = st.columns(2)
with col3:
    st.markdown("**Insight:** The Total Service Score is the primary predictor of satisfaction.")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(x='satisfaction', y='Total_Service_Score', data=df_train, 
                palette={'0': 'lightcoral', '1': 'mediumseagreen'}, ax=ax)
    ax.set_xticklabels(["Dissatisfied (0)", "Satisfied (1)"])
    ax.set_title("Total Service Score Distribution")
    st.pyplot(fig)

with col4:
    st.markdown("**Insight:** Delayed flights show lower satisfaction, validating the delay hypothesis.")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(x='satisfaction', y='Departure_Delay_in_Minutes', data=df_train[df_train['Departure_Delay_in_Minutes'] < 100], 
                palette={'0': 'lightcoral', '1': 'mediumseagreen'}, ax=ax)
    ax.set_xticklabels(["Dissatisfied (0)", "Satisfied (1)"])
    ax.set_title("Departure Delay (<100m) vs. Satisfaction")
    st.pyplot(fig)

# 5.3 Model Performance
st.header("Model Performance Summary")
st.markdown(f"The **Random Forest Classifier** was selected as the final model due to its high performance on the test set, significantly outperforming the Logistic Regression baseline.")

st.table(pd.DataFrame({
    'Metric': ['AUC Score', 'F1-Score', 'Accuracy'],
    'Logistic Regression (Baseline)': [0.9294, 0.88, 0.88],
    'Random Forest (Final)': [0.9918, 0.95, 0.95]
}).set_index('Metric'))

st.success("Project Conclusion: Digital and Core Inflight Services (Online Boarding, Seat Comfort) are the primary drivers of passenger satisfaction.")