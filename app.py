import streamlit as st
import joblib
import pandas as pd
import numpy as np


# ==== cache data and models ===
@st.cache_resource
def load_models():
    labelEnc = joblib.load("models/labelEnc.pkl")
    # print(labelEnc.classes_)
    TargetEnc = joblib.load("models/TargetEnc.pkl")
    # print(TargetEnc.mapping)
    model = joblib.load("models/car_price_pred_model.pkl")
    model.set_params(predictor='cpu_predictor', device='cpu')
    # print(model)
    return labelEnc, TargetEnc, model

@st.cache_data
def load_csv():
    return pd.read_csv("cleaned_data.csv")

labelEnc, TargetEnc, model = load_models()
df = load_csv()

# st.set_option()
st.title("Used Car Price Prediction")
st.subheader("Predict the price of your choice of cars.")
# ===================================
# styling
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: red;
        color: white;
        border: none;
        padding: 0.6em 1.2em;
        border-radius: 8px;
        font-size: 18px;
        font-weight: bold;
        transition: 0.3s;
    }
    div.stButton > button:first-child:hover {
        background-color: darkred;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)
# ============================

with st.container(border=True):
    st.html("<h4>Input the data</h4>")

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        manuf = st.selectbox("Manufacturer", df["manufacturer"].unique())
        # st.write(manuf)

        km_drive = st.number_input("KM_Driven", min_value=100)
        # st.write(km_drive)

        seller = st.selectbox("Car seller - 1:Dealer, 0:Individual", [0,1])
        # st.write(seller)

        manuf_year = st.selectbox("Select manufactured year", list(range(2008, 2022)))
        manuf_year = 2025 - manuf_year
        # st.write(manuf_year)
    
    with col2:
        car_model = st.selectbox("Model", df[df["manufacturer"] == manuf]["model"].unique())
        # st.write(car_model)
        
        st.write("if you want to predict price of insured vehicle")
        insurance = int(st.checkbox("Insurance Premium"))
        # st.write(insurance)

        owner = st.selectbox("owner", [1,2,3,4,5])
        # st.write(owner)

        features_count = st.selectbox("Select number of top features in car", [0, 1, 2, 6, 7, 8, 9])
        # st.write(features_count)

    
    states = [
    'andhra pradesh', 'arunachal pradesh', 'assam', 'bihar', 'chandigarh',
    'chhattisgarh', 'dadra and nagar haveli', 'delhi', 'goa', 'gujarat',
    'haryana', 'himachal pradesh', 'jammu & kashmir', 'jharkhand', 'karnataka',
    'kerala', 'madhya pradesh', 'maharashtra', 'manipur', 'meghalaya', 'nagaland',
    'odisha', 'pondicherry', 'punjab', 'rajasthan', 'sikkim', 'tamil nadu',
    'telangana', 'tripura', 'uttar pradesh', 'uttarakhand', 'west bengal'
    ]
    selected_state = st.selectbox("Select states", states)

    state_vector = [1 if state == selected_state else 0 for state in states]

    st.write("Selected State:", selected_state)
    # st.write("One-hot encoded vector:", state_vector)

    # handling inputs

    manuf_encoded = int(labelEnc.transform([manuf])[0])
    temp_df = pd.DataFrame({'model': [car_model]})
    model_encoded = TargetEnc.transform(temp_df)["model"].iloc[0]
    del temp_df
    # st.write(manuf_encoded)
    # st.write(model_encoded)

    final_input = [manuf_encoded, model_encoded, km_drive, insurance, seller, owner, manuf_year, features_count]
    final_input.extend(state_vector)
    # st.write(final_input)

    # st.write(dm)
    # prediction
    st.write(final_input)
    if st.button("Predict price"):
        final_input = np.array(final_input, dtype=np.float32).reshape(1, -1)
        st.write(final_input)
        prediction = model.predict(final_input)[0]
        st.success(f"Estimated Car price: Rs. {int(prediction)}")
