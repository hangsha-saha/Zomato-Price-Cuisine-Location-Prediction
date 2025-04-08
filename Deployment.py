import streamlit as st
import pandas as pd
import pickle
from importnb import Notebook

# Importing from notebook
with Notebook():
    import InsightsGenerator

# Importing dictionaries
import dictionaries
cuidict = dictionaries.cuisinedict
addict = dictionaries.Addressdict

# Loading models
loaded_model = pickle.load(open('MLProject.pkl', 'rb'))
model2 = pickle.load(open('rfc_model.pkl', 'rb'))

# Streamlit UI
st.set_page_config(page_title="Zomato Cuisine & Price Predictor", layout="centered")
st.title("🍽️ Cuisine, Price & Location Predictor")

# Input form
with st.form("prediction_form"):
    cuisine = st.selectbox("Select a Cuisine", list(cuidict.keys()))
    preferred_location = st.selectbox("Select a Location", list(addict.keys()))
    price_for_one = st.number_input("Enter Price for One", min_value=1.0)

    submit = st.form_submit_button("Predict")

if submit:
    try:
        # Generate insights
        lst1 = InsightsGenerator.main2(preferred_location, cuisine)
        location = lst1[0]
        cuis = lst1[1]

        a = cuidict.get(cuis, 1)
        b = addict.get(location, 1)

        lst = InsightsGenerator.main(loc=preferred_location, keyword=cuisine)
        c = lst[0]

        output = loaded_model.predict([[a, b, c]])
        out = model2.predict([[a, price_for_one, c]])[0]
        locout = [i for i in addict if addict[i] == out]

        # Display Results
        st.subheader("🔮 Prediction Results:")
        st.markdown(f"**Predicted Price for One:** ₹ {round(output[0], 2)}")
        st.markdown(f"**Suggested Location:** {locout[0]}")
        st.markdown(f"**Insights:**")
        st.markdown(f"- Average Price: ₹ {round(lst[0], 2)}")
        st.markdown(f"- Nearby Top-Rated Restaurants: {lst[1]}")
        st.markdown(f"- Popular Dishes: {lst[2]}")
        st.markdown(f"- Customer Reviews: {lst[3]}")
        st.markdown(f"- Additional Insight: {lst[-1]}")
        
    except Exception as e:
        st.error(f"⚠️ An error occurred: {str(e)}")