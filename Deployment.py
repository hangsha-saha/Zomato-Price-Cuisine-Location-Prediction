# Importing Python Libraries  
import os
import pandas as pd 
import numpy as np 
from markupsafe import escape
import flask
import pickle
from flask import Flask, render_template, request

import sys
from importnb import Notebook

# Add the full directory to the Python path
# sys.path.append("C:/Users/hp/Desktop/SET_Project/Restaurant-Price-And-Location-Recommender/IpynbFiles")

# Import using the correct name and syntax
with Notebook():
    import InsightsGenerator

# Calling The Dictionaries From Dictionaries.py
import dictionaries 
cuidict = dictionaries.cuisinedict
addict = dictionaries.Addressdict

# Initializing Flask # Add path for pickle
app = Flask(__name__, template_folder='template')
loaded_model = pickle.load(open('MLProject.pkl','rb')) # Calling The Price Recommendation model 
model2 = pickle.load(open('rfc_model.pkl','rb')) # Calling The Location Recommendation model
@app.route('/')
def home():
    print('home')
    return render_template('index.html', cuisines=list(cuidict.keys()), locations=list(addict.keys()))


loc = ''
keyword = ''
# @app.route('/predict',methods = ['POST'])
# def result():
#     if request.method == 'POST':
#         print('predicting1')
#         cuisine = request.form.get('cuisine') # It Takes Cuisine From Webpage
        
#         price_for_one = request.form.get('price_for_one') # It Users Takes Price For One
#         print('predicting3')
     
#         Preferred_Location = request.form.get('Preferred_Location') # It Takes The Location
#         print(Preferred_Location)
#         lst1 = InsightsGenerator.main2(Preferred_Location,cuisine) # This Initializes The Main Function Of Output.ipynb File That Generates The Insights
#         location = lst1[0]
#         cuis = lst1[1]
#         a = 1
#         b = 1
#         for j in cuidict.keys(): # Searches For The Encoded Value Of Cuisine In Cuisine Dict
#             if(cuis==j):
#                 print(j)
#                 a = cuidict.get(j)*a
#         for i in addict.keys(): # Searches For The Encoded Values Of Address In Address Dict 
#             if(location==i):
#                 print(i)
#                 b = addict.get(i)*b
#         print(location,cuis)
#         lst = InsightsGenerator.main(loc = Preferred_Location, keyword = cuisine)
#         print(lst)
#         c = lst[0]
#         output = loaded_model.predict([[a, b, c]]) # Predicts The Price as Per The Pickle File
#         out = model2.predict([[a,price_for_one,c]])[0] # Predicts The Location as per The Pickle File
#         print(out)
#         locout = [i for i in addict if addict[i] == out] # Retraces The Address Predicted by model in address dict
#         print(locout)
#         return render_template('prediction.html',prediction=output[0],locationpred = locout[0], prediction1=round(lst[0],2),prediction2=lst[1],prediction3=lst[2],prediction4=lst[3],
#                               prediction5=lst[-1]) # Directs To The  Landing Page


@app.route('/predict', methods=['POST'])
def result():
    if request.method == 'POST':
        print('predicting1')
        cuisine = request.form.get('cuisine')  # Takes cuisine from webpage
        price_for_one = request.form.get('price_for_one')  # Takes price for one
        print('predicting3')
        Preferred_Location = request.form.get('Preferred_Location')  # Takes the location
        print(Preferred_Location)

        # Check if the cuisine or location exists in dictionaries
        if cuisine not in cuidict or Preferred_Location not in addict:
            print("Cuisine or location not found, redirecting to error page.")
            return render_template('error.html', message="Cuisine or location not found.")

        lst1 = InsightsGenerator.main2(Preferred_Location, cuisine)  # Generate insights
        location = lst1[0]
        cuis = lst1[1]
        a = cuidict.get(cuis, 1)  # Get encoded cuisine value
        b = addict.get(location, 1)  # Get encoded address value
        print(location, cuis)

        lst = InsightsGenerator.main(loc=Preferred_Location, keyword=cuisine)  # Get insights list
        print(lst)
        c = lst[0]

        # Predict price and location
        output = loaded_model.predict([[a, b, c]])
        out = model2.predict([[a, price_for_one, c]])[0]

        # Retrace the predicted address from the dictionary
        locout = [i for i in addict if addict[i] == out]
        print(locout)

        return render_template('prediction.html',prediction=output[0],locationpred = locout[0], prediction1=round(lst[0],2),prediction2=lst[1],prediction3=lst[2],prediction4=lst[3],
                              prediction5=lst[-1])    

if __name__ == '__main__': # Initializing The Flask
    app.run(debug=True)