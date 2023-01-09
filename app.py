from flask import Flask, request, render_template
from flask_cors import cross_origin
import sklearn
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
model = pickle.load(open("rf_best.pkl", "rb"))

@app.route("/")
@cross_origin()
def home():
	return render_template("home.html")


@app.route("/predict", methods = ["GET", "POST"])
@cross_origin()
def predict():
	if request.method == "POST":

		# Date_of_Journey
		date_dep = request.form["Dep_Time"]
		Journey_day = int(pd.to_datetime(date_dep, format="%Y-%m-%dT%H:%M").day)
		Journey_month = int(pd.to_datetime(date_dep, format ="%Y-%m-%dT%H:%M").month)
		# print("Journey Date : ",Journey_day, Journey_month)

		# Departure
		Dep_hour = int(pd.to_datetime(date_dep, format ="%Y-%m-%dT%H:%M").hour)
		Dep_min = int(pd.to_datetime(date_dep, format ="%Y-%m-%dT%H:%M").minute)
		# print("Departure : ",Dep_hour, Dep_min)

		# Arrival
		date_arr = request.form["Arrival_Time"]
		Arrival_hour = int(pd.to_datetime(date_arr, format ="%Y-%m-%dT%H:%M").hour)
		Arrival_min = int(pd.to_datetime(date_arr, format ="%Y-%m-%dT%H:%M").minute)
		# print("Arrival : ", Arrival_hour, Arrival_min)

		# Duration
		dur_hour = abs(Arrival_hour - Dep_hour)
		dur_min = abs(Arrival_min - Dep_min)
		dur_total = dur_hour*60 + dur_min
		# print("Duration : ", dur_hour, dur_min)

		# Total Stops
		Total_stops = int(request.form["stops"])
		# print(Total_stops)

		# Index(['Total_Stops', 'journey_month', 'journey_day', 'dep_hour', 'dep_mins',
	   # 'arr_hour', 'arr_mins', 'dur_hours', 'dur_minutes', 'dur_total',
	   # 'air_Air India', 'air_Business Class Flights', 'air_GoAir',
	   # 'air_IndiGo', 'air_Jet Airways', 'air_Multiple carriers',
	   # 'air_Premium Economy flights', 'air_SpiceJet', 'air_Vistara',
	   # 'src_Chennai', 'src_Delhi', 'src_Kolkata', 'src_Mumbai', 'dest_Cochin',
	   # 'dest_Delhi', 'dest_Hyderabad', 'dest_Kolkata', 'dest_New Delhi'],

		# Airline
		# Working as per OneHotEncoder operation
		airline=request.form['airline']
		print(airline)
		air_zeros = [[0]*9]
		air_index = ['Air India','Business Class', 'GoAir', 'IndiGo', 'Jet Airways', 'Other carriers','Premium Economy Class',\
		'SpiceJet', 'Vistara']
		air_df = pd.DataFrame(columns=air_index, data=air_zeros)
		air_df.loc[:, airline] = 1
		air_data = air_df.iloc[0].values
		print(air_data)

		# Source
		# Banglore = 0 (not in column)
		# 'src_Chennai', 'src_Delhi', 'src_Kolkata', 'src_Mumbai',
		Source = request.form["Source"]
		src_index = ['Chennai', 'Delhi', 'Kolkata', 'Mumbai']
		src_values = [[0]*4]
		src_df = pd.DataFrame(columns=src_index, data=src_values)
		try:
			src_df[:, Source] = 1
		except:
			pass
		src_data = src_df.iloc[0].values

		# Destination
		# Banglore = 0 (not in column)
		Dest = request.form["Destination"]
		dest_index = ['Cochin', 'Hyderabad','Kolkata', 'New Delhi']
		dest_values = [[0]*4]
		dest_df = pd.DataFrame(columns=dest_index, data=dest_values)
		try:
			dest_df[:, Source] = 1
		except:
			pass
		dest_data = dest_df.iloc[0].values

	#     ['Total_Stops', 'Journey_day', 'Journey_month', 'Dep_hour',
	#    'Dep_min', 'Arrival_hour', 'Arrival_min', 'Duration_hours',
	#    'Duration_mins', 'Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo',
	#    'Airline_Jet Airways', 'Airline_Jet Airways Business',
	#    'Airline_Multiple carriers',
	#    'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',
	#    'Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy',
	#    'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',
	#    'Destination_Cochin', 'Destination_Delhi', 'Destination_Hyderabad',
	#    'Destination_Kolkata', 'Destination_New Delhi']
		
		final_array = list([Total_stops, Journey_month, Journey_day, Dep_hour, Dep_min, Arrival_hour, Arrival_min, dur_hour, dur_min, dur_total])
		final_array.extend(list(air_data))
		final_array.extend(list(src_data))
		final_array.extend(list(dest_data))
		prediction=model.predict([final_array])

		output=round(prediction[0],2)

		return render_template('home.html',prediction_text="Your Flight price should be around Rs. {}".format(output))


	return render_template("home.html")




if __name__ == "__main__":
	app.run(debug=True)
