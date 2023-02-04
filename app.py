from flask import Flask, request, render_template
from flask_cors import cross_origin
import statistics
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
import plotly
import json

app = Flask(__name__)

with open("models/rf_grid.pkl", "rb") as f:
	rf_model = pickle.load(f)

with open("models/etc_grid.pkl", "rb") as f:
	etc_model = pickle.load(f)

with open("models/lgbm_grid.pkl", "rb") as f:
	lgbm_model = pickle.load(f)

with open("models/lasso.pkl", "rb") as f:
	lasso_model = pickle.load(f)

locations = pd.read_csv('locations.csv')


@app.route("/")
@cross_origin()
def home():
	print("$$$$$$$$$$ Throwing default home $$$$$$$$$$$$$$$$$")
	return render_template("home.html", show='none')


def gm(src, dest):
	df = locations[locations['city'].isin([src, dest])]
	fig = px.line_mapbox(df,
						 lat="lat", lon="lon", zoom=3)
	fig.update_layout(mapbox_style="open-street-map", mapbox_zoom=4,
    margin={"r":0,"t":0,"l":0,"b":0})

	return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


@app.route("/predict", methods=["GET", "POST"])
@cross_origin()
def predict():
	if request.method != "POST":
		return render_template("home.html", show='none')
	# Date_of_Journey
	date_dep = request.form["Dep_Time"]
	dep_weekend = int(pd.to_datetime(
		date_dep, format="%Y-%m-%dT%H:%M").day_of_week)
	dep_weekend = 1 if dep_weekend in {5, 6} else 0
	dep_month = int(pd.to_datetime(
		date_dep, format="%Y-%m-%dT%H:%M").month)
	print(dep_weekend, dep_month)

	# Total Stops
	stops = request.form["stops"]
	stops_index = ["1", "2", "3", "4"]
	stops_data = np.zeros(4)
	if stops != 0:
		idx = np.where(stops_index == stops)
		stops_data[idx] = 1
	print(stops)

	# Airline
	airline = request.form['airline']
	print(airline)
	air_index = ['Air India', 'Business', 'GoAir', 'IndiGo', 'Jet Airways',
	 'Other', 'PremiumEcon', 'SpiceJet', 'Vistara']
	air_data = np.zeros(9, dtype='int8')
	if airline != 'Air Asia':
		air_df = pd.DataFrame(columns=air_index)
		air_df.loc[0] = air_data
		air_df.loc[:, airline] = 1
		air_data = air_df.to_numpy()[0]

	# Source
	# Bangalore = 0 (not in column)
	# 'Source_Chennai', 'Source_Kolkata', 'Source_Mumbai', 'Source_New Delhi',
	source = request.form["Source"]
	print(source)
	src_index = ['Chennai', 'Kolkata', 'Mumbai', 'New Delhi']
	src_data = np.zeros(4, dtype='int8')
	if source != 'Bangalore':
		src_df = pd.DataFrame(columns=src_index)
		src_df.loc[0] = src_data
		src_df.loc[:, source] = 1
		src_data = src_df.to_numpy()[0]

	# Destination
	# Bangalore = 0 (not in column)
	# 'Cochin', 'Hyderabad', 'Kolkata', 'New Delhi',
	dest = request.form["Destination"]
	print(dest)
	dest_index = ['Cochin', 'Hyderabad', 'Kolkata', 'New Delhi']
	dest_data = np.zeros(4, dtype='int8')
	if dest != 'Bangalore':
		dest_df = pd.DataFrame(columns=dest_index)
		dest_df.loc[0] = dest_data
		dest_df.loc[:, dest] = 1
		dest_data = dest_df.to_numpy()[0]

	mnth_data = np.zeros(3)
	# Month
	if dep_month in {4, 5, 6}:
		mnth_index = ['M4', 'M5', 'M6']
		dep_month = f"M{dep_month}"
		mnth_data[mnth_index.index(dep_month)] = 1

	final_array = [dep_weekend] +\
		list(air_data) +\
		list(src_data) +\
		list(dest_data) +\
		list(stops_data) +\
		list(mnth_data)

	# Average the ensemble predictions
	prediction = statistics.mean([
		rf_model.predict([final_array])[0],
		etc_model.predict([final_array])[0],
		lgbm_model.predict([final_array])[0],
		lasso_model.predict([final_array])[0]
	])

	output = round(prediction, 2)

	print("$$$$$$$$$$ Throwing CUSTOM home $$$$$$$$$$$$$$$$$")
	details={'source':source, 'dest': dest, 'date': date_dep,'stops': 'direct' if stops=='0' else f"{stops} stop"}
	print(details)
	return render_template('home.html',
					   prediction_text = f"should cost approximately ₹ {output}", 
					   details = f"Your {details['stops']} flight from {details['source']} to {details['dest']} on {details['date']}",
					   graphJson = gm(source, dest),
					   show = 'block')


if __name__ == "__main__":
	app.run(debug=True)
