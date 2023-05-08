import streamlit as st
from datetime import date
from csv import reader
import pandas as pd
from plotly import graph_objs as go
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import plotly.express as px
import spacy


clf = LinearRegression()

#-----------------------------------------------------------------

st.set_page_config(
    page_title="Homicide Crime Prediction",
    page_icon="👋",
)

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")


#------------------------------------------------------------------
st.title('Homicide Crime Prediction')
#------------------------------------------------------------------

#@st.cache(allow_output_mutation=True)
#def loadmodel(modelname):
#	nlp = spacy.load(model)
#	return (nlp)


crimedata = st.file_uploader("upload file", type={"csv", "txt"})
crimedata_df=""
if crimedata is not None:
    crimedata_df = pd.read_csv(crimedata)

crimedata_df_dataframe=pd.DataFrame( list(reader(crimedata_df)))
st.write(crimedata_df)

# Plot raw data
try:
	#Group our data by years
	dfByYear = crimedata_df[1:].groupby('Year').size().reset_index(name='counts')
	def plot_raw_data():
		fig = go.Figure()
		fig.add_trace(go.Scatter(x=dfByYear['Year'], y=dfByYear['counts'], mode='lines+markers', name="No. Of Crime"))
		fig.add_trace(go.Scatter(x=crimedata_df['Year'][1:], y=crimedata_df['age'], mode='lines+markers', name="Age"))
		fig.layout.update(title_text='Graph Of The Crime Per Year And Age', xaxis_rangeslider_visible=True)
		st.plotly_chart(fig)
			
	plot_raw_data()
	

except:
	pass

try:
	crimedata_df['Year'].value_counts()
	plt.figure(figsize=(10,5))
	sns.countplot(crimedata_df['Year'][1:])
	st.pyplot()
except:
	pass

try:
	st.write("This Is Your Data Shape (Row, Column): ", crimedata_df.shape)

	st.write("\n")
	st.write("\n")
	st.write("\n")


	col1, col2 = st.columns(2)

	with col1:
		st.write("These Are Nulls Data")
		st.write(crimedata_df.isnull().sum())


	with col2:
		st.write("Counting Of Crime Per Year")
		st.write(dfByYear)
		



#------------------------------------------------------------------

	

	correlation = crimedata_df.corr()
	# constructing a heatmap to understand the correlation
	fig, ax = plt.subplots()
	sns.heatmap(correlation, cbar=True, square=True, fmt='.1f',annot=True, annot_kws={'size':8}, cmap='RdBu', vmin= -1, vmax=1, ax=ax)
	st.write("\n")
	st.write("\n")

	st.write("This Is The Crrelation Between The Data ")
	st.write(fig)

#------------------------------------------------------------------

	# encoding "Gender_Type" Column
	crimedata_df.replace({'gender':{'male':0,'female':1, 'unknown':2}},inplace=True)

#------------------------------------------------------------------

	X = crimedata_df.drop(['first_name', 'last_name', 'race', 'cause', 'death_loc', 'district', 'street_address', 'date', 'time','notes','gender'],axis=1)
	Y = crimedata_df['gender']

#------------------------------------------------------------------

	#Splitting into Training data and Test Data
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2) # random_state=2

	st.write("\n")
	st.write("\n")
	st.write("Full Shape, Train Shape, Test Shape: ", X.shape, X_train.shape, X_test.shape)

#------------------------------------------------------------------

	#Group our data by years
	dfByYear = crimedata_df[1:].groupby('Year').size().reset_index(name='counts')

	

	#Train by using new data based on years and number of crimes in this year
	n_x_train, n_x_test, n_y_train, n_y_test = train_test_split(dfByYear['Year'], dfByYear['counts'], test_size = 0.2)

	#fit them with the classifier
	clf.fit(n_x_train.values.reshape(-1, 1), n_y_train)

	st.write("\n")
	st.write("\n")

	#Users Prediction Year
	TheYear = st.text_input('Enter The Year: ', '2024')

	#predict the number of crimes for year 2025
	year = clf.predict(np.array([[int(TheYear)]]))[0]


	if st.button('amount'):
		st.write('The Predicted Amount Is: ', int(year))
	
	#--------------------------------------------------------------

	model = LogisticRegression()

	# Training the LogisticRegression model
	model.fit(X_train,Y_train)

	X_train_prediction = model.predict(X_train)
	training_data_accuracy = accuracy_score(X_train_prediction,Y_train)

	# accuracy on test data
	X_test_prediction = model.predict(X_test)
	test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

	#input_data = (3402, 20, 21215, 39.358600, -76.702790, 2019)
	input_d = st.text_input('Enter Data: ', '3402, 20, 21215, 39.358600, -76.702790, 2019')

	input_d = '3402, 20, 21215, 39.358600, -76.702790, 2019'
	input_list = [item for item in input_d.split(',') if item]

	res = [eval(i) for i in input_list]

	inputdata = tuple(res)


	# change the input data to a numpy array
	input_data_as_numpy_array = np.asarray(inputdata)

	# reshape the numpy array as we are predicting for only on instance
	input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

	prediction = model.predict(input_data_reshaped)

	if st.button('Gender'):
		if(prediction[0]==0):
			st.write('The Person Gender is a Male')
		else:
			st.write('The Person Gender is a Female')

#------------------------------------------------------------------
except:
  st.write("Upload Your CSV File, Please!")

#------------------------------------------------------------------

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

#------------------------------------------------------------------