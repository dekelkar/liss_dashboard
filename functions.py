import base64
import io
import re
import uuid
from dateutil import parser
import streamlit as st
import numpy as np
import datetime
from datetime import timedelta
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.statespace.sarimax import SARIMAX
register_matplotlib_converters()
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error
from math import sqrt
from datetime import date
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import pandas as pd
from openpyxl import load_workbook
from openpyxl.drawing.image import Image
import openpyxl
from openpyxl.drawing.image import Image

def create_button():
    button_uuid = str(uuid.uuid4()).replace('-', '')
    button_id = re.sub('\d+', '', button_uuid)
    custom_css = f""" 
            <style>
                #{button_id} {{
                  background: linear-gradient(30deg, #C74B76, #461F5C); 
                color: #FFFFFF;  
                font-size: 20px;  
                padding: 25px 40px;  
                border: none;  
                border-radius: 8px;
                }} 
                #{button_id}:hover {{
                    border-color: rgb(246, 51, 102);
                    color: rgb(246, 51, 102);
                }}
                #{button_id}:active {{
                    box-shadow: none;
                    background-color: rgb(246, 51, 102);
                    color: white;
                    }}
            </style> """
    return button_id, custom_css





def get_week_dates(date):
    """
    This function returns the week days for a given date
    :param date: given date (datetime)
    :return: list of all the 7 days of the specific week of the given date (list of datetime objects)
    """

    start_day = 6  # Assuming Sunday is the first day of the week (0-indexed)
    date = parser.parse(date).date()
    # Find the start of the current week (Sunday)
    while date.weekday() != start_day:
        date -= timedelta(days=1)

    # Generate a list of all dates in the week
    week_dates = []
    for i in range(7):
        week_dates.append(date.strftime("%Y-%m-%d"))
        date += timedelta(days=1)

    return week_dates

def str_to_date(date_str):
    """
    This function converts a string into date object
    :param date_str: date string
    :return: date object
    """
    date = parser.parse(date_str).date()
    return date


@st.cache_data
def create_graph(dataframe, flag):
    """
    This function creates a prediction graph
    :param dataframe: a dataframe containing 2 columns: date and  real arrivlas qty in each date
    :param flag: will be used as caching indication
    :return: line chart with 2 lines - first line for the real data and the second line is for the prediction values
    """
    # Set train and test
    df_all=dataframe
    train_all = df_all[:-7] #all the data except from the last seven days
    test_all = df_all.iloc[-7:] #the last seven days
    last_date_recorded=df_all.iloc[-1]['date']
    seven_dates = get_week_dates(last_date_recorded) #The week dates of the last date in the dataframe
    dt = datetime.date.today() #get today's value
    # x = dt.weekday()
    x=2
    if x==6:
        x=1
    elif x==7:
        x=2
    remaining_days = seven_dates[-(7 - x) - 1:] #Remaining days of the week
    past_days = seven_dates[:x] #The days of the week that passed
    num_to_predict = 7 - x #The number of days we need to predict
    history = train_all['arrivals'].values.tolist()
    test1 = test_all['arrivals'].values.tolist()
    predictions = list()

    # Performing prediciton
    if num_to_predict == 0:
        for t in range(7):
            model = SARIMAX(history, order=(2, 1, 4), seasonal_order=(1, 0, 1, 7))
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = int(output[0])
            predictions.append(yhat)
            obs = int(test1[t])
            history.append(obs)

    else:
        for t in range(num_to_predict):
            model = SARIMAX(history, order=(2, 1, 4), seasonal_order=(1, 0, 1, 7))
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = int(output[0])
            predictions.append(yhat)
            obs = int(test1[t])
            history.append(obs)

    # tomorrow_row={'date':'','arrivals':0,'patients_in_hospital':0,'patients_added':0,'patients_dismissed':0,'capacity':0,'capacity_precentage':0,'predictions_history':predictions[0]}
    # index_position = len(dataframe)  # Index of the last row in the database
    # dataframe.loc[index_position] = tomorrow_row  # Add a new row to the dataset

    # Set values for the past part of the graph
    x_past = past_days
    y_past = predictions[:len(past_days)-1]
    y_past.append(predictions[0])


    x_future = remaining_days
    y_future = predictions

    # Set real data values

    x_real_data = seven_dates[:x]
    y_real_data = df_all.iloc[-x:,1].to_list()

    # Create a graph object
    fig = go.Figure()

    # Create the future part
    fig.add_trace(go.Scatter(
        x=x_future,
        y=y_future,
        mode='lines',
        line=dict(color='#74508E', width=8),
        name='Prediction for the next days',
    ))

    # Create the past part
    fig.add_trace(go.Scatter(
        x=x_past,
        y=y_past,
        mode='lines',
        line=dict(color='#FCAA27', width=8),
        name='Predictions History',
    ))

    # Add the first line with a fixed color
    fig.add_trace(go.Scatter(
        x=x_real_data,
        y=y_real_data,
        mode='lines',
        line=dict(color='#C74B76', width=8),
        name='Real Data',
    ))

    # Set layout properties
    fig.update_layout(
        xaxis=dict(title='Date'),
        yaxis=dict(title='Admission Qty')
    )

    # Display the chart in the streamlit app
    st.plotly_chart(fig, use_container_width=True)
    in_hospital=int(dataframe.iloc[-1]['patients_in_hospital'])
    admissions=predictions[-1]
    calc_leaves=int(in_hospital*0.27+admissions-28-1)
    st.session_state.recomended_leaves=max(0,calc_leaves)
    st.session_state.prediction_for_next_day=predictions[0]
    test1_for_rmse=test1[:len(predictions)]
    st.session_state.rmse = (sqrt(mean_squared_error(test1_for_rmse, predictions))/np.mean(test1))*100







def create_excel_report(data_to_plot):
    x_values = data_to_plot['date'].to_list()
    # Line Graph Data
    days_of_week_line = x_values
    capacity_percentage = data_to_plot.capacity_precentage.to_list()  # Example capacity percentage for each day

    # Bar Chart Data
    days_of_week_bar = x_values
    admissions = data_to_plot.arrivals.to_list()
    patients = data_to_plot.patients_added.to_list()
    dismissed = data_to_plot.patients_dismissed.to_list()
    output = io.BytesIO()
    # Saving the graphs to an Excel file
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Line Graph
        line_df = pd.DataFrame({'Day': days_of_week_line, 'Capacity Percentage': capacity_percentage})
        line_df.to_excel(writer, sheet_name='Line Graph Data', index=False)
        line_ax = line_df.plot(x='Day', y='Capacity Percentage', marker='o', color='#74508E')
        line_ax.set_xlabel('Day of the Week')
        line_ax.set_ylabel('Capacity Percentage')
        line_ax.set_title('Capacity Percentage Over the Week')

        # Save Line Graph as an image
        line_image_stream = BytesIO()
        plt.savefig(line_image_stream, format='png')
        plt.close()

        # Bar Chart
        colors = ['#C74B76', '#FCAA27', '#74508E']
        bar_df = pd.DataFrame(
            {'Day': days_of_week_bar, 'Arrivals': admissions, 'Patients_added': patients, 'Dismissed': dismissed})
        bar_df.to_excel(writer, sheet_name='Bar Chart Data', index=False)
        bar_ax = bar_df.plot(x='Day', kind='bar', rot=45, width=0.7, color=colors)
        bar_ax.set_xlabel('Day of the Week')
        bar_ax.set_ylabel('Count')
        bar_ax.set_title('Admissions, Arrivals, and Dismissed Patients by Day')
        bar_ax.legend()

        # Save Bar Chart as an image
        bar_image_stream = BytesIO()
        plt.savefig(bar_image_stream, format='png')
        plt.close()

        # Insert Line Graph and Bar Chart images into Excel file
        worksheet_line = writer.sheets['Line Graph Data']
        worksheet_bar = writer.sheets['Bar Chart Data']
        line_image_stream.seek(0)
        bar_image_stream.seek(0)
        worksheet_line.insert_image('D2', '', {'image_data': line_image_stream})
        worksheet_bar.insert_image('D2', '', {'image_data': bar_image_stream})

        # Save the Excel file
        writer.close()
    b64 = base64.b64encode(output.getvalue()).decode()
    button_id, custom_css = create_button()
    label='Download Weekly report'
    file_name='Weekly report'
    href = custom_css + f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" id="{button_id}" download="{file_name}.xlsx">' \
                        f'{label}</a>'
    return href



