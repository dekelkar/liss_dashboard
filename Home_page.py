import base64
import datetime

import july
import numpy as np
import streamlit as st
from PIL import Image
from matplotlib import pyplot as plt
from pandas import date_range
from datetime import date
import functions
import pandas as pd
from datetime import datetime, timedelta

from datetime import datetime
from datetime import timedelta
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


st.set_page_config(
    page_title="Liss dash",
    page_icon="👶",
    layout='wide'
)

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )


add_bg_from_local('background.jpg')

if 'database' not in st.session_state:
    df = pd.read_csv('arrivals_all.csv')
    columns=df.columns
    # df=df[['date','arrivals','patients_in_hospital','patients_added','patients_dismissed','capacity','capacity_precentage','predictions_history']]
    st.session_state.database = df
if 'flag' not in st.session_state:
    st.session_state.flag=1
if 'recomended_leaves' not in st.session_state:
    st.session_state.recomended_leaves='10'
if 'data_for_report' not in st.session_state:
    st.session_state.data_for_report=pd.read_excel('report_testtttt.xlsx')
if 'rmse' not in st.session_state:
    st.session_state.rmse=12
if 'prediction_for_next_day' not in st.session_state:
    st.session_state.prediction_for_next_day='TBD'






with st.sidebar:
    image=Image.open('img.png')
    image1=Image.open('img_1.png')
    st.image(image)
    st.title("מחלקת טרום לידה - ליס בית חולים ליולדות")
col11,col22=st.columns([5,2])
with col11:
    st.markdown("""
        <style>
        .title-wrapper > h1 {
            color: #F7630C;  /* Set the color */
        }
        </style>
    """, unsafe_allow_html=True)
    st.title('Patient Admission Prediction Tool')

with col22:
    font = "Rubik"
    font_size='10px'
    metric_label="➕📝Insert today's data"
    label_html = f'<span style="font-family: {font}; font-size: {font_size};"></span>'
    st.markdown(f"<h3>{metric_label}</h3>", unsafe_allow_html=True)
    st.markdown(label_html, unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)


    def form():
        def update_database():
            """
            This function is triggered once a user updates value of admissions
            in the current day, and updated the database accordingly
            """
            admission = int(st.session_state.admissions_live) # get the user input
            patients=int(st.session_state.patients)
            dismissed=int(st.session_state.dismissed)
            current_date = date.today()  # get the current day
            yesterday = current_date - timedelta(days=1)
            current_date = current_date.strftime("%Y-%m-%d")
            yesterday=yesterday.strftime("%Y-%m-%d")
            database = st.session_state.database  # Load the database from session states
            date_exists = database['date'].isin([current_date]).any()  # Check if there is an input for today
            row = database.iloc[-1]
            updated_database = database  # Copy the databse, to avoid performing changes on the original database
            if not date_exists:  # In case this is the first input for today
                patients_in_hospital = int(row[2])
                capacity=patients_in_hospital+int(patients)-int(dismissed)
                capacity_precentage=(capacity/28)*100


                index=len(database)-1
                updated_database.iloc[index,0]=current_date
                updated_database.iloc[index, 1] = admission
                updated_database.iloc[index, 2] = patients_in_hospital
                updated_database.iloc[index, 3] = patients
                updated_database.iloc[index, 4] = dismissed
                updated_database.iloc[index, 5] = capacity
                updated_database.iloc[index, 6] = capacity_precentage

            else:  # In case this is not the first input today
                index=len(database)-1
                updated_database.iloc[index, 1] = int(admission) # Update the value of today's row
                updated_database.iloc[index, 3] = int(patients)  # Update the value of today's row
                updated_database.iloc[index, 4] = int(dismissed)  # Update the value of today's row
                row=updated_database.iloc[-1]
                x=row[2]
                y=row[3]
                z=row[4]
                capacity=int(row[2]+row[3]-row[4])
                capacity_prec=(capacity/28)*100
                updated_database.iloc[index, 5]  = capacity
                updated_database.iloc[index, 6]= capacity_prec
            st.session_state.database = updated_database  # Update the databse in the session states
            updated_database.to_csv('arrivals_all.csv',index=False)
            st.session_state.data_for_report=updated_database[-7:]
            st.session_state.flag += 1
            print('------------------------------------------------update database flag', st.session_state.flag)
        with st.form("Update today's data",clear_on_submit=True):
            col1,col2,col3=st.columns(3)
            with col1:
                st.text_input(label='🤰Admissions',key='admissions_live')
            with col2:
                st.text_input(label='➕Patients', key='patients')
            with col3:
                st.text_input(label='➖Patients', key='dismissed')

            submit=st.form_submit_button(label='Submit Data',on_click=update_database)

def main():
    form()

main()

col1,col2=st.columns([2,5])
font_size_label="30px"
with col2:
    metric_label_graph = "🤰Upcoming Admissions"
    metric_label_graph_html=f'<p style="text-align: left;"><span style="font-family: {font}; font-size: {font_size_label};">{metric_label_graph}</span></p>'
    metric_graph_html = f'<span style="font-family: {font}; font-size: {font_size};"></span>'
    st.markdown(metric_label_graph_html, unsafe_allow_html=True)
    functions.create_graph(st.session_state.database, st.session_state.flag)
    st.markdown("""
    <h3><span style="background-color: #C74B76;"><strong>Model prediction error: {}% </strong></span></h3>""".format(round(st.session_state.rmse, 2)),unsafe_allow_html=True)


with col1:
    metric_value = st.session_state.recomended_leaves
    metric_label = "🦋Recomended leaves"
    font = "Rubik Black"
    font_size = "50px"
    metric_label = f'<p style="text-align: left;"><span style="font-family: {font}; font-size: {font_size_label};">{metric_label}</span></p>'
    metric_html = f'<p style="text-align: center;"><span style="font-family: {font}; font-size: {font_size};">{metric_value}</span></p>'
    st.markdown(f"<h3>{metric_label}</h3>", unsafe_allow_html=True)
    st.markdown(metric_html, unsafe_allow_html=True)

    metric_value = st.session_state.prediction_for_next_day
    metric_label = "📊Prediction for tomorrow"
    font = "Rubik Black"
    font_size = "50px"
    font_size_label = "30px"
    metric_label = f'<p style="text-align: left;"><span style="font-family: {font}; font-size: {font_size_label};">{metric_label}</span></p>'
    metric_html = f'<p style="text-align: center;"><span style="font-family: {font}; font-size: {font_size};">{metric_value}</span></p>'
    st.markdown(f"<h3>{metric_label}</h3>", unsafe_allow_html=True)
    st.markdown(metric_html, unsafe_allow_html=True)


st.markdown("""
    <style>
    .stButton button {
        background: linear-gradient(30deg, #C74B76, #461F5C);  /* Set the gradient colors and direction */
        color: #FFFFFF;  /* Set the text color */
        font-size: 20px;  /* Set the font size */
        padding: 25px 40px;  /* Set the button padding */
        border: none;  /* Remove the button border */
        border-radius: 8px;  /* Add rounded corners to the button */
    }
    </style>
""", unsafe_allow_html=True)

st.markdown(functions.create_excel_report(st.session_state.data_for_report), unsafe_allow_html=True)


