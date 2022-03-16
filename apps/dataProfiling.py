import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import math

from streamlit_pandas_profiling import st_profile_report
from pandas_profiling import ProfileReport

from dataclasses import dataclass

import os
basePath = os.getcwd()

def app():

    st.header("Data Profiling")
    st.write("")


    ######################################
    ## Sidebar
    ######################################
    # Input your csv
    st.markdown("""<style>
    .sidebar .sidebar-content {
    background-color: #3090C7;
    background-image: linear-gradient(#3090C7,#3090C7);
    color: white; }
    </style>""", unsafe_allow_html=True)

    st.sidebar.header('Upload your CSV data')
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

    print("*"*100)
    print(uploaded_file)

    st.sidebar.markdown("""
    [Example CSV input file]
    ({}/easyInput/Apr21Data.csv)
    """.format(basePath))

    ######################################
    # Main panel
    ######################################
    st.subheader('Dataset')

    @dataclass(unsafe_hash=True)
    class FileInfo:
        name: str
        size: float

    def convert_size(size_bytes):
       if size_bytes == 0:
           return "0B"
       size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
       i = int(math.floor(math.log(size_bytes, 1024)))
       p = math.pow(1024, i)
       s = round(size_bytes / p, 2)
       return "%s %s" % (s, size_name[i])

    def barPlot(data, title):
        xVal = list(data.keys())
        yVal = list(data.values())

        figure = px.bar(x=xVal,
                        y=yVal, labels = {'x':'Feature', 'y':'Null Ratio'},
                        text=np.round(yVal, 2),
                        title=title,
                        width=1000, height=600)
        figure.update_layout({
            'plot_bgcolor': 'rgba(0, 0, 0, 0)',
            'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        })
        st.plotly_chart(figure)

    def getPandasProfile(df, uploadedFile):
        profile = ProfileReport(df,
                                title="Accounting Data Profile",
                dataset={
                "description": "This profiling report was generated for Analytics Vidhya Blog",
                "copyright_holder": "ByteIQ",
                "url": "ByteIQDatasetProfile",
            },
            variables={
                "descriptions": {
                    "File Name": uploadedFile.name,
                    "File Size": convert_size(uploadedFile.size)
                }
            }
        )
        st_profile_report(profile)


    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(df.head(5))
        na_distribution = dict(pd.isna(df).mean().reset_index().values)
        barPlot(na_distribution, "Nulls Ratio in Dataset")
        getPandasProfile(df, uploaded_file)
    else:
        st.info('Upload the data in csv format (comma separated)')
        if st.button('Press to use Example Dataset'):
            df = pd.read_csv(
                '{}/easyInput/Apr21Data.csv'.format(basePath))
            st.markdown('The **Apr21Data.csv** dataset is used as the example.')
            st.write(df.head(5))
            na_distribution = dict(pd.isna(df).mean().reset_index().values)
            barPlot(na_distribution, "Nulls Ratio in Dataset")
            getPandasProfile(df, FileInfo("Apr21Data.csv",17786691))
