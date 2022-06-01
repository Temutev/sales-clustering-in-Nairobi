import streamlit as st
import numpy as np 
import pandas as pd 

def app():
    st.title("Clustering Product Sales within Nairobi.")

    st.header("Introduction")
    st.markdown(
        """
        Data can greatly improve the flow and conversions at every step 
        of a business' digital marketing funnel from sparking the interest
        of a possible customer to a customer converting with a purchase of
        a product or service.
    """
    )
    st.markdown(
        """
        As such, the data used is a representation of sales that have been done for different consumers 
        within Nairobi . 
        The purpose of this experiment is to provide insights on the data gathered.
        Here is an overview of the dataframe .
    """
    )
    DATA_URL = ("tassia_mappings.csv")


    @st.cache(allow_output_mutation=True)
    def load_data():
        data = pd.read_csv(DATA_URL)
        data['ENTRY_TIME'] = pd.to_datetime(data['ENTRY_TIME'],infer_datetime_format = True, utc = True)
        return data


    # Load rows of data into the dataframe.
    df = load_data()
    st.write(df.head())
