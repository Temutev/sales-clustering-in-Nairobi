import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def app():
    st.title('Shop Mapping in Relation to other Variables.')
    DATA_URL = ("tassia_mappings.csv")

    @st.cache(allow_output_mutation=True)
    def load_data():
        data = pd.read_csv(DATA_URL)
        data['ENTRY_TIME'] = pd.to_datetime(data['ENTRY_TIME'],infer_datetime_format = True, utc = True)
        return data

    # Load rows of data into the dataframe.
    df = load_data()
    st.subheader('Shop Mapping based on Product Category and Values Sold.')
    px.set_mapbox_access_token("pk.eyJ1IjoidGV2aW50ZW11IiwiYSI6ImNsM3ZmcDZ6MTAwMW8zYnFncnk0eTE5Z2YifQ.eFQUPP_jh_WQo0YgsNUqPg")
    fig = px.scatter_mapbox(df, lat="LATITUDE", lon="LONGITUDE",color="PRODUCT_CATEGORY", size="VALUE_SOLD",
                    color_continuous_scale=px.colors.cyclical.IceFire, size_max=20,zoom=12)
    #fig.show()
    
    # Display the figure
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('Shop Mapping based on Representative Category and Value Sold.')
    fig01 = px.scatter_mapbox(df, lat="LATITUDE", lon="LONGITUDE",color="REP_CATEGORY", size="VALUE_SOLD",
                    color_continuous_scale=px.colors.cyclical.IceFire, size_max=20,zoom=12)
    #fig.show()
    
    # Display the figure
    st.plotly_chart(fig01, use_container_width=True)



    st.subheader('Shop Mapping based on Sales Representatives and Value Sold.')
    fig02 = px.scatter_mapbox(df, lat="LATITUDE", lon="LONGITUDE",color="SALES_REP", size="VALUE_SOLD",
                    color_continuous_scale=px.colors.cyclical.IceFire, size_max=20,zoom=12)
    #fig.show()
    
    # Display the figure
    st.plotly_chart(fig02, use_container_width=True)
