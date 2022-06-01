from pyparsing import alphas
import streamlit as st

import numpy as np
import pandas as pd

import pydeck as pdk
import altair as alt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# To perform KMeans clustering 
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] =(25,25)


def app():
    st.title('Analysis.')
    DATA_URL = ("tassia_mappings.csv")

    @st.cache(allow_output_mutation=True)
    def load_data():
        data = pd.read_csv(DATA_URL)
        data['ENTRY_TIME'] = pd.to_datetime(data['ENTRY_TIME'],infer_datetime_format = True, utc = True)
        return data

    # Load rows of data into the dataframe.
    df = load_data()
    #st.write(df)

    # unique sales rep in the region 
    st.subheader('Sales Representatives in Nairobi')
    un =  df['SALES_REP'].nunique()
    st.write('The number of sales representatives in Tassia Location are:',un)

    ns =  df['SALES_REP'].unique()
    st.write('They include:',ns)

    # Bar chart for Rep_Category and value sold
    st.subheader("Graphical Representation of Representative Category vs Value Sold")
    rep = pd.DataFrame(
        df.groupby(['REP_CATEGORY'])['VALUE_SOLD'].mean().sort_values(ascending=False)[:3]
        )
    st.bar_chart(rep)
    st.write("""
    Tassia KCP has the highest sales followed by  FootSoldiers-BrilliantCom
    
    """)


    # RFM
    st.subheader('RFM : Recency, Frequency, Monetary Analysis')
    st.write("""
    Marketers spend a lot to attract new customers as compared to expenses
    on retaining the current customers, To maintain and extend business, one 
    ought to realize being able to hold existing customers is as crucial as finding
    new customers. If the rate of customer retention is greater than the rate of new 
    customers, then the database as a whole is reducing if existing customers go off 
    then transactions will be less. In a way, holding current customers’ 
    priority exceeds looking for new customers.
    """)
    st.write("""
    This Analysis will help Marketers who are obsessed with customers and they can group 
    their customers and add deals accordingly.
    """)

    st.write("""
    RFM is an effective customer segmentation technique where it will be very helpful for 
    marketers, to make strategic choices in the business. It engages marketers to rapidly 
    distinguish and segment customers into similar clusters and target them with separated 
    and personalized promoting methodologies. This in turn makes strides in customer engagement
     and retention.
    """)

    st.write("""
    Using RFM segmentation marketers can able to target particular clusters of clients and target
    according to their behavior and in this way create much higher rates of customer response, 
    furthermore expands loyalty and customer lifetime. In general marketers have overall information
     on their existing customers such as buy history, browsing history, earlier campaign reaction patterns, 
     and demographics, that can be utilized to recognize a particular cluster of customers that can provide 
     offers/discounts/deals accordingly.
    """)

    monetary = df.groupby("SHOPID").VALUE_SOLD.sum()
    monetary = monetary.reset_index()
    st.markdown('Monetary')
    st.write(monetary.head())

    #Frequency function
    # Getting the count of customer verification made by each shop based on SHOPID.
    frequency = df.groupby("SHOPID").CUSTOMER_VERIFICATION.count()
    frequency = frequency.reset_index()
    st.markdown('Frequency')
    st.write(frequency.head())

    #creating master dataset
    master = monetary.merge(frequency, on = "SHOPID", how = "inner")
    st.markdown('Monetary combined with Frequency to design an RFM Clustering model')
    st.write(master.head())

    # Finding max data
    maximum = max(df.ENTRY_TIME)

    # Adding one more day to the max data, so that the max date will have 1 as the difference and not zero.
    maximum = maximum + pd.DateOffset(days = 1)

    df['DIFF'] = maximum - df['ENTRY_TIME']
    #Recency Value
    #Dataframe merging by recency
    recency = df.groupby('SHOPID')['DIFF'].min()
    recency = recency.reset_index()

    #RFM combined DataFrame
    #Combining all recency, frequency and monetary parameters
    RFM = master.merge(recency, on = "SHOPID")
    RFM.columns = ['SHOPID','VALUE_SOLD','FREQUENCY','RECENCY']

    # outlier treatment for VALUE_SOLD
    Q1 = RFM.VALUE_SOLD.quantile(0.25)
    Q3 = RFM.VALUE_SOLD.quantile(0.75)
    IQR = Q3 - Q1
    RFM = RFM[(RFM.VALUE_SOLD >= Q1 - 1.5*IQR) & (RFM.VALUE_SOLD <= Q3 + 1.5*IQR)]

    # outlier treatment for FREQUENCY
    Q1 = RFM.FREQUENCY.quantile(0.25)
    Q3 = RFM.FREQUENCY.quantile(0.75)
    IQR = Q3 - Q1
    RFM = RFM[(RFM.FREQUENCY >= Q1 - 1.5*IQR) & (RFM.FREQUENCY <= Q3 + 1.5*IQR)]


    #Scaling the RFM data
    # standardise all parameters
    RFM_norm1 = RFM.drop("SHOPID", axis=1)
    RFM_norm1.RECENCY = RFM_norm1.RECENCY.dt.days

    
    standard_scaler = StandardScaler()
    RFM_norm1 = standard_scaler.fit_transform(RFM_norm1)

    RFM_norm1 = pd.DataFrame(RFM_norm1)
    RFM_norm1.columns = ['FREQUENCY','VALUE_SOLD','RECENCY']
    st.markdown('Scaled RFM Dataframe')
    st.write(RFM_norm1.head())
    
    sse_ = []
    for k in range(2, 15):
        kmeans = KMeans(n_clusters=k).fit(RFM_norm1)
        sse_.append([k, silhouette_score(RFM_norm1, kmeans.labels_)])
    # Kmeans with K=4
    model_clus5 = KMeans(n_clusters = 4, max_iter=100)
    model_clus5.fit(RFM_norm1)

    # analysis of clusters formed
    RFM.index = pd.RangeIndex(len(RFM.index))
    RFM_km = pd.concat([RFM, pd.Series(model_clus5.labels_)], axis=1)
    RFM_km.columns = ['SHOPID', 'VALUE_SOLD', 'FREQUENCY', 'RECENCY', 'CLUSTERID']

    RFM_km.RECENCY = RFM_km.RECENCY.dt.days
    km_clusters_amount = pd.DataFrame(RFM_km.groupby(["CLUSTERID"]).VALUE_SOLD.mean())
    km_clusters_frequency = pd.DataFrame(RFM_km.groupby(["CLUSTERID"]).FREQUENCY.mean())
    km_clusters_recency = pd.DataFrame(RFM_km.groupby(["CLUSTERID"]).RECENCY.mean())

    df_rfm = pd.concat([pd.Series([0,1,2,3]), km_clusters_amount, km_clusters_frequency, km_clusters_recency], axis=1)
    df_rfm.columns = ["CLUSTERID", "VALUE_SOLD_MEAN", "FREQUENCY_MEAN", "RECENCY_MEAN"]
    st.markdown("Clustered Dataframe")
    st.write(df_rfm.head())

    st.markdown('Monetary Spent')
    st.bar_chart(km_clusters_amount)
    st.write("""
    From the above diagram ,we can note that cluster 1 had the highest spending compared to the other clusters.
    Cluster 3 had the lowest spending. This can be attributed to various factors such as how long they have been operating
    or when they joined the organisation.
    """)
    
    st.markdown('Customer Frequency')
    st.bar_chart(km_clusters_frequency)
    st.write("""
    From the above diagram ,we can note that cluster 1 and 0 are frequent customers.
    Cluster  2 and 3 have lower frequency .
    """)

    st.markdown('Recency')
    st.bar_chart(km_clusters_recency)
    st.write("""
    The less the recent value the recent customer purchased products. 
    Even here the last group has fewer values.
    """)


