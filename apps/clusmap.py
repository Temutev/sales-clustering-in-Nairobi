import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from PIL import Image
from sklearn import metrics

def app():
    st.title('Mapping Sale Hotspots.')
    DATA_URL = ("tassia_mappings.csv")

    @st.cache(allow_output_mutation=True)
    def load_data():
        data = pd.read_csv(DATA_URL)
        data['ENTRY_TIME'] = pd.to_datetime(data['ENTRY_TIME'],infer_datetime_format = True, utc = True)
        return data

    # Load rows of data into the dataframe.
    df = load_data()

    anomaly = DBSCAN(eps=0.5, min_samples=6).fit(df[['LONGITUDE','LATITUDE']])
    fig1 = px.scatter(df, x="LONGITUDE", y="LATITUDE", color="VALUE_SOLD")
    # Display the figure
    st.plotly_chart(fig1, use_container_width=True)
    st.write("""
    The graph is used to identify the desirable hotspot shapes. 
    The process of finding a cluster is aimed at detecting conglomerate cities
     based on their sales volume.
    """)

    st.subheader("Heatmap")
    #state['sales'] = state['geolocation_city'].map(geolocation['geolocation_city'].value_counts())
    fig2 = px.density_mapbox(df, lat='LATITUDE', lon='LONGITUDE', z='VALUE_SOLD', radius=50, zoom=12,
                            hover_name="LOCATION_NAME", mapbox_style="stamen-toner")
    #fig2.show()
    # Display the figure
    st.plotly_chart(fig2, use_container_width=True)

    st.write("""
    The heatmap below above where sales concentration is.These regions represent the expected shapes for
     identification made by machine.If you zoom in on the map, you can see, in detail, where sales are concentrated. 
    """)
    st.subheader("DBSCAN")

    st.write("""
    For the identification of these shapes, density-based clustering was used, due to their ability to identify
    arbitrary shapes in spatial data. These algorithms separate the data from higher to lower dense. It requires two parameters, 
    the radius (ε — Epsilon)and the minimum number of points required to form a dense region (MinPts). A point is a Core Point 
    if it has more than a specified number of points (MinPts) within Epsilon (ε)— these are points that the interior of the cluster. 
    A Border Point has fewer than within Epsilon, but in the neighborhood of the core point. A Noise Point is any point that is not a 
    core point nor a border point.
    """)
    image = Image.open('cluster.png')
    st.image(image, caption='DBSCAN Clustering illustration')

    st.write("""
    The algorithm correctly finds the density clusters on the map, which are the locations. This occurs because the points of sale 
    tend to concentrate in the locations with shorter distances between them. Thus, it makes the algorithm identify each location as a 
    cluster and scattered points as a noise. However, the shape we are seeking represents a conglomerate of locations with the highest 
    density of sales, not only isolated points of locations.
    """)
    
    clustering2 = DBSCAN(eps=0.014, min_samples=15).fit(df[['LONGITUDE','LATITUDE']])
    x = lambda x: "noise" if x == -1 else str(x)
    df['clusters_BDSCAN'] = list(map(x,list(clustering2.labels_)))
    fig3 = px.scatter_mapbox(df[df['clusters_BDSCAN'] != 'noise'], lat="LATITUDE", lon="LONGITUDE",
                            color="clusters_BDSCAN", title='DBSCAN Clustering', hover_data=["VALUE_SOLD"],
                            hover_name="LOCATION_NAME",zoom=11,
                            width=1000, height=500, mapbox_style="stamen-toner")
    #fig3.show()
    # Display the figure
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("OPTICS")
    st.write("""
    Finally, the multi-scale OPTICS algorithm was used, which is not based on a certain distance, but on the peaks and valleys
    within the graph. It made it possible to offer greater flexibility in the calibration of the detected clusters. In this way,
    the algorithm found only four clusters this time, which represents double of that found by DBSCAN. Consequently, being more specific 
    for which regions represent high sales density. 
    """)


    clust = OPTICS(min_samples=70, xi=.06, min_cluster_size=.06)
    clust.fit(df[['LONGITUDE','LATITUDE']])
    df['clusters_OPTICS'] = list(map(x,list(clust.labels_)))
    fig5 = px.scatter_mapbox(df[df['clusters_OPTICS'] != 'noise'], lat="LATITUDE", lon="LONGITUDE",
                            color="clusters_OPTICS", title="OPTICS ", hover_name="LOCATION_NAME", hover_data=["VALUE_SOLD"],
                            zoom=10, width=1000, height=500, mapbox_style="stamen-toner")
    fig5.show()
    st.plotly_chart(fig5, use_container_width=True)

    st.subheader("Silhouette Score")
    st.write("""
    After applied the clustering algorithm, the silhouette score of each of them is evaluated. 
    The silhouette value is a measure of how similar an object is to its own cluster (cohesion) 
    compared to other clusters (separation). The best value is 1 and the worst value is -1. 
    Values near 0 indicate overlapping clusters. Negative values generally indicate that a sample 
    has been assigned to the wrong cluster, as a different cluster is more similar.

    All algorithm shows reasonable good score values. Optics demonstrates the best score performance, 
    followed by DBSCAN.
    """)

    clusters_DBSCAN = df[df['clusters_BDSCAN'] != 'noise']
    clusters_OPTICS = df[df['clusters_OPTICS'] != 'noise']


    st.write("DBSCAN: ",metrics.silhouette_score(clusters_DBSCAN[['LONGITUDE','LATITUDE']],
                                            clusters_DBSCAN['clusters_BDSCAN']))
    st.write("OPTICS: ",metrics.silhouette_score(clusters_OPTICS[['LONGITUDE','LATITUDE']],
                                            clusters_OPTICS['clusters_OPTICS']))

    st.subheader("Davies Bouldin Score")
    st.write("""

    The Davies Bouldin score validates how well the clustering has been done is made using quantities and 
    features inherent to the dataset. The minimum score is zero, with lower values indicating better clustering.
    However, a good value does not imply the best information restoration .
    """)

    
    st.write("DBSCAN: ",metrics.davies_bouldin_score(clusters_DBSCAN[['LONGITUDE','LATITUDE']],
                                                clusters_DBSCAN['clusters_BDSCAN']))
    st.write("OPTICS: ",metrics.davies_bouldin_score(clusters_OPTICS[['LONGITUDE','LATITUDE']],
                                              clusters_OPTICS['clusters_OPTICS']))