import streamlit as st
from multiapp import MultiApp
from apps import home, analysis,shop, clusmap,links

st.set_page_config(layout="wide")


apps = MultiApp()

# Add all your application here

apps.add_app("Home", home.app)
apps.add_app("RFM Analysis",analysis.app)
apps.add_app("Shop Mappings", shop.app)
apps.add_app("Mapping Sale Hotspots", clusmap.app)
apps.add_app("Links",links.app)
# The main app
apps.run()