import mainfile as mf 
import plotly.express as px
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
import warnings
import numpy as np
from pandas.plotting import scatter_matrix


warnings.filterwarnings("ignore")
st.set_page_config(page_title="Customer Segmentation", page_icon= ':bar_chart:',                   layout="wide",  
                    initial_sidebar_state="expanded")
st.markdown("<style> footer {visibility: hidden;} </style>", unsafe_allow_html=True)


with st.sidebar:
    st.title(" Customer Segmentation & Analysis")
    st.success ("This project is an End to End Data Science The Target is to  Customer Segmentation customer segmentation based on their purchasing behavior in an e-commerce dataset. The goal is to identify distinct groups of customers with similar preferences and behaviors, enabling personalized marketing strategies and recommendations. :heart_eyes: by:[Mohamed Badr](https://www.linkedin.com/in/mohamed-badr-301378248/?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app) :heart_eyes:")


    x = st.selectbox("X-axis ",mf.continuous_f)
    y = st.selectbox("Y-axis ",mf.continuous_f )
    color = st.selectbox("Color ",[None]+mf.discrete_f )
    size = st.selectbox("Size",[None]+mf.continuous_f, key="size_selectbox")

if True:
    #metric

    st.subheader('1-Metrics')
    st.markdown('''***''')
    col1 ,col2,col3,col4 =st.columns([.5,.5,.5,.5])
    col1.metric('Total Purcahse $',mf.customer_behavior['Total_spending'].sum())
    col2.metric('Total Products Purchased',mf.customer_behavior['total_products_purchased'].sum())
    col3.metric('N.Transaction',mf.customer_behavior['N.transaction'].sum())
    col4.metric('N.Proudact',len(mf.customer_behavior['unique_products'].unique()))
    st.markdown('''***''')
    
    #scatter
           
    fig, ax = plt.subplots()
    st.subheader('2-Explore the data')
    maincol1, maincol2 = st.columns([.5,.4]) 
    
    with maincol1:
        st.subheader('Scatter Plot')

        fig=px.scatter(mf.customer_behavior,x,y,color=color,size=size)
        st.plotly_chart(fig)


        st.subheader('Correlation Matrix')
        st.pyplot(mf.corrplot())

    #corr 
    with maincol2 :
        ############################
        st.subheader("shopping_day")
        #############################
        fig, ax = plt.subplots()
        target_counts=mf.customer_behavior['shopping_day'].value_counts()
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',  'Friday', 'Sunday']
        ax.pie(target_counts, labels=days, autopct="%1.1f%%", startangle=140 , **{'wedgeprops': dict(width=0.7)})
        st.pyplot(fig)

        ##############################
        st.subheader('Months Histogtam')
        ##############################
        fig = px.bar(data_frame=mf.customer_behavior, x='Month',y='Total_spending')
        st.plotly_chart(fig)
    
    ########################
    st.header('Radar Chart')
    ########################
    st.pyplot(mf.radar_chart())
    
    st.header('Clusters Profiling')
    col1 ,col2,col3,col4 =st.columns([.5,.5,.5,.5])
    
    with col1:
        st.markdown('''***
### Cluster 0:

This is a significant cluster containing as many customers as the 2nd largest cluster in total spending. Customers in this cluster :
- exhibit lower spending habits
- moderate product purchases.''')
    with col2:
        st.markdown('''***
### Cluster 1:

Customers in this cluster demonstrate distinct characteristics:

- Their average spending is notably low.
- Order cancellation rates are minimal within this cluster.
- Customers make infrequent product purchases.
- They have a relatively short tenure as customers.
- The most common period since the last purchase is 372 days, suggesting many customers might have left the market.
-Most customers in this cluster concentrate their purchases on a single product category.''') 
    with col3:
        st.markdown('''***
### Cluster 2:
This cluster comprises the highest total spending customers, and they exhibit the following characteristics:
- They have very high average spending.
- Order cancellation rates are also notably high within this cluster.
- Customers in this group make a significant number of product purchases.
- They have a longer tenure as customers, indicating loyalty.
- These customers have not churned; they are still active.
-They demonstrate a wide range of product purchases without a strong focus on specific product categories.''') 
    with col4:
        st.markdown('''***
### Cluster 3:
 This cluster is characterized by lower total spending and a smaller customer base. Customers in this cluster exhibit the following traits:

- They have a relatively high average spending level despite lower total spending.
- Many of them seem to be still active customers, with their last purchase occurring just 21 days ago.
- Within this cluster, they possess the second-longest tenure as customers, indicating some level of loyalty and engagement.''')
    
    st.dataframe(mf.cluster_profiling)
    
    st.header('3-Personalization and Recommendations')
    col1 ,col2,col3,col4 =st.columns([.5,.5,.5,.5])
    
    with col1:
        st.markdown('''***
### Cluster 0:
Low Spending, Moderate Product Purchase
- Promotions: Offer discounts or promotions to incentivize increased spending, especially on products they have shown an interest in.Product Suggestions: Recommend products related to their previous purchases to encourage cross-selling.
- Communication: Use email marketing to inform them about special offers and new product arrivals.''')
    with col2:
        st.markdown('''***
### Cluster 1:
 Very Low Spending, Low Activity
- Reactivation Campaign: Launch a reactivation campaign targeting customers with a 372-day gap since their last purchase to bring them back to the market.
- Product Focus: Highlight the single product category they are interested in to encourage repeat purchases.
- Personalized Discounts: Provide exclusive discounts to this segment to boost their engagement.
.''') 
    with col3:
        st.markdown('''***
### Cluster 2: 
 High Spending, Loyal Customers
- VIP Treatment: Recognize their loyalty with VIP status, offering early access to sales or exclusive products.
- Personalized Recommendations: Utilize data on their wide range of product purchases to provide highly personalized product recommendations.
- Loyalty Programs: Introduce loyalty programs to reward them for their continuous engagement.
***''') 
    with col4:
        st.markdown('''***
### Cluster 3: 
 High Spending, Active, and Loyal
- Exclusive Offers: Provide exclusive offers to further reward their loyalty and encourage them to continue purchasing.
- Premium Services: Consider offering premium customer services like dedicated support or early access to new releases.
- Feedback Channels: Engage with them through surveys or feedback channels to gather insights for further improvements.''')
