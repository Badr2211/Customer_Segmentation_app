# Customer_Segmentation_app
This project is an End to End Data Science The Target is to  Customer Segmentation customer segmentation based on their purchasing behavior in an e-commerce dataset. The goal is to identify distinct groups of customers with similar preferences and behaviors, enabling personalized marketing strategies and recommendations. 

# Customer Segmentation and Analysis

This project involves the analysis and segmentation of customer behavior using an Online Retail dataset. The main objectives are to preprocess the data, identify clusters of customers, and perform exploratory data analysis (EDA).

## Overview

The project is structured into several main components:

1. **Data Loading and Preprocessing:**
    - The data is loaded from the 'Online Retail.xlsx' file.
    - The 'Country' column is cleaned by replacing 'Israel' with 'Palestine'.
    - Canceled transactions are identified, and additional features related to customer behavior are engineered.

2. **Outlier Detection:**
    - An Isolation Forest model is used to identify and remove outliers from the customer behavior data.

3. **Principal Component Analysis (PCA):**
    - Standardization of features is performed.
    - PCA is applied to reduce the dimensionality of the data.

4. **K-Means Clustering:**
    - K-Means clustering is employed to group customers based on their behavior.
    - The optimal number of clusters is determined through analysis.

5. **Cluster Profiling:**
    - Key metrics and characteristics are computed for each cluster.

6. **Visualization:**
    - Various visualizations are generated to provide insights into customer segmentation.

## How to Use

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/customer-segmentation.git
   cd customer-segmentation

   pip install -r requirements.txt

   streamlit run app.py

## Usage
    - The Streamlit app allows users to interactively explore customer segmentation results.
    - Analyze the clusters and their characteristics to understand customer behavior.
    - Review the generated visualizations and insights.
    
## Features
- Data Preprocessing: Load and preprocess data, handling outliers and feature engineering.
- Customer Segmentation: Utilize clustering techniques to segment customers into distinct groups.
- Principal Component Analysis (PCA): Reduce dimensionality for better visualization.
- K-Means Clustering: Apply K-Means clustering to group customers.
- Visualizations: Present insights through various visualizations.
- Deployment: Deployed on Streamlit Cloud for easy access.
## Results
- Detailed insights into customer segmentation.
- Visualizations including radar charts, correlation matrices, and cluster profiling.
- Actionable recommendations for business enhancement based on each customer cluster.
## Deployment
- The application is deployed on Streamlit Cloud for easy access. Visit (https://customerseg.streamlit.app/)[link] to interact with the app and explore customer segmentation.

##License
This project is licensed under the MIT License.

