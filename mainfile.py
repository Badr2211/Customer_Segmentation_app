
import warnings
import seaborn as sns
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from matplotlib.ticker import FixedLocator, FixedFormatter
import scipy.cluster.hierarchy as sch
from matplotlib import pyplot

warnings.filterwarnings("ignore")


def corrplot():
    sns.set_style('whitegrid')
    numeric_columns = customer_behavior.select_dtypes(exclude=['object']).columns
    corr=customer_behavior[numeric_columns].corr()
    fig, ax = plt.subplots()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask, k=1)] = True
    sns.heatmap(corr,mask=mask,annot=True,center=0, fmt='.2f', linewidths=2,ax=ax)
    plt.title('Correlation Matrix', fontsize=3)
    return fig

def uniq (df):
    df=pd.DataFrame([[i,df[i].unique(),df[i].dtype,len(df[i].unique())]for i in df.columns],columns=['feature','val','types','len']).set_index('feature')
    return df
def load_data (path):
    
    if path ==None :
        path ='Online Retail.xlsx'
        extention =path.split('.')[-1]
    else :
        extention =path.name.split('.')[-1]
    if extention == 'csv':
       data = pd. read_csv(path)
    elif extention in ("xls", "xlsx"):
        data = pd. read_excel(path,engine='openpyxl')
    elif extention == 'json' :
       data = json.load(path)
    elif extention == 'txt' :
       with open(file_path, 'r') as file:
            data = file.read()
    elif extention == 'db':
        conn = sqlite3.connect(file_path)
        query = "SELECT * FROM ;"
        data = pd.read_sql(query, conn)
        conn.close()
    else:
        raise ValueError("Unsupported file extension")
    return data

def load():
    df = load_data('Online Retail.xlsx')
    return df

"""- Load the data"""

df = load()


# remove israeal from data
df['Country'] = df['Country'].replace({'Israel': 'Palestine'})
df['Country'].isin(['Israel']).sum()


"""CustomerID' is crucial for my analysis, and rows with null 'CustomerID' values cannot be imputed or filled in any meaningful way"""



# Identify canceled transactions
cancel = df['InvoiceNo'].str.contains('c', case=False, na=False)
cancel_transaction = df[cancel]
df['Order_Cancellation'] = [1 if i else 0 for i in cancel]

# Group by CustomerID and find the most frequently purchased item
customer_behavior = df.groupby('CustomerID', as_index=False)['Description'].agg(lambda x: x.mode().iloc[0])

# Calculate total spending excluding canceled transactions
df['Total_spending'] = np.where(cancel == False, df['UnitPrice'] * df['Quantity'], 0)

# Calculate Total Spending per Customer
customer_behavior['Total_spending'] = df.groupby(['CustomerID'])['Total_spending'].sum().values

# Filter quantity for successful transactions
df['Quantity'] = np.where(cancel == False, df['Quantity'], 0)

# Calculate total products purchased per customer
customer_behavior['total_products_purchased'] = df.groupby(['CustomerID'])['Quantity'].sum().values
customer_behavior['N.Order_Cancellation'] = df.groupby('CustomerID')['Order_Cancellation'].sum().values

# Identify successful transactions
df['sucsess_transaction'] = np.where(cancel == False, 1, 0)

# Calculate the number of successful transactions per customer
customer_behavior['N.transaction'] = df.groupby('CustomerID')['sucsess_transaction'].sum().values

# Convert InvoiceDate to datetime and extract only the date
df['Day'] = df['InvoiceDate'].dt.date

# Find the most recent purchase date for each customer
customer_behavior['Day'] = df.groupby('CustomerID')['Day'].max().values

# Find the most recent date in the entire dataset
most_recent_date = df['Day'].max()

# Convert InvoiceDay to datetime type before subtraction
customer_behavior['Day'] = pd.to_datetime(customer_behavior['Day'])
most_recent_date = pd.to_datetime(most_recent_date)

# Calculate the number of days since the last purchase for each customer
customer_behavior['Days_Since_Last_Purchase'] = (most_recent_date - customer_behavior['Day']).dt.days

# Remove the InvoiceDay column
customer_behavior.drop(columns=['Day'], inplace=True)

# Extract year and month from InvoiceDate
df['Year'] = df['InvoiceDate'].dt.year
df['Month'] = df['InvoiceDate'].dt.month

# Find the most common month of purchase for each customer
customer_behavior['Month'] = df.groupby('CustomerID')['Month'].agg(lambda x: x.mode().iloc[0]).values

# Create columns for start and end dates of customer transactions
customer_behavior['start_date'] = df.groupby(['CustomerID'])['InvoiceDate'].min().values
customer_behavior['end_date'] = df.groupby(['CustomerID'])['InvoiceDate'].max().values

# Calculate the duration of customer transactions
customer_behavior['N.day_as_customer'] = (customer_behavior['end_date'] - customer_behavior['start_date']).dt.days

# Drop start_date and end_date columns
customer_behavior.drop(columns=['start_date', 'end_date'], inplace=True)

# Capture the country associated with each customer
customer_behavior['Country'] = df.groupby(['CustomerID'])['Country'].min().values
customer_behavior['UK'] = [1 if country == 'United Kingdom' else 0 for country in customer_behavior['Country']]
customer_behavior.drop(columns=['Country'], inplace=True)

# Calculate the average basket size for each customer
customer_behavior['Average Basket Size'] = df.groupby(['CustomerID'])['Quantity'].mean().values.astype(int)

# Calculate the average transaction value per customer
customer_behavior['Average_Transaction_Value'] = customer_behavior['Total_spending'] / (customer_behavior['N.transaction'] + 1)

# Calculate the number of unique products purchased by each customer
customer_behavior['unique_products'] = df.groupby('CustomerID')['StockCode'].nunique().values

# Calculate the frequency of shopping on specific days of the week
customer_behavior['shopping_day'] = df.groupby(['CustomerID', df['InvoiceDate'].dt.dayofweek])['InvoiceNo'].nunique().unstack(fill_value=0).idxmax(axis=1).values

# Identify customers with abandoned carts
abandoned_cart_customers = df[df['Description'].isnull()]['CustomerID'].unique()
customer_behavior['cart_abandonment_rate'] = customer_behavior.index.isin(abandoned_cart_customers).astype(int)

"""## Outliers
"""

desc=customer_behavior.describe()

model = IsolationForest(contamination=0.05, random_state=0)

# Fitting the model on our dataset (converting DataFrame to NumPy to avoid warning)
customer_behavior['Outlier_Scores'] = model.fit_predict(customer_behavior.iloc[:, 2:].to_numpy())

# Creating a new column to identify outliers (1 for inliers and -1 for outliers)
customer_behavior['Is_Outlier'] = [1 if x == -1 else 0 for x in customer_behavior['Outlier_Scores']]

outlier_percentage = customer_behavior['Is_Outlier'].value_counts(normalize=True) * 100

#Remove the outliers from the main dataset
customer_behavior = customer_behavior[customer_behavior['Is_Outlier'] == 0]

# Drop the 'Outlier_Scores' and 'Is_Outlier' columns
customer_behavior = customer_behavior.drop(columns=['Outlier_Scores', 'Is_Outlier'])


after=customer_behavior.describe()
after.loc['max':,:]

"""#EDA for our new feature
***
"""

unique=uniq(customer_behavior)


continuous_f=customer_behavior[list(unique[unique['types']!='object'].index
)]
shopping_day_counts = customer_behavior['shopping_day'].value_counts()

# Set custom x-labels
custom_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',  'Friday', 'Sunday']
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
"""
#PCA

"""

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()
customer_behavior_scaled = scaler.fit_transform(customer_behavior.iloc[:,2:])
customer_behavior_scaled = pd.DataFrame(customer_behavior_scaled, columns=[f'{c}' for c in customer_behavior.columns[2:]])

sns.set_style('whitegrid')
corr=customer_behavior_scaled.corr()

mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask, k=1)] = True

# Apply PCA
pca = PCA().fit(customer_behavior_scaled)

# Calculate the Cumulative Sum of the Explained Variance
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

# Set the optimal k value (based on our analysis, we can choose 6)
optimal_k = 5
pca = PCA(n_components=optimal_k)#n_components=3
pca.fit(customer_behavior_scaled)

Xhat = pca.transform(customer_behavior_scaled)

customer_behavior_PCA = pd.DataFrame(columns=[f'Projection on Component {i+1}' for i in range(optimal_k)], data=Xhat)
customer_behavior_PCA.index= customer_behavior.index
customer_behavior_PCA

# Create the PCA component DataFrame
pc_df = pd.DataFrame(pca.components_.T, columns=['PC{}'.format(i+1) for i in range(pca.n_components_)],
                     index=customer_behavior_scaled.columns)


#4-Determining Optimal Number of Clusters


from sklearn.cluster import KMeans# Create and fit a range of models
km_list = list()

for clust in range(1,21):
    km = KMeans(n_clusters=clust, random_state=42)
    km = km.fit(customer_behavior_PCA)

    km_list.append(pd.Series({'clusters': clust,
                              'inertia': km.inertia_,
                              'model': km}))

plot_data = (pd.concat(km_list, axis=1)
             .T
             [['clusters','inertia']]
             .set_index('clusters'))


"""***
Based in our analysis i see ```K=4``` is the optimal value for $K$   
***
"""

optimal_k=4
# Initialize and fit the K-means model
km = KMeans(n_clusters=optimal_k,init='k-means++', random_state=42)

cluster_labels = km.fit_predict(customer_behavior_PCA)

# Add the cluster labels as a new column in the DataFrame
customer_behavior['Cluster_Labels'] = cluster_labels

# Now, original_df contains the cluster labels as a new column

"""***
# 6-Cluster Profiling
"""

clusters_info = customer_behavior.groupby('Cluster_Labels', as_index=False).sum()
clusters_info['N.customer']=customer_behavior['Cluster_Labels'].value_counts().values

clusters_info['sum of cancellation'] =customer_behavior.groupby('Cluster_Labels')['N.Order_Cancellation'].sum().values
cluster_profiling =customer_behavior.groupby('Cluster_Labels', as_index=False).sum()[['Total_spending']]
cluster_profiling ['N.customer']=customer_behavior['Cluster_Labels'].value_counts().values


cluster_profiling['AVG_C_spending'] =cluster_profiling['Total_spending']/cluster_profiling['N.customer']


cluster_profiling ['AVG_C_cancellation']= clusters_info['sum of cancellation']/cluster_profiling['N.customer']

cluster_profiling ['AVG_C_prouduct']=clusters_info['total_products_purchased']/cluster_profiling['N.customer']

cluster_profiling ['AVG_C_prouduct']=clusters_info['total_products_purchased']/cluster_profiling['N.customer']

cluster_profiling ['N.day_as_customer(mode)'] = customer_behavior.groupby('Cluster_Labels')['N.day_as_customer'].agg(lambda x: x.mode().iloc[0]).values


cluster_profiling ['N.day_as_customer(AVG_C)'] = (clusters_info['N.day_as_customer']/cluster_profiling['N.customer']).values.astype(int)

cluster_profiling ['prefer day'] = customer_behavior.groupby('Cluster_Labels')['shopping_day'].agg(lambda x: x.mode().iloc[0]).values


cluster_profiling ['prefer month'] = customer_behavior.groupby('Cluster_Labels')['Month'].agg(lambda x: x.mode().iloc[0]).values


cluster_profiling ['Days_Since_Last_Purchase(mode)'] = customer_behavior.groupby('Cluster_Labels')['Days_Since_Last_Purchase'].agg(lambda x: x.mode().iloc[0]).values


cluster_profiling ['unique_products(mode)']= customer_behavior.groupby('Cluster_Labels')['unique_products'].agg(lambda x: x.mode().iloc[0]).values

cluster_profiling.insert(0, 'Cluster_Labels', [0,1,2,3])

cluster_profiling

"""
#7-Visualization
"""
continuous_f=list(unique[unique['types']=='float'].index)
discrete_f=list(unique[unique['types']!='float'].index)
customer_behavior_scaled['Cluster_Labels']=customer_behavior['Cluster_Labels'].values
def radar_chart():
    colors  = ['#e8000b', '#023eff','#1ac938', '#ffaa00']# '#ff55aa', '#0088ff', '#44cc00']

    unique=uniq(customer_behavior_scaled)
    unique
    continuous_f=list(unique[unique['types']=='float'].index)
    discrete_f=list(unique[unique['types']!='float'].index)

    continuous_result = customer_behavior_scaled.groupby('Cluster_Labels', as_index=False).median()[continuous_f]

    # Grouping for discrete features
    discrete_result = customer_behavior_scaled.groupby('Cluster_Labels', as_index=False).agg(lambda x: x.mode().iloc[0])[discrete_f]

    # Merge the two dataframes on the 'Cluster_Labels'

    clusters_info = pd.concat([continuous_result, discrete_result], axis=1, sort=False)
    clusters_info
    cluster_centroids = clusters_info#customer_behavior_scaled.groupby('Cluster_Labels').median()
    # Function to create a radar chart

    num_clusters=km.n_clusters
    def create_radar_chart(ax, angles, data, color, cluster):
        # Plot the data and fill the area
        ax.fill(angles, data, color=color, alpha=0.4)
        ax.plot(angles, data, color=color, linewidth=5, linestyle='solid')

        # Add a title
        ax.set_title(f'Cluster {cluster}', size=60, color=color, y=1.1)

    # Set data
    labels=np.array(cluster_centroids.columns)
    num_vars = len(labels)

    # Compute angle of each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # The plot is circular, so we need to "complete the loop" and append the start to the end
    labels = np.concatenate((labels, [labels[0]]))
    angles += angles[:1]

    # Initialize the figure
    fig, ax = plt.subplots(figsize=(40, 20), subplot_kw=dict(polar=True), nrows=1, ncols=4)



    # Create radar chart for each cluster
    colors  = ['#e8000b', '#023eff','#1ac938', '#ffaa00']# '#ff55aa', '#0088ff', '#44cc00']

    #['#e8000b', '#023eff','#1ac938', '#023eff']
    for i, color in enumerate(colors[:num_clusters]):
        data = cluster_centroids.loc[i].tolist()
        data += data[:1]  # Complete the loop
        create_radar_chart(ax[i], angles, data, color, i)

    # Add input data

    for i in range(num_clusters):
        ax[i].set_xticks(angles[:-1])
        ax[i].set_xticklabels(labels[:-1], size=25)

        # Add a grid
        ax[i].grid(color='grey', linewidth=0.5)

    # Display the plot
    plt.tight_layout()

    return fig
