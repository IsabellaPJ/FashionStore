import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def app():
    st.set_page_config(page_title="Customer Classification", page_icon=":guardsman:", layout="wide")

    st.title("Customer Classification")

    st.sidebar.header("Select Dataset")
    option = st.sidebar.selectbox("", ["Online Retail (UCI)", "C2C Store"])

    if option == "Online Retail (UCI)":
        df = pd.read_excel("Online Retail.xlsx")
    else:
        df = pd.read_csv("online store.csv", encoding="ISO-8859-1")

    st.sidebar.subheader("Dataset Preview")
    st.sidebar.write(df.head())

    st.sidebar.subheader("Data Summary")
    st.sidebar.write(df.describe())

    st.sidebar.subheader("Data Information")
    st.sidebar.write(df.info())

    st.sidebar.subheader("Data Cleaning")

    st.sidebar.write("Remove Missing Values")
    df.dropna(inplace=True)

    st.sidebar.write("Remove Duplicates")
    df.drop_duplicates(inplace=True)

    st.sidebar.write("Remove Cancelled Orders")
    df = df[df["Quantity"] > 0]

    st.sidebar.write("Data Summary After Cleaning")
    st.sidebar.write(df.describe())

    st.sidebar.subheader("Data Visualization")

    st.sidebar.write("Visualize the Distribution of Customers by Country")
    country_data = df.groupby("Country")["CustomerID"].nunique().reset_index()
    fig = plt.figure(figsize=(12, 6))
    sns.barplot(x="Country", y="CustomerID", data=country_data)
    plt.xticks(rotation=90)
    st.sidebar.pyplot(fig)

    st.sidebar.write("Visualize the Distribution of Orders by Month")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Month"] = df["InvoiceDate"].dt.strftime("%Y-%m")
    orders_data = df.groupby("Month")["Quantity"].sum().reset_index()
    fig = plt.figure(figsize=(12, 6))
    sns.lineplot(x="Month", y="Quantity", data=orders_data)
    plt.xticks(rotation=90)
    st.sidebar.pyplot(fig)

    st.sidebar.write("Visualize the Distribution of Orders by Hour")
    df["Hour"] = df["InvoiceDate"].dt.hour
    orders_data = df.groupby("Hour")["Quantity"].sum().reset_index()
    fig = plt.figure(figsize=(12, 6))
    sns.lineplot(x="Hour", y="Quantity", data=orders_data)
    plt.xticks(rotation=90)
    st.sidebar.pyplot(fig)

    st.sidebar.subheader("Customer Segmentation")

    #st.sidebar.write("Select the Number of Clusters")
    k = st.sidebar.slider("Select number of clusters", 2, 10)

    #st.sidebar.write("Select the Features for Clustering")
    feature_cols = st.sidebar.multiselect("Select the features for clustering", ["Quantity", "UnitPrice"])

    st.sidebar.write("Standardize the Features")
    X = df[feature_cols]
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

	#Perform K-Means Clustering"
    kmeans = KMeans(n_clusters=k) 
    kmeans.fit(X_std)
    df["Cluster"] = kmeans.labels_
    
    st.sidebar.write("Evaluate the Clustering Results")
    silhouette_avg = silhouette_score(X_std, kmeans.labels_)
    st.sidebar.write(f"Silhouette Score: {silhouette_avg}")
    
    st.write("## Results")
    st.write("### Customer Segments")
    fig = plt.figure(figsize=(12, 6))
    sns.scatterplot(x=feature_cols[0], y=feature_cols[1], hue="Cluster", data=df, palette="Set1")
    st.pyplot(fig)
    
    st.write("### Cluster Details")
    cluster_details = df.groupby("Cluster")[feature_cols].mean().reset_index()
    st.write(cluster_details)


if __name__ == "__main__":
    app()