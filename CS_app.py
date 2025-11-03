import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------
# Load Model and Scaler
# ---------------------------------------
with open("kmeans_model1.pkl", "rb") as model_file:
    kmeans = pickle.load(model_file)

with open("scaler1.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# ---------------------------------------
# Streamlit Page Styling
# ---------------------------------------

st.set_page_config(
    page_title="Customer Segmentation App",
    page_icon="💡",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
        /* Background color */
        .stApp {
            background-color: #C1E1C1;
            font-family: "Segoe UI", sans-serif;
        }

        /* Title style */
        h1 {
            color: #2C3E50;
            text-align: center;
            font-size: 2.2rem;
            margin-bottom: 0.5em;
        }

        /* Subheaders */
        h2, h3, h4 {
            color: #34495E;
            margin-top: 1.5em;
        }

        /* Upload button */
        .stFileUploader label {
            background-color: #3498DB;
            color: white !important;
            padding: 0.6em 1.2em;
            border-radius: 8px;
            text-align: center;
            font-weight: 600;
        }

        /* Buttons */
        div.stDownloadButton > button {
            background-color: #2ecc71;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 0.5em 1.2em;
            border: none;
        }

        div.stDownloadButton > button:hover {
            background-color: #27ae60;
            color: white;
        }

        /* Info boxes */
        .stAlert {
            border-radius: 8px;
        }

        /* Charts */
        .stPlotlyChart, .stPyplot {
            background-color: #ffffff;
            padding: 15px;
            border-radius: 12px;
            box-shadow: 0px 2px 6px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)



# ---------------------------------------
# Streamlit Page Setup
# ---------------------------------------
st.title("💡 Customer Segmentation App")
st.write("Upload your CSV file and get customer clusters with visualization.")

# ---------------------------------------
# File Upload
# ---------------------------------------
uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Data Preview")
    st.write(df.head())

    required_cols = ["Annual Income (k$)", "Spending Score (1-100)"]
    if not all(col in df.columns for col in required_cols):
        st.error(f"CSV must contain columns: {required_cols}")
    else:
        X = df[required_cols]
        X_scaled = scaler.transform(X)

        # Predict clusters using pre-trained model
        df['Cluster Label'] = kmeans.predict(X_scaled)


        # ---------------------------------------
        # Scatter Plot Visualization
        # ---------------------------------------
        st.subheader("🌷 Customer Segments Scatter Plot")

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            data=df,
            x="Annual Income (k$)",
            y="Spending Score (1-100)",
            hue="Cluster Label",
            palette="Set2",
            s=90,
            edgecolor="black",
            ax=ax
        )

        # Mark cluster centers (rescale back to original)
        centers = scaler.inverse_transform(kmeans.cluster_centers_)
        ax.scatter(
            centers[:, 0],
            centers[:, 1],
            c='black',
            s=200,
            marker='X',
            label='Centroids'
        )

        ax.set_title("Customer Segments by Income and Spending Score", fontsize=14)
        ax.legend()
        st.pyplot(fig)

        # Display results
        st.subheader("📁 Clustered Data with Labels")
        st.write(df)

        # Download CSV
        csv_download = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Labeled CSV",
            data=csv_download,
            file_name='clustered_output.csv',
            mime='text/csv'
        )

else:
    st.info("👆 Upload a CSV file to get started.")

