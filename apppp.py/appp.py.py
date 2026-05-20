import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Title
st.title("HypothesisLab AI")

# Upload CSV
uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"]
)

# Check upload
if uploaded_file is not None:

    # Read dataset
    df = pd.read_csv(uploaded_file)

    # Preview
    st.subheader("Dataset Preview")
    st.write(df.head())

    # Shape
    st.subheader("Dataset Shape")

    st.write("Rows:", df.shape[0])
    st.write("Columns:", df.shape[1])

    # Columns
    st.subheader("Column Names")

    st.write(df.columns)

    # Statistics
    st.subheader("Statistical Summary")

    st.write(df.describe())

    # Missing Values
    st.subheader("Missing Values")

    st.write(df.isnull().sum())

    # Correlation Matrix
    st.subheader("Correlation Matrix")

    corr = df.corr(numeric_only=True)

    st.write(corr)

    # Heatmap
    st.subheader("Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(10,6))

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)

    # Hypothesis Testing
    st.subheader("Hypothesis Testing - t-Test")

    # Numeric columns only
    numeric_cols = df.select_dtypes(
        include='number'
    ).columns

    # Dropdowns
    col1 = st.selectbox(
        "Select First Column",
        numeric_cols
    )

    col2 = st.selectbox(
        "Select Second Column",
        numeric_cols
    )

    # t-test
    t_stat, p_value = ttest_ind(
        df[col1],
        df[col2]
    )

    # Results
    st.write("t-statistic:", t_stat)

    st.write("p-value:", p_value)

    # Interpretation
    if p_value < 0.05:
        st.success(
            "Statistically Significant Difference Found"
        )
    else:
        st.warning(
            "No Significant Difference Found"
        )
