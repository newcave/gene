import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

st.title("Gene Sequence Classification with Random Forest")

# Check to use default CSV
use_default = st.checkbox("Use default dataset (gene_sequence_dataset_1000.csv)")

# Step 1: Load dataset
if use_default:
    try:
        df = pd.read_csv("gene_sequence_dataset_1000.csv")
        st.success("Loaded default dataset: gene_sequence_dataset_1000.csv")
    except FileNotFoundError:
        st.error("Default dataset not found in the current directory.")
        df = None
else:
    uploaded_file = st.file_uploader("Upload your gene sequence CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = None

# Proceed if dataframe is loaded
if df is not None:
    st.subheader("Uploaded Data Sample")
    st.write(df.head())

    # Step 2: Encode sequences and labels
    le_seq = LabelEncoder()
    X = le_seq.fit_transform(df['GeneSequence']).reshape(-1, 1)

    le_label = LabelEncoder()
    y = le_label.fit_transform(df['Label'])

    # Step 3: Split and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Step 4: Predict and report
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=le_label.classes_, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    st.subheader("Classification Report")
    st.dataframe(report_df)

    st.success("Model trained and evaluated successfully!")
