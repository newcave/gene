import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# App config
st.set_page_config(page_title="Gene Anomaly Classifier", layout="wide")
st.title("🧬 Gene Sequence Binary Classification (Normal vs Abnormal)")

# Sidebar - model hyperparameters
st.sidebar.header("Random Forest Parameters")
n_estimators = st.sidebar.slider("Number of Trees (n_estimators)", 10, 500, 100, step=10)
max_depth = st.sidebar.slider("Maximum Depth (max_depth)", 1, 50, 10)
min_samples_split = st.sidebar.slider("Min Samples Split", 2, 20, 2)
random_state = st.sidebar.number_input("Random State", value=42)
test_size = st.sidebar.slider("Test Set Size", 0.1, 0.5, 0.2, step=0.05)

# Option: use default or upload
use_default = st.checkbox("Use default dataset (gene_sequence_dataset_1000_modified.csv)")

# Load dataset
df = None
if use_default:
    try:
        df = pd.read_csv("gene_sequence_dataset_1000.csv")
        st.success("Default dataset loaded successfully.")
    except FileNotFoundError:
        st.error("Default CSV file not found in app directory.")
else:
    uploaded_file = st.file_uploader("Upload your gene sequence CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

# Proceed if data is available
if df is not None:
    st.subheader("📄 Raw Data Sample")
    st.write(df.head())

    # Filter only Normal/Abnormal and create binary label
    df = df[df['Label'].isin(['Normal', 'Abnormal'])].copy()
    df['BinaryLabel'] = df['Label'].apply(lambda x: 1 if x == 'Abnormal' else 0)

    # Encode gene sequences
    le_seq = LabelEncoder()
    X = le_seq.fit_transform(df['GeneSequence']).reshape(-1, 1)
    y = df['BinaryLabel']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Train classifier
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state
    )
    clf.fit(X_train, y_train)

    # Predictions
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    # Accuracy
    acc_train = accuracy_score(y_train, y_train_pred)
    acc_test = accuracy_score(y_test, y_test_pred)

    # Classification report
    report = classification_report(
        y_test, y_test_pred, target_names=["Normal", "Abnormal"], output_dict=True
    )
    report_df = pd.DataFrame(report).transpose()

    st.subheader("📊 Classification Report (Binary)")
    st.dataframe(report_df)

    # Accuracy bar plot
    st.subheader("🎯 Accuracy Comparison")

    fig, ax = plt.subplots()
    ax.bar(["Train Accuracy", "Test Accuracy"], [acc_train, acc_test], color=["skyblue", "salmon"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Random Forest Accuracy")
    st.pyplot(fig)

    st.success("✅ Model trained and evaluated successfully!")
