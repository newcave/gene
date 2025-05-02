import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Gene Classifier", layout="wide")
st.title("🧬 Gene Sequence Classification with Random Forest")

# Sidebar for model parameters
st.sidebar.header("Random Forest Settings")
n_estimators = st.sidebar.slider("Number of Trees (n_estimators)", 10, 500, 100, step=10)
max_depth = st.sidebar.slider("Maximum Depth (max_depth)", 1, 50, 10)
min_samples_split = st.sidebar.slider("Min Samples Split", 2, 20, 2)
random_state = st.sidebar.number_input("Random State", value=42)
test_size = st.sidebar.slider("Test Set Size", 0.1, 0.5, 0.2, step=0.05)

# Checkbox to use default dataset
use_default = st.checkbox("Use default dataset (gene_sequence_dataset_1000.csv)")

# Load data
if use_default:
    try:
        df = pd.read_csv("gene_sequence_dataset_1000.csv")
        st.success("Loaded default dataset.")
    except FileNotFoundError:
        st.error("Default dataset not found.")
        df = None
else:
    uploaded_file = st.file_uploader("Upload your gene sequence CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = None

# Process
if df is not None:
    st.subheader("📄 Uploaded Data Sample")
    st.write(df.head())

    # Encoding
    le_seq = LabelEncoder()
    X = le_seq.fit_transform(df['GeneSequence']).reshape(-1, 1)

    le_label = LabelEncoder()
    y = le_label.fit_transform(df['Label'])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Model
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state
    )
    clf.fit(X_train, y_train)

    # Predict
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    # Accuracy
    acc_train = accuracy_score(y_train, y_train_pred)
    acc_test = accuracy_score(y_test, y_test_pred)

    # Report
    report = classification_report(y_test, y_test_pred, target_names=le_label.classes_, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    st.subheader("📊 Classification Report")
    st.dataframe(report_df)

    # Plot Accuracy Comparison
    st.subheader("🎯 Accuracy Comparison")

    fig, ax = plt.subplots()
    ax.bar(["Train Accuracy", "Test Accuracy"], [acc_train, acc_test], color=["skyblue", "salmon"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Accuracy on Train vs Test Set")
    st.pyplot(fig)

    st.success("✅ Model trained and evaluated successfully!")
