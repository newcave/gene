import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

st.title("🧬 Gene Sequence Classification with Random Forest")

# Sidebar for model parameters
st.sidebar.header("Random Forest Settings")
n_estimators = st.sidebar.slider("Number of Trees (n_estimators)", 10, 500, 100, step=10)
max_depth = st.sidebar.slider("Maximum Depth (max_depth)", 1, 50, 10)
min_samples_split = st.sidebar.slider("Min Samples Split", 2, 20, 2)
random_state = st.sidebar.number_input("Random State", value=42)

# Checkbox to use default dataset
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

# Step 2: Process and Train
if df is not None:
    st.subheader("📄 Uploaded Data Sample")
    st.write(df.head())

    # Encode sequences and labels
    le_seq = LabelEncoder()
    X = le_seq.fit_transform(df['GeneSequence']).reshape(-1, 1)

    le_label = LabelEncoder()
    y = le_label.fit_transform(df['Label'])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # Train model with user parameters
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state
    )
    clf.fit(X_train, y_train)

    # Evaluate model
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=le_label.classes_, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    st.subheader("📊 Classification Report")
    st.dataframe(report_df)

    st.success("✅ Model trained and evaluated successfully!")
