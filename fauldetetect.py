import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------
# Auto-train baseline model if not found
# ---------------------------------------------
# def train_baseline_model():
    # st.warning("Models not found — training baseline models now. This will take ~10 seconds...")

    # np.random.seed(42)
    # n = 500
    # X = np.random.randn(n, 30)
    # y = np.random.choice([0, 1, 2, 3], size=n)  # 0=Normal, 1=SLG, 2=LL, 3-Phase

    # scaler = StandardScaler().fit(X)
    # X_scaled = scaler.transform(X)

    # rf = RandomForestClassifier(n_estimators=200, random_state=42)
    # rf.fit(X_scaled, y)

    # os.makedirs("models", exist_ok=True)
    # joblib.dump(rf, "models/rf_model.pkl")
    # joblib.dump(scaler, "models/scaler.pkl")

    # # Dummy CNN for demo purpose
    # cnn = tf.keras.Sequential([
    #     tf.keras.layers.Input(shape=(512,6)),
    #     tf.keras.layers.GlobalAveragePooling1D(),
    #     tf.keras.layers.Dense(4, activation='softmax')
    # ])
    # cnn.save("models/cnn_model.h5")

    # st.success("✅ Baseline models trained and saved successfully!")
def train_baseline_model():
    st.warning("Models not found — training baseline models now. This will take ~10 seconds...")

    np.random.seed(42)
    # Create synthetic numeric data like your CSV would have
    cols = [f"Signal_{i}" for i in range(6)]
    df = pd.DataFrame(np.random.randn(500, len(cols)), columns=cols)
    
    # Extract features the same way we’ll do during inference
    sample_features = extract_features(df)
    n_features = len(sample_features)

    # Create random training data
    X = np.random.randn(500, n_features)
    y = np.random.choice([0, 1, 2, 3], size=500)

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_scaled, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(rf, "models/rf_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    # Dummy CNN
    cnn = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(512,6)),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    cnn.save("models/cnn_model.h5")

    st.success("✅ Baseline models trained and saved successfully!")


# ---------------------------------------------
# Load models (train if missing)
# ---------------------------------------------
@st.cache_resource
def load_models():
    try:
        rf_model = joblib.load("models/rf_model.pkl")
        cnn_model = tf.keras.models.load_model("models/cnn_model.h5")
        scaler = joblib.load("models/scaler.pkl")
    except FileNotFoundError:
        train_baseline_model()
        rf_model = joblib.load("models/rf_model.pkl")
        cnn_model = tf.keras.models.load_model("models/cnn_model.h5")
        scaler = joblib.load("models/scaler.pkl")
    return rf_model, cnn_model, scaler


rf_model, cnn_model, scaler = load_models()

# ---------------------------------------------
# Helper: extract simple features
# ---------------------------------------------
def extract_features(df):
    df_num = df.select_dtypes(include=[np.number]).iloc[:, :6]  # only first 6 numeric cols
    feats = []
    for col in df_num.columns:
        data = df_num[col].values
        feats += [
            np.mean(data),
            np.std(data),
            np.max(data),
            np.min(data),
            np.sqrt(np.mean(data**2)),  # RMS
        ]
    return np.array(feats)
# def extract_features(df):
    # feats = []
    # for col in df.columns:
    #     try:
    #         data = pd.to_numeric(df[col], errors='coerce').dropna().values
    #         if len(data) == 0:
    #             continue
    #         feats += [
    #             np.mean(data),
    #             np.std(data),
    #             np.max(data),
    #             np.min(data),
    #             np.sqrt(np.mean(data**2))
    #         ]
    #     except Exception:
    #         continue
    # return np.array(feats)





# ---------------------------------------------
# Streamlit UI
# ---------------------------------------------
st.set_page_config(page_title="⚡ Fault Detection ML App", layout="wide")

st.title("⚡ Machine Learning for Fault Detection in Electrical Networks")

st.markdown("""
Upload a **CSV file** containing 3-phase voltage/current signals.  
Each column should represent a signal (e.g. Ia, Ib, Ic, Va, Vb, Vc).
""")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Data:")
    st.dataframe(df.head())

    st.write("### Signal Visualization:")

    # Select columns to plot
    cols_to_plot = st.multiselect(
        "Select signals to visualize:",
        df.columns.tolist(),
        default=df.columns[:6].tolist()
    )

    if cols_to_plot:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Normalize signals for visual comparability
        for col in cols_to_plot:
            try:
                y = pd.to_numeric(df[col], errors="coerce").dropna().values
                if len(y) == 0:
                    continue
                y_norm = (y - np.mean(y)) / (np.std(y) + 1e-6)
                ax.plot(y_norm, label=col, linewidth=1.2)
            except Exception as e:
                st.warning(f"Skipped column '{col}' (non-numeric data)")


        ax.set_title("Normalized Electrical Signals Over Time", fontsize=13, weight="bold")
        ax.set_xlabel("Time (samples)", fontsize=11)
        ax.set_ylabel("Normalized Amplitude", fontsize=11)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
        ax.grid(alpha=0.3)
        st.pyplot(fig)
    else:
        st.info("Select at least one signal to visualize.")


    # Feature extraction
    features = extract_features(df)
    features_scaled = scaler.transform([features])

    # Predict with RandomForest
    rf_pred = rf_model.predict(features_scaled)[0]
    rf_probs = rf_model.predict_proba(features_scaled)[0]

    # CNN predict
    # Use only numeric columns and ensure correct dtype
    df_num = df.select_dtypes(include=[np.number]).astype(np.float32)

    # Trim or pad to expected CNN shape (512, 6)
    seq = df_num.values
    if seq.shape[0] > 512:
        seq = seq[:512, :6]  # truncate
    else:
        # pad if smaller
        pad_rows = 512 - seq.shape[0]
        seq = np.pad(seq, ((0, pad_rows), (0, max(0, 6 - seq.shape[1]))), mode='constant')

    seq = seq[:, :6]  # make sure we have 6 features
    seq = np.expand_dims(seq, axis=0)  # (1, 512, 6)

    # Predict
    cnn_pred = np.argmax(cnn_model.predict(seq), axis=1)[0]
    cnn_probs = cnn_model.predict(seq)[0]

    fault_classes = ["Normal", "SLG Fault", "LL Fault", "3-Phase Fault"]

    st.markdown("### 🧠 Model Results:")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Random Forest Model")
        st.metric(label="Detected Fault", value=fault_classes[rf_pred])
        st.bar_chart(pd.DataFrame(rf_probs, index=fault_classes, columns=["Probability"]))

    with col2:
        st.subheader("1D CNN Model")
        st.metric(label="Detected Fault", value=fault_classes[cnn_pred])
        st.bar_chart(pd.DataFrame(cnn_probs, index=fault_classes, columns=["Probability"]))
else:
    st.info("Please upload a CSV file to begin.")
