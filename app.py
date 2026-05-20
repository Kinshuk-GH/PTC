import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


# ---------------- Backend: data and models ----------------

import os
from sklearn.model_selection import train_test_split

@st.cache_data
def load_data():
    df = pd.read_excel(r"D:\IITK LFR\PTC\PTC_updated.xlsx")
    return df


def build_fnn(
    in_dim,
    hidden_dims=(64, 64, 64, 64, 64, 64, 64, 64, 64, 64),
    out_dim=1,):

    inputs = keras.Input(shape=(in_dim,))
    x = inputs

    for h in hidden_dims:
        x = keras.layers.Dense(h, activation="relu")(x)

    outputs = keras.layers.Dense(out_dim)(x)

    model = keras.Model(inputs, outputs)

    return model


@st.cache_resource
def load_models_and_scalers():

    # =========================================================
    # Create checkpoint folders
    # =========================================================

    os.makedirs("checkpoints/eff", exist_ok=True)
    os.makedirs("checkpoints/fopt", exist_ok=True)

    # =========================================================
    # Load dataset
    # =========================================================

    df = load_data()

    # =========================================================
    # ---------------- EFFICIENCY MODEL -----------------------
    # Inputs: Latitude, Sunshape, Focal length
    # Output: Efficiency
    # =========================================================

    X_eff = df[["Latitude", "Sunshape", "Focal length"]].values.astype(np.float32)

    y_eff = df["Efficiency"].values.astype(np.float32)

    Xeff_train, Xeff_val, yeff_train, yeff_val = train_test_split(
        X_eff,
        y_eff,
        test_size=0.2,
        random_state=42,
    )

    eff_model = build_fnn(
        in_dim=3,
        hidden_dims=(64,) * 10,
        out_dim=1,
    )

    eff_model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"],
    )

    checkpoint_callback_eff = tf.keras.callbacks.ModelCheckpoint(
        filepath="checkpoints/eff/best_eff.weights.h5",
        save_weights_only=True,
        save_best_only=True,
        monitor="val_loss",
        mode="min",
        verbose=1,
    )

    eff_model.fit(
        Xeff_train,
        yeff_train,
        validation_data=(Xeff_val, yeff_val),
        epochs=100,
        batch_size=32,
        callbacks=[checkpoint_callback_eff],
        verbose=1,
    )

    # Load best weights
    eff_model.load_weights("checkpoints/eff/best_eff.weights.h5")

    # =========================================================
    # ---------------- FOPT MODEL -----------------------------
    # Inputs: Latitude, Sunshape
    # Output: Optimal focal length
    # =========================================================

    # Create target using grouped maximum efficiency
    idx = df.groupby(["Latitude", "Sunshape"])["Efficiency"].idxmax()

    fopt_df = df.loc[idx].copy()

    X_fopt = fopt_df[["Latitude", "Sunshape"]].values.astype(np.float32)

    y_fopt = fopt_df["Focal length"].values.astype(np.float32)

    Xf_train, Xf_val, yf_train, yf_val = train_test_split(
        X_fopt,
        y_fopt,
        test_size=0.2,
        random_state=42,
    )

    fopt_model = build_fnn(
        in_dim=2,
        hidden_dims=(64,) * 10,
        out_dim=1,
    )

    fopt_model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"],
    )

    checkpoint_callback_fopt = tf.keras.callbacks.ModelCheckpoint(
        filepath="checkpoints/fopt/best_fopt.weights.h5",
        save_weights_only=True,
        save_best_only=True,
        monitor="val_loss",
        mode="min",
        verbose=1,
    )

    fopt_model.fit(
        Xf_train,
        yf_train,
        validation_data=(Xf_val, yf_val),
        epochs=100,
        batch_size=32,
        callbacks=[checkpoint_callback_fopt],
        verbose=1,
    )

    # Load best weights
    fopt_model.load_weights("checkpoints/fopt/best_fopt.weights.h5")

    return {
        "eff_model": eff_model,
        "fopt_model": fopt_model,
    }

# ---------------- Prediction helpers ----------------

def predict_fopt(latitude, sunshape, backend):
    x = np.array([[latitude, sunshape]], dtype=np.float32)
    pred = backend["fopt_model"].predict(x, verbose=0)
    return float(pred[0][0])


def predict_efficiency(latitude, sunshape, focal_length, backend):
    x = np.array([[latitude, sunshape, focal_length]], dtype=np.float32)
    pred = backend["eff_model"].predict(x, verbose=0)
    return float(pred[0][0])

# ---------------- Frontend: Streamlit UI ----------------

df = load_data()
backend = load_models_and_scalers()

st.set_page_config(page_title="Solar Focal Length Explorer", layout="wide")

st.title("Solar Focal Length Explorer (Gemma 4 Hackathon)")

st.markdown(
    """
This app uses a TensorFlow feed-forward neural network to study how **latitude** and **sunshape** 
affect the **optimal focal length** and **optical efficiency** of a parabolic trough / LFR system.
Gemma 4 is used to explain the trends in natural language.
"""
)

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("Inputs")

    lat_min = float(df["Latitude"].min())
    lat_max = float(df["Latitude"].max())
    latitude = st.slider(
        "Latitude (deg)",
        min_value=round(lat_min, 1),
        max_value=round(lat_max, 1),
        value=round((lat_min + lat_max) / 2, 1),
        step=0.1,
    )

    sunshape_values = sorted(df["Sunshape"].unique())
    sunshape = st.selectbox("Sunshape parameter", sunshape_values)

    use_fopt = st.checkbox("Use predicted optimal focal length", value=True)

    f_value = None
    if use_fopt:
        if st.button("Compute optimal focal length"):
            f_value = predict_fopt(latitude, sunshape, backend)
            st.success(f"Predicted optimal focal length: {f_value:.3f} (model units)")
    else:
        f_min = float(df["Focal length"].min())
        f_max = float(df["Focal length"].max())
        f_value = st.slider(
            "Focal length",
            min_value=round(f_min, 2),
            max_value=round(f_max, 2),
            value=round((f_min + f_max) / 2, 2),
            step=0.05,
        )

with col_right:
    st.subheader("Model Predictions and Plots")

    if f_value is not None:
        eff_at_f = predict_efficiency(latitude, sunshape, f_value, backend)
        st.metric(
            "Predicted efficiency at selected focal length",
            f"{eff_at_f:.2f} %",
        )

        f_min = df["Focal length"].min()
        f_max = df["Focal length"].max()
        f_grid = np.linspace(f_min, f_max, 60)
        eff_grid = [
            predict_efficiency(latitude, sunshape, f, backend) for f in f_grid
        ]

        fig, ax = plt.subplots()
        ax.plot(f_grid, eff_grid, label="Efficiency vs focal length")
        ax.axvline(f_value, color="red", linestyle="--", label="Selected focal length")
        ax.set_xlabel("Focal length")
        ax.set_ylabel("Efficiency (%)")
        ax.set_title(f"Efficiency curve at latitude={latitude:.2f}, sunshape={sunshape}")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

        st.markdown("### Ask Gemma to explain")
        st.write(
            "Click the button below to generate a natural-language explanation of the trends."
        )

        if st.button("Explain with Gemma 4"):
            summary = {
                "latitude": latitude,
                "sunshape": sunshape,
                "f_selected": f_value,
                "eff_at_f_selected": eff_at_f,
                "f_grid_sample": f_grid[::5].tolist(),
                "eff_grid_sample": [float(e) for e in eff_grid[::5]],
            }

            prompt = f"""
You are an expert in solar thermal optics.

I have a parabolic trough / Fresnel system. A neural network model predicts the optical efficiency
as a function of site latitude, sunshape parameter, and focal length.

For latitude {latitude:.2f} degrees and sunshape {sunshape}, the model predicts:
- Optimal or selected focal length: {f_value:.3f}
- Efficiency at this focal length: {eff_at_f:.2f} %

Here are sampled points from the efficiency vs. focal length curve for this case:
{list(zip(summary["f_grid_sample"], [round(x, 2) for x in summary["eff_grid_sample"]]))}

1. Explain qualitatively how focal length influences efficiency at this latitude and sunshape.
2. Relate this to general trends with latitude and sunshape (e.g., what might happen at lower/higher latitudes?).
3. Summarize 3–5 design rules for choosing focal length given a site's latitude and sunshape.
"""

            st.code(prompt, language="markdown")
            st.info("Replace this block with a real Gemma 4 API call and display the response here.")
    else:
        st.info("Select or compute a focal length on the left to see predictions and plots.")
