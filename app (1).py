import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Exam Marks Predictor", page_icon="📝", layout="centered")

DATA_FILENAME = "EXAM_MARK-DS.csv"  # place this file next to app.py when deploying
DEGREE = 3  # matches the final model trained in the source notebook


@st.cache_resource(show_spinner="Training model...")
def train_model(df: pd.DataFrame):
    df = df.copy()
    hours_mean = df["hours"].mean()
    df["hours"] = df["hours"].fillna(hours_mean)

    X = df[["hours", "age", "internet"]]
    y = df["marks"]

    poly = PolynomialFeatures(degree=DEGREE)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)

    y_pred = model.predict(X_poly)
    metrics = {
        "MAE": mean_absolute_error(y, y_pred),
        "MSE": mean_squared_error(y, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y, y_pred)),
        "R2": r2_score(y, y_pred),
    }

    return model, poly, hours_mean, metrics


def load_data():
    try:
        return pd.read_csv(DATA_FILENAME)
    except FileNotFoundError:
        st.warning(
            f"Couldn't find `{DATA_FILENAME}` next to app.py. "
            "Upload the training CSV below to continue."
        )
        uploaded = st.file_uploader("Upload EXAM_MARK-DS.csv", type="csv")
        if uploaded is not None:
            return pd.read_csv(uploaded)
        st.stop()


st.title("📝 Exam Marks Predictor")
st.write(
    "Predicts a student's exam **marks** from study **hours**, **age**, "
    "and whether they have **internet access**, using a degree-3 polynomial "
    "regression model."
)

df = load_data()
model, poly, hours_mean, metrics = train_model(df)

with st.expander("Model performance (on training data)"):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE", f"{metrics['MAE']:.2f}")
    c2.metric("RMSE", f"{metrics['RMSE']:.2f}")
    c3.metric("MSE", f"{metrics['MSE']:.2f}")
    c4.metric("R² Score", f"{metrics['R2']:.3f}")

st.divider()
st.subheader("Enter student details")

col1, col2 = st.columns(2)
with col1:
    hours = st.number_input(
        "Study hours per day",
        min_value=0.0,
        max_value=24.0,
        value=float(round(hours_mean, 2)),
        step=0.25,
    )
    age = st.number_input("Age", min_value=5, max_value=100, value=17, step=1)
with col2:
    internet = st.radio("Internet access?", options=["Yes", "No"], horizontal=True)

internet_val = 1 if internet == "Yes" else 0

if st.button("Predict Marks", type="primary"):
    new_data = pd.DataFrame({"hours": [hours], "age": [age], "internet": [internet_val]})
    new_data_poly = poly.transform(new_data)
    prediction = model.predict(new_data_poly)[0]

    st.success(f"### Predicted Marks: {prediction:.2f}")

st.caption(
    "Model: Polynomial Regression (degree 3) on features [hours, age, internet]."
)
