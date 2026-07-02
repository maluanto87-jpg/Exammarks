# 📝 Exam Marks Predictor

A web app that predicts a student's exam marks from their study habits, using a Polynomial Regression model. Built with **Streamlit** and **scikit-learn**.

🔗 **Live demo:**https://exammarks-grjavpvbwx78fwkvurzcnu.streamlit.app/

---

## 📋 Overview

This app takes three inputs — **study hours per day**, **age**, and **internet access** (yes/no) — and predicts a student's exam marks using a degree-3 polynomial regression model.

---

## 🗂️ Project Structure

```
exam-marks/
├── app.py                 # Streamlit web app (trains model at startup)
├── EXAM_MARK-DS.csv       # Training dataset
├── requirements.txt       # Python dependencies
├── runtime.txt            # Pinned Python version for deployment
└── README.md
```

---

## 🧠 Model

- **Algorithm:** Polynomial Regression (degree 3) — `PolynomialFeatures` + `LinearRegression` (scikit-learn)
- **Features used:** `hours` (study hours per day), `age`, `internet` (1 = has access, 0 = no access)
- **Target:** `marks` — the student's exam score
- **Preprocessing:** missing values in `hours` are filled with the column mean before training

| Metric | Description |
|---|---|
| MAE | Mean Absolute Error |
| MSE | Mean Squared Error |
| RMSE | Root Mean Squared Error |
| R² Score | Proportion of variance explained |

> The app trains the model fresh from `EXAM_MARK-DS.csv` every time it starts (cached via `@st.cache_resource`), so there's no pickled model file to go stale or mismatch scikit-learn versions between your machine and the deployment environment. Training metrics are computed on the training data itself and shown in an expandable panel in the app.

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```
> On Windows, if `pip` isn't recognized, use `py -m pip install -r requirements.txt` instead.

### 3. Run the app locally
```bash
streamlit run app.py
```
> On Windows: `py -m streamlit run app.py`

The app will open automatically in your browser at `http://localhost:8501`. If `EXAM_MARK-DS.csv` isn't found next to `app.py`, the app will prompt you to upload it directly in the browser.

---

## ☁️ Deploy to Streamlit Community Cloud (Free)

1. Push this repo to GitHub, with `app.py`, `EXAM_MARK-DS.csv`, `requirements.txt`, and `runtime.txt` all in the **root** folder (exact filenames — no spaces, no "(1)" suffixes; these are the most common causes of a broken deploy).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **"New app"**, select this repo and branch, and set the main file to `app.py`.
4. Click **Deploy** — Streamlit Cloud installs everything from `requirements.txt` automatically, using the Python version pinned in `runtime.txt`.
5. Copy the generated URL and share it, or add it to the top of this README.

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) — web app framework
- [scikit-learn](https://scikit-learn.org/) — model training (Polynomial Regression)
- [pandas](https://pandas.pydata.org/) / [numpy](https://numpy.org/) — data handling

---

## 📌 Notes

- Model performance metrics shown in the app are computed on the **training data**, not a held-out test set — treat them as a fit-quality check rather than a generalization estimate.
- Degree-3 polynomial features can overfit on small datasets; if predictions look unstable for values far outside the training data's range, consider lowering `DEGREE` in `app.py`.

