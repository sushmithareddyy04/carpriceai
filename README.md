# 🚗 CarPriceAI – Advanced Used Car Price Prediction System

**CarPriceAI** is an industrial-grade, end-to-end Machine Learning web application and automotive valuation system built with Python, Scikit-Learn, and Streamlit. It solves used car pricing opacity and information asymmetry by predicting vehicle resale values using vehicle specifications, age, mileage, fuel type, transmission, and brand prestige tiers.

Developed by **Sushmitha Reddy M (Joy University)** according to an industrial data science curriculum spanning 10 structured modules.

---

## 🏆 Project Highlights & Benchmark Results

- **Champion Model:** Blended Ensemble (Random Forest + Gradient Boosting Regressor)
- **Validation Accuracy:** **94.8% $R^2$ Score**
- **Mean Absolute Error (MAE):** **₹ 96,280**
- **Root Mean Squared Error (RMSE):** **₹ 182,590**
- **Inference Latency:** Sub-50ms real-time valuation

| Algorithm | MAE (INR) | RMSE (INR) | $R^2$ Score | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Blended Ensemble (RF + GB)** | **₹ 96,280** | **₹ 182,590** | **0.9480** | 🏆 **Champion** |
| Random Forest Regressor | ₹ 97,662 | ₹ 184,287 | 0.9470 | Ensemble Base |
| Gradient Boosting Regressor | ₹ 99,702 | ₹ 188,795 | 0.9444 | Ensemble Base |
| Decision Tree Regressor | ₹ 139,760 | ₹ 248,087 | 0.9039 | Baseline |
| Multiple Linear Regression | ₹ 242,757 | ₹ 401,963 | 0.7478 | Baseline |

---

## 🔬 10 Industrial ML Curriculum Modules

1. **Module 1 (Python for ML):** Clean, modular OOP architecture with immutable dataclasses (`CarDataConfig`), validators (`DataValidator`), functional pipelines, and structured logging.
2. **Module 2 (Statistics & Probability):** Central tendency, dispersion, Fisher-Pearson skewness, kurtosis, and Interquartile Range (IQR) outlier detection for price and kilometers driven.
3. **Module 3 (Data Wrangling):** Unit extraction regex (`1248 CC` -> `1248.0`, `18.9 kmpl` -> `18.9`, `82 bhp` -> `82.0`), duplicate dropping, and brand-grouped median imputation.
4. **Module 4 (EDA & Visualization):** Statistical correlation matrices, price vs. year depreciation curves, and fuel/transmission distributions.
5. **Module 5 (Supervised Learning):** Training 4 regression models: Multiple Linear Regression, Decision Tree Regressor, Random Forest Regressor, and Gradient Boosting Regressor.
6. **Module 6 (Unsupervised Learning / Feature Clustering):** K-Means clustering ($k=3$) segmenting vehicles into **Economy Commuter**, **Mid-Range Executive**, and **Luxury Premium** tiers.
7. **Module 7 (Model Evaluation):** Comprehensive benchmarking using MAE, MSE, RMSE, MAPE, and $R^2$ score with automated champion model selection.
8. **Module 8 (Feature Engineering):** Brand prestige tiering, vehicle age calculation ($2026 - \text{year}$), One-Hot/Ordinal categorical encoding, and `StandardScaler` feature scaling.
9. **Module 9 (Ensemble Methods):** Hyperparameter tuning and optimal model blending combining Random Forest (70%) and Gradient Boosting (30%).
10. **Module 10 (ML Projects & Deployment):** Serializing the trained champion model and preprocessor using `pickle` (`.pkl`) and developing a responsive Streamlit web application.

---

## 📁 Repository Structure

```
CarPriceAI/
├── data/
│   ├── generate_dataset.py          # Script generating authentic automotive dataset
│   └── car_details_v3.csv           # Cleaned dataset (4,200+ vehicle records)
├── models/
│   ├── car_price_champion.pkl       # Serialized champion ensemble model
│   ├── preprocessor.pkl             # Serialized StandardScaler & feature mappings
│   └── model_metadata.json          # Metrics, leaderboard, and training metadata
├── src/
│   ├── __init__.py                  # Package initialization
│   ├── module1_python_ml.py         # Config dataclass, validator, OOP abstractions
│   ├── module2_statistics.py        # Central tendency, dispersion, IQR outlier bounds
│   ├── module3_wrangling.py         # Regex unit stripping, imputation, deduplication
│   ├── module4_eda.py               # Correlation matrix, depreciation analytics
│   ├── module5_supervised.py        # Linear, Decision Tree, RF, Gradient Boosting
│   ├── module6_clustering.py        # K-Means clustering for market segmentation
│   ├── module7_evaluation.py        # MAE, MSE, RMSE, R2, MAPE, champion selector
│   ├── module8_feature_eng.py       # Brand tiers, age, encoding, StandardScaler
│   ├── module9_ensemble_tuning.py   # Hyperparameter tuning, Blended Ensemble
│   ├── module10_pipeline_serialization.py # Pickle serialization & loading
│   └── inference_engine.py          # Production valuation inference engine
├── app.py                           # Full interactive Streamlit dashboard
├── train_pipeline.py                # Master training pipeline executing Modules 1-10
├── requirements.txt                 # Project dependencies
└── README.md                        # Documentation
```

---

## 🚀 Deployment Guide

### Option 1: Streamlit Community Cloud (Recommended & Free)
1. Push this repository to GitHub:
   ```bash
   git remote add origin https://github.com/<YOUR_USERNAME>/carpriceai.git
   git push -u origin main
   ```
2. Visit [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
3. Click **"New App"** -> Choose your repository -> Main branch: `main` -> Main file path: `app.py`.
4. Click **Deploy!** Your app will be live with a permanent custom URL.

### Option 2: Docker Container (Cloud Run / AWS / Hugging Face)
A multi-stage `Dockerfile` is included:
```bash
docker build -t carpriceai .
docker run -p 8501:8501 carpriceai
```

### Option 3: Render / Railway / Heroku
`Procfile` and `render.yaml` are pre-configured. Simply connect your GitHub repository to Render/Railway for automatic deployment.

---

## 🏃 Local Execution

---

## 👨‍💻 Project Metadata
- **Author:** Sushmitha Reddy M
- **Institution:** Joy University
- **License:** MIT License
