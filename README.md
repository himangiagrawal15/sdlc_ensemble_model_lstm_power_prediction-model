Power Forecasting Model using Ensemble & LSTM
📌 Project Overview

This project implements a power forecasting system that predicts future electricity demand using a combination of Ensemble Learning and LSTM (Long Short-Term Memory) networks.
The solution is part of an SDLC-based workflow, covering requirement analysis, design, development, testing, and deployment.

The final_script.py file contains the complete code for data preprocessing, model training, evaluation, and prediction.
⚙️ Features

    Data Preprocessing: Handles missing values, scaling, and time-series structuring.

    LSTM Model: Captures long-term temporal dependencies in power usage data.

    Ensemble Learning: Combines predictions from multiple models to improve accuracy.

    Evaluation Metrics: Includes MAE, RMSE, and MAPE for model performance comparison.

    Visualization: Plots predicted vs actual power usage trends.
Workflow (SDLC Phases)
Requirement Analysis – Understanding power usage patterns and forecasting needs.

    Design – Architecting the model pipeline with LSTM and Ensemble techniques.

    Implementation – Writing and integrating code in final_script.py.

    Testing – Evaluating model accuracy and stability.

    Deployment – Preparing the model for real-world predictions

To run this file :
git clone https://github.com/himangiagrawal15/sdlc_ensemble_model_lstm_power_prediction-model.git
cd sdlc_ensemble_model_lstm_power_prediction-model

In the terminal run:
   python final_script.py





EXPLANATION:


---

## **1. Models Used**

Your script uses **three forecasting models** combined into an **ensemble**:

| Model Name             | Type                                                                           | Library            |
| ---------------------- | ------------------------------------------------------------------------------ | ------------------ |
| **ProphetForecaster**  | Additive time-series model with trend & seasonality + external regressors      | `prophet`          |
| **LSTMForecaster**     | Deep learning recurrent neural network for sequential data                     | `TensorFlow/Keras` |
| **XGBoostForecaster**  | Gradient boosting regression trees                                             | `xgboost`          |
| **EnsembleForecaster** | Weighted average of the above models (weights based on validation performance) | Custom             |

---

## **2. Factors / Features Considered by Each Model**

### **ProphetForecaster**

Uses **time series** + **external regressors**:

* **Macro-economic**: `Is_Holiday`, `Is_Working_Day`, `Is_Festival`, `Is_Festival_Season`, `Repo_Rate`, `CRR`, `WPI_Inflation`, `CPI_Inflation`, `GDP_Growth_Rate`, `Industrial_Production_Index`, `Economic_Sentiment`, `Monetary_Policy_Index`
* **Weather**: `temperature_c`, `humidity`, `pressure`
* **Time features**: `Month`, `Day_of_Week`

---

### **LSTMForecaster**

Uses recent **sequence patterns** (default 30 days) with:

* `Actuals` (past demand values)
* `temperature_c_mean`
* `humidity_mean`
* `pressure_mean`
* `Month`
* `Day_of_Week`
* `demand_ma_7` (7-day moving average of demand)

---

### **XGBoostForecaster**

Uses a **large combined feature set**:

* **Weather**: temperature (mean/min/max), humidity, pressure, cloud cover, precipitation, wind speed
* **Time**: Month, Day\_of\_Week, quarter, dayofyear, cyclic encodings (`month_sin`, `month_cos`, `day_sin`, `day_cos`)
* **Macro-economic**: holidays/festivals, Repo\_Rate, CRR, inflation rates, GDP growth, industrial production, employment rate, AP\_GSDP, credit growth
* **Agricultural**: rice & cotton production, rice price
* **Derived features**: Monetary\_Policy\_Index, Inflation\_Spread, Total\_Agri\_Production, Economic\_Sentiment
* **Season & quarter dummies**
* **Demand history**: demand lags (1-day, 7-day), moving averages (7-day, 30-day)

---

## **3. Outputs Produced**

When you run `run_complete_pipeline()`:

1. **Forecasts**

   * **Ensemble Forecast** for the next 15 days
   * **Individual model forecasts** (Prophet, LSTM, XGBoost)

2. **Performance Metrics**

   * For each model on validation set: MAPE, RMSE, and adaptive weights used in ensemble.

3. **Summary Statistics**

   * Forecast period
   * Average daily demand
   * Peak demand
   * Minimum demand
   * Total energy (MWh) over the forecast period

4. **Visualizations**

   * Historical vs forecast demand
   * Individual model forecast comparison
   * Forecast overlap plots
   * Model performance chart

5. **Report DataFrame**

   * Date
   * Day of week
   * Ensemble forecast (MW)
   * Prophet forecast (MW)
   * LSTM forecast (MW)
   * XGBoost forecast (MW)

---


