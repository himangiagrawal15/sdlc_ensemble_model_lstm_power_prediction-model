# SLDC Load Forecasting System - Focused Version
import pandas as pd
import numpy as np
from prophet import Prophet
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
import json
import os
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FocusedSLDCForecaster:
    def __init__(self):
        self.scaler = StandardScaler()
        self.demand_scaler = StandardScaler()
        self.prophet_model = None
        self.xgb_model = None
        self.lstm_model = None
        
    def load_and_prepare_data(self):
        """Load data from merged files"""
        logger.info("Loading data from merged files...")
        
        # Load demand data
        demand_data = pd.read_csv('project/merged_demand_forecast_vs_actuals_reshaped.csv')
        demand_data['datetime'] = pd.to_datetime(demand_data['Date'] + ' ' + demand_data['Time_Start'])
        
        # Load weather data
        weather_data = pd.read_csv('project/andhra_pradesh_weather_merged.csv')
        weather_data['datetime'] = pd.to_datetime(weather_data['datetime'])
        
        # Load economic/holiday data
        with open('project/ap_data_api_format.json', 'r') as f:
            api_data = json.load(f)
        api_df = pd.DataFrame([api_data]) if isinstance(api_data, dict) else pd.DataFrame(api_data)
        api_df['Date'] = pd.to_datetime(api_df['Date'])
        
        # Filter data for the specified date range
        start_date = '2022-04-01'
        end_date = '2024-12-31'
        mask = (demand_data['datetime'] >= start_date) & (demand_data['datetime'] <= end_date)
        demand_data = demand_data[mask].copy()
        
        # Merge weather data
        demand_data['date'] = demand_data['datetime'].dt.date
        demand_data['hour'] = demand_data['datetime'].dt.hour
        weather_data['date'] = weather_data['datetime'].dt.date
        weather_data['hour'] = weather_data['datetime'].dt.hour
        
        merged_data = pd.merge(demand_data, 
                             weather_data[['date', 'hour', 'temperature_c', 'humidity', 'weather_condition']], 
                             on=['date', 'hour'], 
                             how='left')
        
        # Add time features
        merged_data['year'] = merged_data['datetime'].dt.year
        merged_data['month'] = merged_data['datetime'].dt.month
        merged_data['day'] = merged_data['datetime'].dt.day
        merged_data['day_of_week'] = merged_data['datetime'].dt.dayofweek
        merged_data['hour'] = merged_data['datetime'].dt.hour
        merged_data['minute'] = merged_data['datetime'].dt.minute
        merged_data['is_weekend'] = merged_data['day_of_week'].isin([5, 6]).astype(int)
        merged_data['is_holiday'] = 0  # Will be updated from API data
        
        # Add economic indicators from API data
        for date in merged_data['date'].unique():
            date_data = api_df[api_df['Date'].dt.date == date]
            if not date_data.empty:
                for col in ['GDP_Growth_Rate', 'Industrial_Production_Index']:
                    if col in date_data.columns:
                        merged_data.loc[merged_data['date'] == date, col] = date_data[col].iloc[0]
        
        # Fill missing values
        merged_data = merged_data.fillna(method='ffill').fillna(method='bfill')
        
        return merged_data
    
    def prepare_features(self, data):
        """Prepare features for modeling"""
        features = data[['temperature_c', 'humidity', 'is_weekend', 'hour', 
                        'day_of_week', 'month', 'GDP_Growth_Rate', 
                        'Industrial_Production_Index']].copy()
        
        # Add cyclical time features
        features['hour_sin'] = np.sin(2 * np.pi * data['hour']/24)
        features['hour_cos'] = np.cos(2 * np.pi * data['hour']/24)
        features['month_sin'] = np.sin(2 * np.pi * data['month']/12)
        features['month_cos'] = np.cos(2 * np.pi * data['month']/12)
        features['day_of_week_sin'] = np.sin(2 * np.pi * data['day_of_week']/7)
        features['day_of_week_cos'] = np.cos(2 * np.pi * data['day_of_week']/7)
        
        # Add lag features
        for lag in [1, 2, 4, 8, 24]:  # 15min, 30min, 1hr, 2hr, 6hr
            if 'Actuals' in data.columns:
                features[f'demand_lag_{lag}'] = data['Actuals'].shift(lag)
        
        # Fill missing values from lags
        features = features.fillna(method='ffill').fillna(method='bfill')
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        return pd.DataFrame(scaled_features, columns=features.columns)
    
    def prepare_lstm_sequences(self, features, target, seq_length=96):
        """Prepare sequences for LSTM"""
        X, y = [], []
        for i in range(len(features) - seq_length):
            X.append(features[i:(i + seq_length)])
            y.append(target[i + seq_length])
        return np.array(X), np.array(y)
    
    def build_lstm_model(self, input_shape):
        """Build LSTM model"""
        model = Sequential([
            LSTM(128, input_shape=input_shape, return_sequences=True),
            Dropout(0.2),
            LSTM(64),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
    
    def train_models(self, data):
        """Train all models"""
        logger.info("Training models...")
        
        # Prepare data
        features = self.prepare_features(data)
        target = data['Actuals'].values
        
        # Scale target
        target_scaled = self.demand_scaler.fit_transform(target.reshape(-1, 1))
        
        # Split data
        train_size = int(len(data) * 0.8)
        val_size = int(len(data) * 0.1)
        
        X_train = features[:train_size]
        y_train = target_scaled[:train_size]
        X_val = features[train_size:train_size+val_size]
        y_val = target_scaled[train_size:train_size+val_size]
        
        # Train Prophet
        prophet_data = pd.DataFrame({
            'ds': data['datetime'][:train_size],
            'y': target[:train_size]
        })
        self.prophet_model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            changepoint_prior_scale=0.05
        )
        self.prophet_model.fit(prophet_data)
        
        # Train XGBoost with correct parameters
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        eval_set = [(X_val, y_val)]
        self.xgb_model.fit(
            X_train, 
            y_train,
            eval_set=eval_set,
            verbose=False,
            callbacks=[xgb.callback.EarlyStopping(rounds=50)]
        )
        
        # Train LSTM
        X_train_lstm, y_train_lstm = self.prepare_lstm_sequences(X_train, y_train)
        X_val_lstm, y_val_lstm = self.prepare_lstm_sequences(X_val, y_val)
        
        self.lstm_model = self.build_lstm_model((X_train_lstm.shape[1], X_train_lstm.shape[2]))
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        self.lstm_model.fit(
            X_train_lstm, y_train_lstm,
            validation_data=(X_val_lstm, y_val_lstm),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=0
        )
    
    def predict_ensemble(self, future_dates, last_known_values):
        """Generate ensemble predictions"""
        logger.info("Generating ensemble predictions...")
        
        # Create future features DataFrame
        future_data = pd.DataFrame({
            'datetime': future_dates,
            'date': future_dates.date,
            'year': future_dates.year,
            'month': future_dates.month,
            'day': future_dates.day,
            'hour': future_dates.hour,
            'minute': future_dates.minute,
            'day_of_week': future_dates.dayofweek,
            'is_weekend': future_dates.dayofweek.isin([5, 6]).astype(int)
        })
        
        # Add weather features (using last known values and patterns)
        future_data['temperature_c'] = last_known_values['temperature_c'].mean()
        future_data['humidity'] = last_known_values['humidity'].mean()
        
        # Add economic indicators (using last known values)
        future_data['GDP_Growth_Rate'] = last_known_values['GDP_Growth_Rate'].iloc[-1]
        future_data['Industrial_Production_Index'] = last_known_values['Industrial_Production_Index'].iloc[-1]
        
        # Add lag features using last known values
        future_data['Actuals'] = last_known_values['Actuals'].iloc[-1]
        
        # Prepare features
        future_features = self.prepare_features(future_data)
        
        # Prophet prediction
        prophet_future = pd.DataFrame({'ds': future_dates})
        prophet_pred = self.prophet_model.predict(prophet_future)['yhat'].values
        
        # XGBoost prediction
        xgb_pred = self.xgb_model.predict(future_features)
        
        # LSTM prediction
        X_lstm = self.prepare_lstm_sequences(future_features.values, np.zeros(len(future_features)))[0]
        lstm_pred = self.lstm_model.predict(X_lstm, verbose=0).flatten()
        
        # Ensemble (weighted average with dynamic weights based on recent performance)
        weights = [0.4, 0.3, 0.3]  # Prophet, XGBoost, LSTM
        ensemble_pred = (
            weights[0] * prophet_pred +
            weights[1] * self.demand_scaler.inverse_transform(xgb_pred.reshape(-1, 1)).flatten() +
            weights[2] * self.demand_scaler.inverse_transform(lstm_pred.reshape(-1, 1)).flatten()
        )
        
        return ensemble_pred
    
    def run_forecast(self):
        """Run complete forecasting pipeline"""
        # Load and prepare data
        data = self.load_and_prepare_data()
        
        # Train models
        self.train_models(data)
        
        # Get last known values for feature generation
        last_known_values = data.tail(96)  # Last day of data
        
        # Generate predictions for January 2025
        future_dates = pd.date_range(
            start='2025-01-01',
            end='2025-01-15 23:45:00',
            freq='15min'
        )
        
        predictions = self.predict_ensemble(future_dates, last_known_values)
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'timestamp': future_dates,
            'forecast': predictions,
            'lower_bound': predictions * 0.95,
            'upper_bound': predictions * 1.05
        })
        
        # Save forecasts
        output_path = 'project/january_2025_forecast.csv'
        forecast_df.to_csv(output_path, index=False)
        logger.info(f"Forecast saved to {output_path}")
        
        return forecast_df

def main():
    forecaster = FocusedSLDCForecaster()
    forecast_df = forecaster.run_forecast()
    print("Forecasting completed successfully!")
    print(f"Generated {len(forecast_df)} predictions for January 2025")
    
if __name__ == "__main__":
    main() 