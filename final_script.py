import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse, r2_score

import xgboost as xgb

# Deep Learning Libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Prophet
from prophet import Prophet

# Plotting
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

class SLDCDataProcessor:
    """Data processor for SLDC forecasting system"""
    
    def __init__(self, weather_path, sldc_path, discom_path, macro_api_data):
        self.weather_path = weather_path
        self.sldc_path = sldc_path
        self.discom_path = discom_path
        self.macro_api_data = macro_api_data
        self.processed_data = None
        
    def load_data(self):
        """Load all data sources"""
        print("Loading data sources...")
        
        # Load weather data
        self.weather_df = pd.read_csv(self.weather_path)
        self.weather_df['datetime'] = pd.to_datetime(self.weather_df['datetime'])
        
        # Load SLDC data
        self.sldc_df = pd.read_csv(self.sldc_path)
        self.sldc_df['Date'] = pd.to_datetime(self.sldc_df['Date'])
        
        # Load DISCOM data - FIX THE DATE COLUMN DETECTION
        self.discom_df = pd.read_csv(self.discom_path)
        print(f"DISCOM columns: {list(self.discom_df.columns)}")  # Debug print
        
        # More robust date column detection
        if 'Date' not in self.discom_df.columns:
            # Look for date-like columns (case insensitive)
            date_cols = [col for col in self.discom_df.columns 
                        if any(date_word in col.lower() for date_word in ['date', 'time', 'day'])]
            
            if date_cols:
                print(f"Found date column: {date_cols[0]}")
                self.discom_df['Date'] = pd.to_datetime(self.discom_df[date_cols[0]])
            else:
                # If no date column found, check if index might be dates
                try:
                    self.discom_df['Date'] = pd.to_datetime(self.discom_df.index)
                    print("Using index as Date column")
                except:
                    raise ValueError("No valid date column found in DISCOM data")
        else:
            self.discom_df['Date'] = pd.to_datetime(self.discom_df['Date'])
        
        # Load macro data from API format
        if isinstance(self.macro_api_data, str):
            # If it's a file path
            self.macro_df = pd.read_csv(self.macro_api_data)
        elif isinstance(self.macro_api_data, list):
            # If it's API response (list of dictionaries)
            self.macro_df = pd.DataFrame(self.macro_api_data)
        else:
            # If it's already a DataFrame
            self.macro_df = self.macro_api_data.copy()
            
        self.macro_df['Date'] = pd.to_datetime(self.macro_df['Date'])
        
        # Process macro data for better feature engineering
        self.macro_df = self.process_macro_features(self.macro_df)
        
        print(f"Weather data shape: {self.weather_df.shape}")
        print(f"SLDC data shape: {self.sldc_df.shape}")
        print(f"DISCOM data shape: {self.discom_df.shape}")
        print(f"Macro data shape: {self.macro_df.shape}")

    def preprocess_data(self):
        """Preprocess and merge all data sources"""
        print("Preprocessing data...")
        
        # Process weather data - aggregate to daily
        weather_daily = self.weather_df.groupby(self.weather_df['datetime'].dt.date).agg({
            'temperature_c': ['mean', 'min', 'max'],
            'humidity': 'mean',
            'wind_speed_kmh': 'mean',
            'pressure': 'mean',
            'precipitation_mm': 'sum',
            'cloud_cover': 'mean',
            'feels_like_c': 'mean',
            'heat_index_c': 'mean',
            'dew_point_c': 'mean',
            'uv_index': 'mean',
            'visibility_km': 'mean',
            'is_sunny': 'sum',
            'is_cloudy': 'sum',
            'is_rainy': 'sum'
        }).reset_index()
        
        # Flatten column names
        weather_daily.columns = ['Date'] + [f"{col[0]}_{col[1]}" if col[1] else col[0] 
                                        for col in weather_daily.columns[1:]]
        weather_daily['Date'] = pd.to_datetime(weather_daily['Date'])
        
        # Process SLDC data - aggregate to daily
        sldc_daily = self.sldc_df.groupby('Date').agg({
            'Demand_Forecast': 'mean',
            'Actuals': 'mean'
        }).reset_index()
        
        # Process DISCOM data - FIXED VERSION
        print(f"DISCOM data columns: {list(self.discom_df.columns)}")
        
        # Find the actual demand columns in DISCOM data
        demand_cols = [col for col in self.discom_df.columns 
                    if any(word in col.lower() for word in ['requisition', 'demand', 'actual'])]
        
        if not demand_cols:
            # If no specific demand columns found, use numeric columns
            numeric_cols = self.discom_df.select_dtypes(include=[np.number]).columns.tolist()
            demand_cols = numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols
        
        print(f"Using DISCOM columns: {demand_cols}")
        
        # Create aggregation dictionary dynamically
        agg_dict = {}
        for col in demand_cols:
            if 'requisition' in col.lower():
                agg_dict['DISCOMS_Requisition'] = col
            elif 'actual' in col.lower() or 'demand' in col.lower():
                agg_dict['Actual_Demand'] = col
        
        # If we couldn't map columns properly, use first two numeric columns
        if not agg_dict and len(demand_cols) >= 2:
            agg_dict = {
                'DISCOMS_Requisition': demand_cols[0],
                'Actual_Demand': demand_cols[1]
            }
        elif not agg_dict and len(demand_cols) >= 1:
            agg_dict = {
                'DISCOMS_Requisition': demand_cols[0]
            }
        
        if agg_dict:
            # Create rename mapping
            rename_mapping = {v: k for k, v in agg_dict.items()}
            discom_daily = self.discom_df.groupby('Date').agg({
                col: 'mean' for col in agg_dict.values()
            }).reset_index()
            discom_daily = discom_daily.rename(columns=rename_mapping)
        else:
            # Fallback: create empty dataframe with expected columns
            print("Warning: Could not find suitable DISCOM columns, creating empty dataframe")
            discom_daily = pd.DataFrame({
                'Date': sldc_daily['Date'],
                'DISCOMS_Requisition': 0,
                'Actual_Demand': 0
            })
        
        # ENSURE ALL DATE COLUMNS ARE THE SAME TYPE
        sldc_daily['Date'] = pd.to_datetime(sldc_daily['Date'])
        weather_daily['Date'] = pd.to_datetime(weather_daily['Date'])
        discom_daily['Date'] = pd.to_datetime(discom_daily['Date'])
        self.macro_df['Date'] = pd.to_datetime(self.macro_df['Date'])
        
        # Start merging data
        merged_df = sldc_daily.copy()
        print(f"Starting merge with SLDC data: {merged_df.shape}")
        
        merged_df = pd.merge(merged_df, weather_daily, on='Date', how='left')
        print(f"After weather merge: {merged_df.shape}")
        
        merged_df = pd.merge(merged_df, discom_daily, on='Date', how='left')
        print(f"After DISCOM merge: {merged_df.shape}")
        
        merged_df = pd.merge(merged_df, self.macro_df, on='Date', how='left')
        print(f"After macro merge: {merged_df.shape}")
        
        # Feature engineering
        merged_df = self.engineer_features(merged_df)
        
        # Handle missing values
        merged_df = merged_df.fillna(method='ffill').fillna(method='bfill')
        
        self.processed_data = merged_df
        print(f"Final processed data shape: {merged_df.shape}")
        
        return merged_df
        
    def process_macro_features(self, macro_df):
        """Process macro API data features"""
        print("Processing macro API features...")
        
        # Create binary features for categorical variables
        if 'Day_of_Week' in macro_df.columns:
            # Create day of week dummies
            day_dummies = pd.get_dummies(macro_df['Day_of_Week'], prefix='Day')
            macro_df = pd.concat([macro_df, day_dummies], axis=1)
        
        # Create Day_of_Week as numeric if it's text

        if 'Season' in macro_df.columns:
            # Create season dummies
            season_dummies = pd.get_dummies(macro_df['Season'], prefix='Season')
            macro_df = pd.concat([macro_df, season_dummies], axis=1)
        
        if 'Quarter' in macro_df.columns:
            # Create quarter dummies
            quarter_dummies = pd.get_dummies(macro_df['Quarter'], prefix='Q')
            macro_df = pd.concat([macro_df, quarter_dummies], axis=1)
        
        # Create interaction features
        if 'Repo_Rate' in macro_df.columns and 'CRR' in macro_df.columns:
            macro_df['Monetary_Policy_Index'] = macro_df['Repo_Rate'] + macro_df['CRR']
        
        if 'WPI_Inflation' in macro_df.columns and 'CPI_Inflation' in macro_df.columns:
            macro_df['Inflation_Spread'] = macro_df['WPI_Inflation'] - macro_df['CPI_Inflation']
        
        if 'Rice_Production' in macro_df.columns and 'Cotton_Production' in macro_df.columns:
            macro_df['Total_Agri_Production'] = macro_df['Rice_Production'] + macro_df['Cotton_Production']
        
        # Create economic sentiment score
        economic_indicators = ['GDP_Growth_Rate', 'Industrial_Production_Index', 
                             'Employment_Rate', 'Credit_Growth']
        available_indicators = [col for col in economic_indicators if col in macro_df.columns]
        
        if available_indicators:
            # Normalize indicators and create composite score
            for col in available_indicators:
                macro_df[f'{col}_normalized'] = (macro_df[col] - macro_df[col].mean()) / macro_df[col].std()
            
            macro_df['Economic_Sentiment'] = macro_df[[f'{col}_normalized' for col in available_indicators]].mean(axis=1)
        
        return macro_df
        
    def preprocess_data(self):
        """Preprocess and merge all data sources"""
        print("Preprocessing data...")
        
        # Process weather data - aggregate to daily
        weather_daily = self.weather_df.groupby(self.weather_df['datetime'].dt.date).agg({
            'temperature_c': ['mean', 'min', 'max'],
            'humidity': 'mean',
            'wind_speed_kmh': 'mean',
            'pressure': 'mean',
            'precipitation_mm': 'sum',
            'cloud_cover': 'mean',
            'feels_like_c': 'mean',
            'heat_index_c': 'mean',
            'dew_point_c': 'mean',
            'uv_index': 'mean',
            'visibility_km': 'mean',
            'is_sunny': 'sum',
            'is_cloudy': 'sum',
            'is_rainy': 'sum'
        }).reset_index()
        
        # Flatten column names
        weather_daily.columns = ['Date'] + [f"{col[0]}_{col[1]}" if col[1] else col[0] 
                                          for col in weather_daily.columns[1:]]
        weather_daily['Date'] = pd.to_datetime(weather_daily['Date'])
        
        # Process SLDC data - aggregate to daily
        sldc_daily = self.sldc_df.groupby('Date').agg({
            'Demand_Forecast': 'mean',
            'Actuals': 'mean'
        }).reset_index()
        
        # Process DISCOM data - aggregate to daily if needed
        discom_daily = self.discom_df.groupby('Date').agg({
            'DISCOMS_Requisition': 'mean',
            'Actual_Demand': 'mean'
        }).reset_index()
        
        # Start merging data
        merged_df = sldc_daily.copy()
        merged_df = pd.merge(merged_df, weather_daily, on='Date', how='left')
        merged_df = pd.merge(merged_df, discom_daily, on='Date', how='left')
        merged_df = pd.merge(merged_df, self.macro_df, on='Date', how='left')
        
        # Feature engineering
        merged_df = self.engineer_features(merged_df)
        
        # Handle missing values
        merged_df = merged_df.fillna(method='ffill').fillna(method='bfill')
        
        self.processed_data = merged_df
        print(f"Final processed data shape: {merged_df.shape}")
        
        return merged_df
    
    def engineer_features(self, df):
        """Engineer additional features"""
        df = df.copy()
        
        # Date features
        df['year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['day'] = df['Date'].dt.day
        df['Day_of_Week'] = df['Date'].dt.dayofweek
        df['quarter'] = df['Date'].dt.quarter
        df['dayofyear'] = df['Date'].dt.dayofyear
        
        # Cyclical features
        df['month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['Day_of_Week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['Day_of_Week'] / 7)
        
        # Temperature features
        if 'temperature_c' in df.columns:
            df['temp_range'] = df['temperature_c_max'] - df['temperature_c_min']
            df['temp_comfort'] = np.abs(df['temperature_c'] - 25)  # Comfort zone deviation
        
        # Lag features for demand
        if 'Actuals' in df.columns:
            for lag in [1, 7, 30]:
                df[f'demand_lag_{lag}'] = df['Actuals'].shift(lag)
            
            # Rolling statistics
            df['demand_ma_7'] = df['Actuals'].rolling(window=7).mean()
            df['demand_ma_30'] = df['Actuals'].rolling(window=30).mean()
            df['demand_std_7'] = df['Actuals'].rolling(window=7).std()
        
        return df

class ProphetForecaster:
    """Facebook Prophet forecaster"""
    
    def __init__(self):
        self.model = None
        self.feature_columns = []
        
    def prepare_data(self, data, target_col='Actuals'):
        """Prepare data for Prophet including macro regressors"""
        prophet_data = data[['Date', target_col]].copy()
        prophet_data.columns = ['ds', 'y']
        
        # Add regressors - prioritize macro features
        macro_regressors = [
            'Is_Holiday', 'Is_Working_Day', 'Is_Festival', 'Is_Festival_Season',
            'Repo_Rate', 'CRR', 'WPI_Inflation', 'CPI_Inflation',
            'GDP_Growth_Rate', 'Industrial_Production_Index', 
            'Economic_Sentiment', 'Monetary_Policy_Index'
        ]
        
        weather_regressors = [
            'temperature_c', 'humidity', 'pressure'
        ]
        
        time_regressors = ['Month', 'Day_of_Week']
        
        all_regressors = macro_regressors + weather_regressors + time_regressors
        available_regressors = [col for col in all_regressors if col in data.columns]
        
        for col in available_regressors[:15]:  # Limit to prevent overfitting
            prophet_data[col] = data[col].values
            
        self.feature_columns = available_regressors[:15]
        print(f"Prophet using {len(self.feature_columns)} regressors: {', '.join(self.feature_columns)}")
        
        return prophet_data
    
    def train(self, data, target_col='Actuals'):
        """Train Prophet model"""
        print("Training Prophet model...")
        prophet_data = self.prepare_data(data, target_col)
        
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        
        # Add regressors
        for col in self.feature_columns:
            self.model.add_regressor(col)
            
        self.model.fit(prophet_data)
        
    def predict(self, future_data):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
            
        future_df = future_data[['Date'] + self.feature_columns].copy()
        future_df.columns = ['ds'] + self.feature_columns
        
        forecast = self.model.predict(future_df)
        return forecast['yhat'].values

class LSTMForecaster:
    """LSTM Neural Network forecaster"""
    
    def __init__(self, sequence_length=30, features=None):
        self.sequence_length = sequence_length
        self.features = features or []
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
    def prepare_sequences(self, data, target_col='Actuals'):
    
    # Define consistent feature set and store it
        feature_cols = ['Actuals', 'temperature_c_mean', 'humidity_mean', 'pressure_mean', 
            'Month', 'Day_of_Week', 'demand_ma_7']
        
        # Store the feature columns for later use in prediction
        self.feature_columns = [col for col in feature_cols if col in data.columns]
        
        print(f"LSTM using features: {self.feature_columns}")
        
        X = data[self.feature_columns].values
        y = data[target_col].values
        
        # Scale features
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Create sequences
        X_seq, y_seq = [], []
        for i in range(self.sequence_length, len(X_scaled)):
            X_seq.append(X_scaled[i-self.sequence_length:i])
            y_seq.append(y_scaled[i])
            
        return np.array(X_seq), np.array(y_seq)
    
    def train(self, data, target_col='Actuals'):
        """Train LSTM model"""
        print("Training LSTM model...")
        X, y = self.prepare_sequences(data, target_col)
        
        # Split data
        train_size = int(0.8 * len(X))
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        # Build model
        self.model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            verbose=0
        )
        
        return history
    
    def predict(self, data, steps=15):
        """Make predictions"""
        if self.model is None:
           raise ValueError("Model not trained yet!")
        
    # Use the same features as training
        if not hasattr(self, 'feature_columns'):
            # Fallback if feature_columns not stored
            feature_cols = ['Actuals', 'temperature_c_mean', 'humidity_mean', 'pressure_mean', 
                'Month', 'Day_of_Week', 'demand_ma_7']
            self.feature_columns = [col for col in feature_cols if col in data.columns]
        
        # Handle missing columns in prediction data
        available_features = []
        for col in self.feature_columns:
            if col in data.columns:
                available_features.append(col)
            else:
                print(f"Warning: Feature '{col}' not found in prediction data")
        
        if len(available_features) != len(self.feature_columns):
            raise ValueError(f"Feature mismatch: Expected {self.feature_columns}, got {available_features}")
        
        X = data[available_features].values
        
        # Check if we have enough data for sequence
        if len(X) < self.sequence_length:
            # Pad with the last available values
            padding_needed = self.sequence_length - len(X)
            if len(X) > 0:
                last_row = X[-1].reshape(1, -1)
                padding = np.repeat(last_row, padding_needed, axis=0)
                X = np.vstack([padding, X])
            else:
                # Use zeros if no data available
                X = np.zeros((self.sequence_length, len(available_features)))
        
        X_scaled = self.scaler_X.transform(X)
        
        # Get last sequence
        last_sequence = X_scaled[-self.sequence_length:].reshape(1, self.sequence_length, -1)
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(steps):
            pred_scaled = self.model.predict(current_sequence, verbose=0)
            pred = self.scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
            predictions.append(pred)
            
            # Update sequence (simplified - in practice, you'd need future feature values)
            new_step = current_sequence[0, -1, :].copy()
            new_step[0] = pred_scaled[0, 0]  # Update demand value
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, :] = new_step
            
        return np.array(predictions)

class XGBoostForecaster:
    """XGBoost regression forecaster"""
    
    def __init__(self):
        self.model = None
        self.feature_columns = []
        
    def prepare_features(self, data):
        """Prepare features for XGBoost including macro features"""
        feature_cols = [
            # Weather features
            'temperature_c', 'temperature_c_min', 'temperature_c_max',
            'humidity', 'pressure', 'cloud_cover_mean',
            'precipitation_mm_sum', 'wind_speed_kmh_mean',
            
            # Time features
            'Month', 'Day_of_Week', 'quarter', 'dayofyear',
            'Month_sin', 'Month_cos', 'day_sin', 'day_cos',
            
            # Macro economic features
            'Is_Holiday', 'Is_Working_Day', 'Is_Festival', 'Is_Festival_Season',
            'Repo_Rate', 'CRR', 'WPI_Inflation', 'CPI_Inflation',
            'Money_Supply_Growth', 'GDP_Growth_Rate', 'Industrial_Production_Index',
            'Employment_Rate', 'AP_GSDP', 'Credit_Growth',
            
            # Agricultural features
            'Rice_Production', 'Cotton_Production', 'Rice_Price',
            
            # Derived macro features
            'Monetary_Policy_Index', 'Inflation_Spread', 'Total_Agri_Production',
            'Economic_Sentiment',
            
            # Season and day features
            'Day_Monday', 'Day_Tuesday', 'Day_Wednesday', 'Day_Thursday', 
            'Day_Friday', 'Day_Saturday', 'Day_Sunday',
            'Season_Winter', 'Season_Spring', 'Season_Summer', 'Season_Autumn',
            'Q_Q1', 'Q_Q2', 'Q_Q3', 'Q_Q4',
            
            # Demand lag features
            'demand_lag_1', 'demand_lag_7', 'demand_ma_7', 'demand_ma_30'
        ]
        
        available_features = [col for col in feature_cols if col in data.columns]
        self.feature_columns = available_features
        
        print(f"Using {len(available_features)} features for XGBoost:")
        print(", ".join(available_features[:10]) + "..." if len(available_features) > 10 else ", ".join(available_features))
        
        return data[available_features]
    
    def train(self, data, target_col='Actuals'):
        """Train XGBoost model"""
        print("Training XGBoost model...")
        X = self.prepare_features(data)
        y = data[target_col]
        
        # Remove rows with NaN values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        self.model.fit(X, y)
        
    def predict(self, future_data):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
            
        X = future_data[self.feature_columns]
        return self.model.predict(X)

class EnsembleForecaster:
    """Ensemble of multiple forecasters with performance-based weighting"""
    
    def __init__(self, weights=None, validation_split=0.2):
        self.weights = weights or {'prophet': 0.33, 'lstm': 0.34, 'xgboost': 0.33}
        self.forecasters = {}
        self.performance_metrics = {}
        self.validation_split = validation_split
        self.adaptive_weights = {}
        
    def add_forecaster(self, name, forecaster):
        """Add a forecaster to the ensemble"""
        self.forecasters[name] = forecaster
        
    def calculate_performance_weights(self, val_data, target_col='Actuals'):
        """Calculate weights based on validation performance"""
        print("Calculating performance-based weights...")
        
        val_predictions = {}
        val_metrics = {}
        
        # Get validation predictions from each model
        for name, forecaster in self.forecasters.items():
            if name == 'lstm':
                # For LSTM, use last part of validation data
                pred = forecaster.predict(val_data, steps=len(val_data))
            else:
                pred = forecaster.predict(val_data)
            
            val_predictions[name] = pred[:len(val_data)]
            
            # Calculate metrics
            y_true = val_data[target_col].values
            y_pred = val_predictions[name]
            
            # Handle length mismatch
            min_len = min(len(y_true), len(y_pred))
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]
            
            mae_score = np.mean(np.abs(y_true - y_pred))
            mse_score = np.mean((y_true - y_pred) ** 2)
            rmse_score = np.sqrt(mse_score)
            mape_score = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            
            val_metrics[name] = {
                'MAE': mae_score,
                'MSE': mse_score, 
                'RMSE': rmse_score,
                'MAPE': mape_score
            }
            
        self.performance_metrics = val_metrics
        
        # Calculate inverse error weights (lower error = higher weight)
        mape_scores = {name: metrics['MAPE'] for name, metrics in val_metrics.items()}
        inverse_mape = {name: 1.0 / (mape + 1e-8) for name, mape in mape_scores.items()}
        total_inverse = sum(inverse_mape.values())
        
        # Normalize to get weights
        self.adaptive_weights = {name: inv_mape / total_inverse 
                               for name, inv_mape in inverse_mape.items()}
        
        print("\n=== MODEL PERFORMANCE ON VALIDATION SET ===")
        for name in val_metrics.keys():
            metrics = val_metrics[name]
            weight = self.adaptive_weights[name]
            print(f"{name.upper():10} | MAPE: {metrics['MAPE']:6.2f}% | "
                  f"RMSE: {metrics['RMSE']:8.1f} | Weight: {weight:5.3f}")
            
        return self.adaptive_weights
        
    def train_all(self, data, target_col='Actuals'):
        """Train all forecasters and calculate performance weights"""
        print("Training ensemble forecasters...")
        
        # Split data for validation
        split_point = int(len(data) * (1 - self.validation_split))
        train_data = data.iloc[:split_point].copy()
        val_data = data.iloc[split_point:].copy()
        
        print(f"Training on {len(train_data)} samples, validating on {len(val_data)} samples")
        
        # Train all models on training data
        for name, forecaster in self.forecasters.items():
            print(f"Training {name}...")
            forecaster.train(train_data, target_col)
            
        # Calculate performance-based weights using validation data
        self.calculate_performance_weights(val_data, target_col)
            
    def predict(self, future_data, steps=15):
        """Make ensemble predictions using performance-based weights"""
        predictions = {}
        
        # Get individual predictions with error handling
        for name, forecaster in self.forecasters.items():
            try:
                if name == 'lstm':
                    pred = forecaster.predict(future_data, steps)
                else:
                    pred = forecaster.predict(future_data)
                predictions[name] = pred
                print(f"✅ {name.upper()} prediction successful")
            except Exception as e:
                print(f"❌ {name.upper()} prediction failed: {str(e)}")
                # Use fallback prediction (average of other models or historical mean)
                if len(predictions) > 0:
                    predictions[name] = np.mean(list(predictions.values()), axis=0)
                else:
                    # Use historical mean as fallback
                    historical_mean = future_data.get('demand_ma_30', [5000] * steps)
                    if isinstance(historical_mean, (int, float)):
                        predictions[name] = np.array([historical_mean] * steps)
                    else:
                        predictions[name] = np.array(historical_mean[:steps])
                
        # Use adaptive weights if available, otherwise use default weights
        weights_to_use = self.adaptive_weights if self.adaptive_weights else self.weights
        
        # Weighted average using performance-based weights
        ensemble_pred = np.zeros(steps)
        total_weight = 0
        
        for name, weight in weights_to_use.items():
            if name in predictions:
                pred_array = np.array(predictions[name])
                if len(pred_array.shape) > 1:
                    pred_array = pred_array.flatten()
                ensemble_pred += weight * pred_array[:steps]
                total_weight += weight
        
        # Normalize if some models failed
        if total_weight > 0 and total_weight != 1.0:
            ensemble_pred = ensemble_pred / total_weight
                
        return ensemble_pred, predictions
    
    def get_model_rankings(self):
        """Get model rankings based on performance"""
        if not self.performance_metrics:
            return None
            
        # Sort by MAPE (lower is better)
        sorted_models = sorted(self.performance_metrics.items(), 
                             key=lambda x: x[1]['MAPE'])
        
        rankings = {}
        for i, (name, metrics) in enumerate(sorted_models, 1):
            rankings[name] = {
                'rank': i,
                'mape': metrics['MAPE'],
                'weight': self.adaptive_weights.get(name, 0)
            }
            
        return rankings

class SLDCForecastingSystem:
    """Main orchestrator for SLDC forecasting system"""
    
    def __init__(self, config):
        self.config = config
        self.data_processor = None
        self.ensemble = None
        self.results = {}
        
    def load_and_process_data(self):
        """Load and process all data"""
        print("=== Loading and Processing Data ===")
        self.data_processor = SLDCDataProcessor(
            weather_path=self.config['weather_path'],
            sldc_path=self.config['sldc_path'], 
            discom_path=self.config['discom_path'],
            macro_api_data=self.config['macro_api_data']
        )
        
        self.data_processor.load_data()
        self.processed_data = self.data_processor.preprocess_data()
        
        return self.processed_data
    
    def setup_forecasters(self):
        """Setup all forecasting models"""
        print("=== Setting up Forecasters ===")
        
        # Initialize forecasters
        prophet_forecaster = ProphetForecaster()
        lstm_forecaster = LSTMForecaster()
        xgboost_forecaster = XGBoostForecaster()
        
        # Create ensemble
        self.ensemble = EnsembleForecaster()
        self.ensemble.add_forecaster('prophet', prophet_forecaster)
        self.ensemble.add_forecaster('lstm', lstm_forecaster)
        self.ensemble.add_forecaster('xgboost', xgboost_forecaster)
        
    def train_models(self):
        """Train all models"""
        print("=== Training Models ===")
        
        # Filter data up to 2024-12-31
        train_data = self.processed_data[
            self.processed_data['Date'] <= '2024-12-31'
        ].copy()
        
        self.ensemble.train_all(train_data, target_col='Actuals')
    def debug_features(self):
        """Debug feature consistency across models"""
        print("\n=== FEATURE DEBUGGING ===")
        
        # Check training data features
        print(f"Training data columns: {len(self.processed_data.columns)}")
        numeric_cols = self.processed_data.select_dtypes(include=[np.number]).columns.tolist()
        print(f"Numeric columns: {len(numeric_cols)}")
        
        # Check what each model expects
        if hasattr(self.ensemble.forecasters['lstm'], 'feature_columns'):
            print(f"LSTM features: {self.ensemble.forecasters['lstm'].feature_columns}")
        
        if hasattr(self.ensemble.forecasters['xgboost'], 'feature_columns'):
            print(f"XGBoost features: {len(self.ensemble.forecasters['xgboost'].feature_columns)}")
        
        if hasattr(self.ensemble.forecasters['prophet'], 'feature_columns'):
            print(f"Prophet features: {self.ensemble.forecasters['prophet'].feature_columns}")    

    def generate_forecasts(self):
        """Generate forecasts for Jan 1-15, 2025"""
        print("=== Generating Forecasts ===")
        
        # Create future dates
        future_dates = pd.date_range(start='2025-01-01', end='2025-01-15', freq='D')
        
        # Create future data with features
        future_data = self.create_future_data(future_dates)
        self.debug_features()
        # Generate predictions
        ensemble_pred, individual_preds = self.ensemble.predict(future_data, steps=15)
        
        # Store results
        self.results = {
            'dates': future_dates,
            'ensemble_forecast': ensemble_pred,
            'individual_forecasts': individual_preds,
            'future_data': future_data
        }
        
        return self.results
    
    def create_future_data(self, future_dates):
        """Create future data with estimated features including macro data"""
        future_data = pd.DataFrame({'Date': future_dates})
        
        # Add date features
        future_data['year'] = future_data['Date'].dt.year
        future_data['Month'] = future_data['Date'].dt.month
        future_data['day'] = future_data['Date'].dt.day
        future_data['Day_of_Week'] = future_data['Date'].dt.dayofweek
        future_data['quarter'] = future_data['Date'].dt.quarter
        future_data['dayofyear'] = future_data['Date'].dt.dayofyear
        
        # Cyclical features
        future_data['month_sin'] = np.sin(2 * np.pi * future_data['Month'] / 12)
        future_data['month_cos'] = np.cos(2 * np.pi * future_data['Month'] / 12)
        future_data['day_sin'] = np.sin(2 * np.pi * future_data['Day_of_Week'] / 7)
        future_data['day_cos'] = np.cos(2 * np.pi * future_data['Day_of_Week'] / 7)
        
        # Add day of week and Month features
        future_data['Month'] = future_data['Month']
        
        # Create day of week dummies
        for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
            future_data[f'Day_{day}'] = (future_data['Date'].dt.day_name() == day).astype(int)
        
        # Estimate weather features (use historical averages for January)
       # Estimate weather features (use historical averages for January)
        jan_weather = self.processed_data[
            self.processed_data['Month'] == 1
        ].select_dtypes(include=[np.number]).mean()

        # Use consistent weather feature names that match training data
        # Use consistent weather feature names that match training data
        weather_mapping = {
            'temperature_c': 'temperature_c_mean',
            'temperature_c_min': 'temperature_c_min', 
            'temperature_c_max': 'temperature_c_max',
            'humidity': 'humidity_mean', 
            'pressure': 'pressure_mean',
            'wind_speed_kmh': 'wind_speed_kmh_mean',
            'cloud_cover': 'cloud_cover_mean',
            'precipitation_mm': 'precipitation_mm_sum'
        }

        for training_col, future_col in weather_mapping.items():
            if training_col in jan_weather.index:
                future_data[future_col] = jan_weather[training_col]
            elif future_col in jan_weather.index:
                future_data[future_col] = jan_weather[future_col]

        # Add any missing weather features with default values
        required_weather_features = ['temperature_c_mean', 'temperature_c_min', 'temperature_c_max', 
                                'humidity_mean', 'pressure_mean']
        for col in required_weather_features:
            if col not in future_data.columns:
                # Use overall mean if January mean not available
                if col.replace('_mean', '').replace('_min', '').replace('_max', '') in self.processed_data.columns:
                    base_col = col.replace('_mean', '').replace('_min', '').replace('_max', '')
                    if '_min' in col:
                        future_data[col] = self.processed_data[base_col].quantile(0.25) if base_col in self.processed_data.columns else 20.0
                    elif '_max' in col:
                        future_data[col] = self.processed_data[base_col].quantile(0.75) if base_col in self.processed_data.columns else 30.0
                    else:
                        future_data[col] = self.processed_data[base_col].mean() if base_col in self.processed_data.columns else 25.0
                else:
                    future_data[col] = 25.0 if 'temp' in col else 50.0 if 'humid' in col else 1013.0
        # Add compatibility for LSTM model - map _mean columns to base names
        if 'temperature_c_mean' in future_data.columns:
            future_data['temperature_c'] = future_data['temperature_c_mean']
        if 'humidity_mean' in future_data.columns:
            future_data['humidity'] = future_data['humidity_mean']
        if 'pressure_mean' in future_data.columns:
            future_data['pressure'] = future_data['pressure_mean']
        # Add macro features for January 2025
        # Use latest available macro data and project forward
        latest_macro = self.processed_data.iloc[-1]
        
        macro_features = [
            'Is_Holiday', 'Is_Working_Day', 'Is_Festival', 'Is_Festival_Season',
            'Repo_Rate', 'CRR', 'WPI_Inflation', 'CPI_Inflation',
            'Money_Supply_Growth', 'GDP_Growth_Rate', 'Industrial_Production_Index',
            'Employment_Rate', 'AP_GSDP', 'Credit_Growth',
            'Rice_Production', 'Cotton_Production', 'Rice_Price'
        ]
        
        for col in macro_features:
            if col in latest_macro.index:
                if col in ['Is_Holiday', 'Is_Working_Day', 'Is_Festival', 'Is_Festival_Season']:
                    # For January 2025, set holidays appropriately
                    if col == 'Is_Holiday':
                        # New Year's Day and Republic Day
                        future_data[col] = future_data['Date'].isin(['2025-01-01', '2025-01-26']).astype(int)
                    elif col == 'Is_Working_Day':
                        # Working days (Monday-Friday, excluding holidays)
                        is_weekday = future_data['Day_of_Week'] < 5
                        is_not_holiday = ~future_data['Date'].isin(['2025-01-01', '2025-01-26'])
                        future_data[col] = (is_weekday & is_not_holiday).astype(int)
                    else:
                        future_data[col] = 0  # Default for festivals
                else:
                    # Use latest value for economic indicators
                    future_data[col] = latest_macro[col]
            else:
                future_data[col] = 0  # Default value
        
        # Add season features for January (Winter)
        season_features = ['Season_Winter', 'Season_Spring', 'Season_Summer', 'Season_Autumn']
        for col in season_features:
            future_data[col] = 1 if col == 'Season_Winter' else 0
        
        # Add quarter features for Q1
        quarter_features = ['Q_Q1', 'Q_Q2', 'Q_Q3', 'Q_Q4']
        for col in quarter_features:
            future_data[col] = 1 if col == 'Q_Q1' else 0
            
        # Add derived macro features
        if 'Repo_Rate' in future_data.columns and 'CRR' in future_data.columns:
            future_data['Monetary_Policy_Index'] = future_data['Repo_Rate'] + future_data['CRR']
        
        if 'WPI_Inflation' in future_data.columns and 'CPI_Inflation' in future_data.columns:
            future_data['Inflation_Spread'] = future_data['WPI_Inflation'] - future_data['CPI_Inflation']
        
        if 'Rice_Production' in future_data.columns and 'Cotton_Production' in future_data.columns:
            future_data['Total_Agri_Production'] = future_data['Rice_Production'] + future_data['Cotton_Production']
        
        # Economic sentiment (use latest value)
        if 'Economic_Sentiment' in latest_macro.index:
            future_data['Economic_Sentiment'] = latest_macro['Economic_Sentiment']
        
        # Add lag features (use recent values)
        recent_demand = self.processed_data['Actuals'].iloc[-30:].mean()
        future_data['demand_lag_1'] = recent_demand
        future_data['demand_lag_7'] = recent_demand
        future_data['demand_ma_7'] = recent_demand
        future_data['demand_ma_30'] = recent_demand
                # For LSTM model compatibility - fill missing features with recent averages or zeros
        lstm_required_features = ['Actuals', 'temperature_c', 'humidity', 'pressure', 
                                'Month', 'Day_of_Week', 'demand_ma_7']
        latest_data = self.processed_data.iloc[-30:]

        for feat in lstm_required_features:
            if feat not in future_data.columns:
                if feat in latest_data.columns:
                    future_data[feat] = latest_data[feat].mean()
                else:
                    future_data[feat] = 0  # Fallback

        print(f"Created future data with {len(future_data.columns)} features")
        
        return future_data
    
    def plot_forecasts(self):
        """Plot forecasting results"""
        print("=== Plotting Results ===")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Historical Data & Forecast', 'Individual Model Forecasts', 
                          'Forecast Comparison', 'Model Performance'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Historical data
        recent_data = self.processed_data.tail(60)
        fig.add_trace(
            go.Scatter(x=recent_data['Date'], y=recent_data['Actuals'],
                      name='Historical Demand', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Ensemble forecast
        fig.add_trace(
            go.Scatter(x=self.results['dates'], y=self.results['ensemble_forecast'],
                      name='Ensemble Forecast', line=dict(color='red', dash='dash')),
            row=1, col=1
        )
        
        # Individual forecasts
        colors = {'prophet': 'green', 'lstm': 'orange', 'xgboost': 'purple'}
        for name, forecast in self.results['individual_forecasts'].items():
            fig.add_trace(
                go.Scatter(x=self.results['dates'], y=forecast[:15],
                          name=f'{name.upper()} Forecast', 
                          line=dict(color=colors.get(name, 'gray'))),
                row=1, col=2
            )
        
        # Forecast comparison
        forecast_df = pd.DataFrame({
            'Date': self.results['dates'],
            'Ensemble': self.results['ensemble_forecast']
        })
        
        for name, forecast in self.results['individual_forecasts'].items():
            forecast_df[name.upper()] = forecast[:15]
            
        for col in forecast_df.columns[1:]:
            fig.add_trace(
                go.Scatter(x=forecast_df['Date'], y=forecast_df[col],
                          name=col, mode='lines+markers'),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="SLDC Andhra Pradesh Power Demand Forecasting Results",
            showlegend=True
        )
        
        fig.show()
        
        return fig
    
    def generate_report(self):
        """Generate forecasting report with model performance"""
        print("=== Generating Report ===")
        
        forecast_df = pd.DataFrame({
            'Date': self.results['dates'],
            'Day': self.results['dates'].strftime('%A'),
            'Ensemble_Forecast_MW': np.round(self.results['ensemble_forecast'], 2)
        })
        
        # Add individual model forecasts
        for name, forecast in self.results['individual_forecasts'].items():
            forecast_df[f'{name.upper()}_Forecast_MW'] = np.round(forecast[:15], 2)
        
        # Summary statistics
        summary = {
            'Forecast Period': 'January 1-15, 2025',
            'Average Daily Demand': f"{np.mean(self.results['ensemble_forecast']):.2f} MW",
            'Peak Demand': f"{np.max(self.results['ensemble_forecast']):.2f} MW", 
            'Min Demand': f"{np.min(self.results['ensemble_forecast']):.2f} MW",
            'Total Energy': f"{np.sum(self.results['ensemble_forecast']) * 24:.2f} MWh"
        }
        
        print("\n=== FORECAST SUMMARY ===")
        for key, value in summary.items():
            print(f"{key}: {value}")
        
        # Model performance and rankings
        if hasattr(self.ensemble, 'get_model_rankings'):
            rankings = self.ensemble.get_model_rankings()
            if rankings:
                print("\n=== MODEL PERFORMANCE RANKINGS ===")
                print(f"{'Model':<10} {'Rank':<6} {'MAPE':<8} {'Weight':<8} {'Performance'}")
                print("-" * 50)
                for name, info in rankings.items():
                    performance = "🥇 Best" if info['rank'] == 1 else "🥈 Good" if info['rank'] == 2 else "🥉 Fair"
                    print(f"{name.upper():<10} {info['rank']:<6} {info['mape']:<7.2f}% {info['weight']:<7.3f} {performance}")
            
        print("\n=== DAILY FORECASTS ===")
        print(forecast_df.to_string(index=False))
        
        return forecast_df, summary
    
    def run_complete_pipeline(self):
        """Run the complete forecasting pipeline"""
        print("🚀 Starting SLDC Power Demand Forecasting System")
        print("=" * 60)
        
        try:
            # Step 1: Load and process data
            self.load_and_process_data()
            
            # Step 2: Setup forecasters
            self.setup_forecasters()
            
            # Step 3: Train models
            self.train_models()
            
            # Step 4: Generate forecasts
            self.generate_forecasts()
            
            # Step 5: Plot results
            self.plot_forecasts()
            
            # Step 6: Generate report
            forecast_df, summary = self.generate_report()
            
            print("\n✅ Forecasting pipeline completed successfully!")
            
            return {
                'forecasts': forecast_df,
                'summary': summary,
                'processed_data': self.processed_data,
                'results': self.results
            }
            
        except Exception as e:
            print(f"❌ Error in forecasting pipeline: {str(e)}")
            raise

# Example usage and configuration
import json

def main():
    # ✅ Load local macro data from JSON file
    with open('ap_data_api_format.json', 'r') as f:
        macro_api_data = json.load(f)  # Load as list of dictionaries

    # ✅ Pass the loaded data into config
    config = {
        'weather_path': 'andhra_pradesh_weather_merged.csv',
        'sldc_path': 'merged_demand_forecast_vs_actuals_reshaped.csv',
        'discom_path': 'merged_discoms_requisition_vs_actual_reshaped.csv',
        'macro_api_data': macro_api_data  # This is the important part
    }

    # ✅ Run forecasting system
    forecasting_system = SLDCForecastingSystem(config)
    results = forecasting_system.run_complete_pipeline()
    return results


# Function to fetch macro data from API
def fetch_macro_data_from_api(api_url, start_date="2022-01-01", end_date="2024-12-31"):
    """
    Fetch macro data from API
    
    Args:
        api_url: Your API endpoint URL
        start_date: Start date for data
        end_date: End date for data
    
    Returns:
        List of dictionaries with macro data
    """
    import requests
    
    try:
        # Example API call - modify according to your API structure
        params = {
            'start_date': start_date,
            'end_date': end_date,
            'format': 'json'
        }
        
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        
        macro_data = response.json()
        print(f"Fetched {len(macro_data)} macro data records from API")
        
        return macro_data
        
    except Exception as e:
        print(f"Error fetching macro data from API: {e}")
        return None

# Function to run with API data
def run_with_api_data(api_url):
    """Run forecasting system with API data"""
    
    # Fetch macro data from API
    macro_data = fetch_macro_data_from_api(api_url)
    
    if macro_data is None:
        print("Failed to fetch macro data from API")
        return None
    
    config = {
        'weather_path': 'andhra_pradesh_weather_merged.csv',
        'sldc_path': 'merged_demand_forecast_vs_actuals_reshaped.csv', 
        'discom_path': 'merged_discoms_requisition_vs_actual_reshaped.csv',
        'macro_api_data': macro_data  # Direct API response
    }
    
    # Initialize and run system
    forecasting_system = SLDCForecastingSystem(config)
    results = forecasting_system.run_complete_pipeline()
    
    return results

if __name__ == "__main__":
    # Example of how to run the system
    print("SLDC Andhra Pradesh Power Demand Forecasting System")
    print("=" * 60)
    print("Options:")
    print("1. Use static macro data (list of dictionaries)")
    print("2. Use API endpoint to fetch macro data")
    print("3. Use CSV file with macro data")
    print("\nConfigure the macro_api_data in config and run main()")
    
    # Example runs:
    results = main()  # With static data
    # results = run_with_api_data('https://your-api-endpoint.com/macro-data')  # With API