import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler

X = df[['연도', '총인구수', '재배면적']].values  # 독립 변수
y = df['소비량'].values  # 타깃 변수

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 스케일링 (LSTM을 위한 MinMax Scaling)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# XGBoost/LightGBM을 위한 Scaled 데이터도 준비
X_train_table = X_train_scaled
X_test_table = X_test_scaled

# LSTM 입력 형태 변환 (시퀀스 데이터 생성)
n_steps = 3  # 이전 3개의 데이터로 다음 값을 예측
X_train_seq = np.array([X_train_scaled[i : i + n_steps] for i in range(len(X_train_scaled) - n_steps)])
y_train_seq = y_train[n_steps:]

X_test_seq = np.array([X_test_scaled[i : i + n_steps] for i in range(len(X_test_scaled) - n_steps)])
y_test_seq = y_test[n_steps:]

from keras.models import Sequential
from keras.layers import LSTM, Dense

# LSTM 모델 생성
lstm_model = Sequential()
lstm_model.add(LSTM(50, activation='relu', input_shape=(n_steps, X_train_seq.shape[2])))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mse')

# LSTM 모델 학습
lstm_model.fit(X_train_seq, y_train_seq, epochs=50, batch_size=16, verbose=1)

# LSTM 예측
y_pred_lstm = lstm_model.predict(X_test_seq)

from xgboost import XGBRegressor

# XGBoost 모델 생성
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train_table, y_train)

# XGBoost 예측
y_pred_xgb = xgb_model.predict(X_test_table)

from lightgbm import LGBMRegressor

# LightGBM 모델 생성
lgbm_model = LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
lgbm_model.fit(X_train_table, y_train)

# LightGBM 예측
y_pred_lgbm = lgbm_model.predict(X_test_table)

# 평가 함수 정의
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{model_name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    return mae, rmse

# LSTM 평가
print("LSTM Model Evaluation:")
evaluate_model(y_test_seq, y_pred_lstm, "LSTM")

# XGBoost 평가
print("\nXGBoost Model Evaluation:")
evaluate_model(y_test, y_pred_xgb, "XGBoost")

# LightGBM 평가
print("\nLightGBM Model Evaluation:")
evaluate_model(y_test, y_pred_lgbm, "LightGBM")

import matplotlib.pyplot as plt

# 시각화
plt.figure(figsize=(10, 6))

# 실제 값
plt.plot(range(len(y_test_seq)), y_test_seq, label="Actual", linestyle='dashed')

# 예측 값
plt.plot(range(len(y_pred_lstm)), y_pred_lstm.flatten(), label="LSTM Prediction")
plt.plot(range(len(y_pred_xgb)), y_pred_xgb, label="XGBoost Prediction")
plt.plot(range(len(y_pred_lgbm)), y_pred_lgbm, label="LightGBM Prediction")

plt.title("Model Comparison")
plt.xlabel("Test Sample Index")
plt.ylabel("Value")
plt.legend()
plt.grid()
plt.show()
