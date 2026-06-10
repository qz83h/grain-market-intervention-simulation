# 라이브러리 임포트
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 데이터 로드
file_path = '/content/소비량 모델.csv'  # Google Drive 경로를 수정하세요
df = pd.read_csv(file_path)

# 데이터 확인
print(df.head())
print(df.info())

# 결측치 처리
df.fillna(method='ffill', inplace=True)  # 결측치를 이전 값으로 채움


# 주요 특성과 타깃 설정
X = df[['연도', '총인구수', '재배면적']]  # 주요 특성
y_production = df['생산량']  # 타깃: 생산량
y_consumption = df['소비량']  # 타깃

# 데이터 분할
X_train, X_test, y_train_production, y_test_production = train_test_split(
    X, y_production, test_size=0.2, random_state=42
)
X_train, X_test, y_train_consumption, y_test_consumption = train_test_split(
    X, y_consumption, test_size=0.2, random_state=42
)

# 데이터 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear Regression
lr_production = LinearRegression()
lr_production.fit(X_train_scaled, y_train_production)

lr_consumption = LinearRegression()
lr_consumption.fit(X_train_scaled, y_train_consumption)

# Ridge Regression
ridge_production = Ridge(alpha=1.0)
ridge_production.fit(X_train_scaled, y_train_production)

ridge_consumption = Ridge(alpha=1.0)
ridge_consumption.fit(X_train_scaled, y_train_consumption)

# Random Forest
rf_production = RandomForestRegressor(random_state=42)
rf_production.fit(X_train, y_train_production)

rf_consumption = RandomForestRegressor(random_state=42)
rf_consumption.fit(X_train, y_train_consumption)

# 모델 평가 함수
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{model_name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    return mae, rmse

# 생산량 예측 평가
print("Production Predictions:")
evaluate_model(y_test_production, lr_production.predict(X_test_scaled), "Linear Regression")
evaluate_model(y_test_production, ridge_production.predict(X_test_scaled), "Ridge Regression")
evaluate_model(y_test_production, rf_production.predict(X_test), "Random Forest")

# 소비량 예측 평가
print("\nConsumption Predictions:")
evaluate_model(y_test_consumption, lr_consumption.predict(X_test_scaled), "Linear Regression")
evaluate_model(y_test_consumption, ridge_consumption.predict(X_test_scaled), "Ridge Regression")
evaluate_model(y_test_consumption, rf_consumption.predict(X_test), "Random Forest")}