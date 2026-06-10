import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit

plt.rc('font', family='AppleGothic')  # 맑은 고딕 폰트 설정
plt.rc('axes', unicode_minus=False)    # 음수 기호 깨짐 방지



# -------------------------------
# 과거 예측값 및 모델별 사용 변수 입력
# -------------------------------
# Prophet 과거 예측값
prophet_past = [4826039.99, 5104292.25, 5470641.86, 4419013.88, 4608516.83, 4886740.11, 5253089.64, 4201461.67, 4390964.61, 4669187.89, 5035537.43, 3983909.45, 4173412.39, 4451635.68]   # Prophet 모델의 과거 예측값

# ARIMA 과거 예측값
arima_past = [3704180, 3704872, 3705118, 3705206, 3705238, 3705249, 3705253, 3705254, 3705255, 3705255, 3705255, 3705255, 3705255, 3705255]     # ARIMA 모델의 과거 예측값

# LSTM 과거 예측값
lstm_past = [4295413, 4224019, 4006185, 4282621.00, 4207400.50, 4137340.00, 4055181.00, 3949746.75, 3869519.75, 3810688.25, 3750012.00, 3714947.50, 3707033.75, 3717849.50]      # LSTM 모델의 과거 예측값

# Prophet 모델에서 사용된 변수들 (총인구수)
prophet_variable1 = [50515666, 50734284, 50948272, 51141463, 51327916, 51529338, 51696216, 51778544, 51826059, 51849861, 51829023, 51638809, 51439038, 51325329]   # Prophet 모델 사용 변수 1


# LSTM 모델에서 사용된 변수들(풍휴작지표, 농가, 농가인구)
lstm_variable1 = [-0.074, -0.030, -0.077, -0.010, -0.036, -0.058, -0.046, -0.007, -0.057, -0.080, -0.071, -0.110, -0.044, -0.057]
lstm_variable2 = [1177318, 1163209, 1151116, 1142029, 1120776, 1088518, 1068274, 1042017, 1020838, 1007158, 1035193, 1031210, 1022797, 999022]      
lstm_variable3 = [3062956, 2962113, 2911540, 2847435, 2751792, 2569387, 2496406, 2422256, 2314982, 2244783, 2314064, 2215498, 2165626, 2088781]

# 실제값 입력 (학습용 타겟 데이터)
meta_target = [4295413, 4224019, 4006185, 4230011, 4240739, 4326915, 4196691, 3972468, 3868045, 3744450, 3506578, 3881601, 3763700, 3702239]    # 실제 과거 생산량 데이터

# -------------------------------
# 메타 모델 입력 데이터프레임 생성 (과거 예측값과 변수 포함)
# -------------------------------
meta_input = pd.DataFrame({
    "Prophet_Past": prophet_past,        # Prophet 과거 예측값
    "ARIMA_Past": arima_past,            # ARIMA 과거 예측값
    "LSTM_Past": lstm_past,              # LSTM 과거 예측값
    "Prophet_변수1": prophet_variable1,  # Prophet 사용 변수 1
    "LSTM_변수1": lstm_variable1,        # LSTM 사용 변수 1
    "LSTM_변수2": lstm_variable2,         # LSTM 사용 변수 2
    "LSTM_변수3": lstm_variable3
})

# -------------------------------
# XGBoost 메타 모델 학습 (과거 데이터를 기반으로 학습)
# -------------------------------
meta_model = XGBRegressor()
meta_model.fit(meta_input, meta_target)

# -------------------------------
# 미래 예측값 준비
# -------------------------------
# Prophet 미래 예측값
prophet_future = [4817985.21, 3766357.23, 3955860.17, 4234083.46, 4600432.99]

# ARIMA 미래 예측값
arima_future = [3704180, 3704872, 3705118, 3705206, 3705238]

# LSTM 미래 예측값
lstm_future = [3713416.00, 3710953.50, 3735898.25, 3767419.75, 3776377.75]

# 미래 예측에 사용될 변수 (Prophet 및 LSTM 사용 변수)
prophet_variable1_future = [51750000, 51700000, 51600000, 51500000, 51450000]

lstm_variable1_future = [-0.0489, -0.0612, -0.0562, -0.0572, -0.0469]
lstm_variable2_future = [1020000,1006000,992000,978000,964000]
lstm_variable3_future = [2118000,2075000,2034000,1993000,1962000]

# 미래 예측용 데이터프레임 생성
future_input = pd.DataFrame({
    "Prophet_Past": prophet_future,        # Prophet 미래 예측값
    "ARIMA_Past": arima_future,            # ARIMA 미래 예측값
    "LSTM_Past": lstm_future,              # LSTM 미래 예측값
    "Prophet_변수1": prophet_variable1_future,  # Prophet 변수 1  
    "LSTM_변수1": lstm_variable1_future,        # LSTM 변수 1
    "LSTM_변수2": lstm_variable2_future,
    "LSTM_변수3": lstm_variable3_future
})

# -------------------------------
# 최종 예측
# -------------------------------
final_pred = meta_model.predict(future_input)

# 결과 확인
print("미래 예측 결과:")
print("Prophet 미래 예측:", prophet_future)
print("ARIMA 미래 예측:", arima_future)
print("LSTM 미래 예측:", lstm_future)
print("최종 XGBoost 스태킹 예측:", final_pred)

# 시각화
plt.plot(range(len(meta_target)), meta_target, label="Actual (과거 실제값)")
plt.plot(range(len(meta_target), len(meta_target) + len(final_pred)), final_pred, label="Forecast (미래 예측값)")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Production")
plt.title("Actual vs Predicted")
plt.show()


def backtesting(meta_input, meta_target, test_size=5):
    # 훈련 및 테스트 데이터 나누기
    train_input = meta_input.iloc[:-test_size]
    train_target = meta_target[:-test_size]
    test_input = meta_input.iloc[-test_size:]
    test_target = meta_target[-test_size:]

    # 모델 학습
    model = XGBRegressor()
    model.fit(train_input, train_target)

    # 테스트 데이터 예측
    test_pred = model.predict(test_input)

    # 평가 지표
    mae = mean_absolute_error(test_target, test_pred)
    r2 = r2_score(test_target, test_pred)

    print(f"백테스트 MAE: {mae:.4f}")
    print(f"백테스트 R²: {r2:.4f}")

    # 결과 시각화
    plt.plot(range(len(train_target)), train_target, label="Train Actual")
    plt.plot(range(len(train_target), len(train_target) + len(test_target)), test_target, label="Test Actual")
    plt.plot(range(len(train_target), len(train_target) + len(test_pred)), test_pred, label="Backtest Prediction")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Production")
    plt.title("Backtesting: Actual vs Predicted")
    plt.show()

# 백테스트 실행
backtesting(meta_input, np.array(meta_target), test_size=5)


# -------------------------------
# Time Series Split 교차 검증
# -------------------------------
def time_series_cross_validation(meta_input, meta_target, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    mae_scores = []
    r2_scores = []

    for train_index, test_index in tscv.split(meta_input):
        # 훈련 및 검증 데이터 분할
        X_train, X_test = meta_input.iloc[train_index], meta_input.iloc[test_index]
        y_train, y_test = meta_target[train_index], meta_target[test_index]

        # 모델 학습 및 예측
        model = XGBRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # 평가 지표 계산
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mae_scores.append(mae)
        r2_scores.append(r2)

    print(f"평균 MAE (교차 검증): {np.mean(mae_scores):.4f}")
    print(f"평균 R² (교차 검증): {np.mean(r2_scores):.4f}")

# 교차 검증 실행
time_series_cross_validation(meta_input, np.array(meta_target))