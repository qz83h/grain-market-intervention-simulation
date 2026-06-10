import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ----- 1. 데이터 입력 및 전처리 -----
data = {
    "연도": [1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001,
           2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011,
           2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021,
           2022, 2023],
    "생산량": [5330826, 4749562, 5059764, 4694956, 5322962, 5449561, 5096879, 5262700, 5290771, 5514796,
             4926746, 4451135, 5000149, 4768368, 4679991, 4407743, 4843478, 4916080, 4295413, 4224019,
             4006185, 4230011, 4240739, 4326915, 4196691, 3972468, 3868045, 3744450, 3506578, 3881601,
             3763700, 3702239]
}

# 데이터프레임 생성
df = pd.DataFrame(data)

# 학습용 데이터 준비
X = df[['연도']]
y = df['생산량']

# ----- 2. XGBoost 모델 학습 -----
# 훈련 및 테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost 모델 생성 및 학습
model_xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model_xgb.fit(X_train, y_train)

# 모델 성능 평가
y_pred_test = model_xgb.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
print(f"XGBoost Model RMSE on Test Data: {rmse:.2f}")

# ----- 3. 미래 예측 (2024~2028년) -----
future_years = pd.DataFrame({'연도': range(2024, 2029)})
future_predictions = pd.DataFrame({
    '연도': future_years['연도'],
    '예측_생산량': model_xgb.predict(future_years)
})

# 실제 데이터 결합 (최근 5개년 실제 생산량)
latest_actual_data = df.tail(5).reset_index(drop=True)
future_predictions['실제_생산량'] = latest_actual_data['생산량']

# ----- 4. 초과 생산량 계산 및 시장격리 여부 판단 -----
평균_생산량 = df['생산량'].mean()
future_predictions['초과생산량'] = future_predictions['실제_생산량'] - future_predictions['예측_생산량']
future_predictions['초과생산량비율'] = (future_predictions['초과생산량'] / future_predictions['예측_생산량']) * 100
future_predictions['시장격리_여부'] = np.where(future_predictions['초과생산량비율'] > 3, 1, 0)

# 결과 출력
print("Market Stabilization Predictions (Using XGBoost):")
print(future_predictions)

# ----- 5. 시각화 -----
fig, ax1 = plt.subplots(figsize=(10, 6))

# 예측된 생산량과 실제 생산량 그래프
ax1.plot(future_predictions['연도'], future_predictions['예측_생산량'], label='Predicted Production (XGBoost)', color='blue', marker='o')
ax1.plot(future_predictions['연도'], future_predictions['실제_생산량'], label='Actual Production', color='green', marker='x')
ax1.fill_between(future_predictions['연도'], future_predictions['예측_생산량'], future_predictions['실제_생산량'],
                 where=(future_predictions['초과생산량'] > 0), color='red', alpha=0.3, label='Excess Production')
ax1.set_xlabel("Year")
ax1.set_ylabel("Production (Tons)", color='black')
ax1.tick_params(axis='y', labelcolor='black')

# 초과 생산량 비율 막대그래프
ax2 = ax1.twinx()
ax2.bar(future_predictions['연도'], future_predictions['초과생산량비율'], color='orange', alpha=0.5, label='Excess Ratio (%)')
ax2.set_ylabel("Excess Production Ratio (%)", color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

# 제목 및 범례 설정
fig.suptitle("Market Stabilization Prediction (Using XGBoost)", fontsize=12)
fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
fig.tight_layout()

plt.show()
