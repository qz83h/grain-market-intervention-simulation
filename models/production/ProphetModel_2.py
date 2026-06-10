from prophet import Prophet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----- 1. 데이터 입력 및 전처리 -----
data = {
    "기준연도": [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    "평균가격": [42748, 40584, 36284, 32266, 40043, 47996, 47367, 54829, 48742, 49321],
}

# 생산량 데이터
data_production = {
    "연도": [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    "생산량": [4240739, 4326915, 4196691, 3972468, 3868045, 3744450, 3506578, 3881601, 3763700, 3702239],
}

# 데이터프레임 생성
df_price = pd.DataFrame(data).rename(columns={"기준연도": "ds", "평균가격": "y"})
df_price['ds'] = pd.to_datetime(df_price['ds'], format='%Y')

df_production = pd.DataFrame(data_production).rename(columns={"연도": "ds", "생산량": "y"})
df_production['ds'] = pd.to_datetime(df_production['ds'], format='%Y')

# ----- 2. Prophet 모델 학습 -----
# 생산량 예측 모델
model_production = Prophet()
model_production.fit(df_production)

# 가격 예측 모델
model_price = Prophet()
model_price.fit(df_price)

# 미래 예측 (2024~2028년)
future_years = pd.DataFrame({'ds': pd.date_range(start='2024', periods=5, freq='Y')})
production_forecast = model_production.predict(future_years)
price_forecast = model_price.predict(future_years)

# ----- 3. 결과 병합 및 시장격리 기준 적용 -----
# 결과 병합
future_predictions = pd.DataFrame({
    '연도': future_years['ds'].dt.year,
    '예측_생산량': production_forecast['yhat'],
    '예측_가격': price_forecast['yhat'],
})

# 평균 생산량과 평년 가격
평균_생산량 = df_production['y'].mean()
평년_가격 = df_price['y'].mean()

# 초과 생산량 비율 계산
future_predictions['초과생산량비율'] = ((future_predictions['예측_생산량'] - 평균_생산량) / 평균_생산량) * 100

# 가격 변동률 계산
future_predictions['가격변동률'] = ((future_predictions['예측_가격'] - 평년_가격) / 평년_가격) * 100

# 시장격리 여부 판정
future_predictions['시장격리_여부'] = np.where(
    (future_predictions['초과생산량비율'] >= 3) | (future_predictions['가격변동률'] <= -10),
    1, 0
)

# ----- 4. 결과 출력 및 시각화 -----
print("Market Stabilization Predictions:")
print(future_predictions)

# 시각화
fig, ax1 = plt.subplots(figsize=(10, 6))

# 예측된 생산량 그래프
ax1.plot(future_predictions['연도'], future_predictions['예측_생산량'], label='Predicted Production', color='blue', marker='o')
ax1.axhline(y=평균_생산량 * 1.03, color='blue', linestyle='--', label='Production Threshold (3% Excess)')
ax1.set_xlabel("Year")
ax1.set_ylabel("Production (Tons)", color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# 가격 변동률 그래프
ax2 = ax1.twinx()
ax2.plot(future_predictions['연도'], future_predictions['가격변동률'], label='Price Change (%)', color='red', linestyle='--', marker='x')
ax2.axhline(y=-10, color='red', linestyle='--', label='Price Drop Threshold (-10%)')
ax2.set_ylabel("Price Change (%)", color='red')
ax2.tick_params(axis='y', labelcolor='red')

# 시장격리 여부 마커
for idx, row in future_predictions.iterrows():
    if row['시장격리_여부'] == 1:
        plt.axvspan(row['연도'] - 0.5, row['연도'] + 0.5, color='gray', alpha=0.2)

# 범례 및 타이틀 설정
fig.suptitle("Market Stabilization Prediction (Production and Price Impact)")
fig.tight_layout()
fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
plt.show()