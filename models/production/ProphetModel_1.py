from prophet import Prophet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 데이터 입력
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

# Prophet 모델 학습용 데이터 준비
df_production = df[['연도', '생산량']].rename(columns={"연도": "ds", "생산량": "y"})
df_production['ds'] = pd.to_datetime(df_production['ds'], format='%Y')

# Prophet 모델 학습
model_production = Prophet()
model_production.fit(df_production)

# 미래 예측 (2024~2028년)
future_years = pd.DataFrame({'ds': pd.date_range(start='2024', periods=5, freq='Y')})
production_forecast = model_production.predict(future_years)

# 예측 결과 병합
future_predictions = pd.DataFrame({
    '연도': future_years['ds'].dt.year,
    '예측_생산량': production_forecast['yhat']
})

# 실제 생산량 데이터 결합 (기존 데이터에서 생산량 추가)
latest_actual_data = df.tail(5)[['연도', '생산량']].reset_index(drop=True)
future_predictions['실제_생산량'] = latest_actual_data['생산량']

# 초과 생산량 계산 및 시장격리 여부 판단
future_predictions['초과생산량'] = future_predictions['실제_생산량'] - future_predictions['예측_생산량']
future_predictions['초과생산량비율'] = (future_predictions['초과생산량'] / future_predictions['예측_생산량']) * 100
future_predictions['시장격리_여부'] = np.where(future_predictions['초과생산량비율'] > 3, 1, 0)

# 결과 출력
print("Market Stabilization Predictions:")
print(future_predictions)

# 시각화
fig, ax1 = plt.subplots(figsize=(10, 6))

# 예측된 생산량과 실제 생산량 그래프
ax1.plot(future_predictions['연도'], future_predictions['예측_생산량'], label='Predicted Production', color='blue', marker='o')
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
fig.suptitle("Market Stabilization Prediction (Actual vs Predicted Production and Excess Ratio)", fontsize=12)
fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
fig.tight_layout()

plt.show()
