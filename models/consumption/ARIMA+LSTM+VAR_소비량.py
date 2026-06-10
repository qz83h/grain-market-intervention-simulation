import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# 한글 폰트 설정 (Windows 환경 예시)
plt.rc('font', family='Malgun Gothic')  # 맑은 고딕 폰트 설정
plt.rc('axes', unicode_minus=False)    # 음수 기호 깨짐 방지

# 제공된 데이터
데이터 = {
    "연도": list(range(1992, 2024)),
    "총인구수": [
        44503200, 45001113, 45416339, 45858029, 46266256, 46684069, 46991171, 47335678, 47732558, 48021543, 48229948,
        48386823, 48583805, 48782274, 48991779, 49268928, 49540367, 49773145, 50515666, 50734284, 50948272, 51141463,
        51327916, 51529338, 51696216, 51778544, 51826059, 51849861, 51829023, 51638809, 51439038, 51325329
    ],
    "소비량": [
        112.9, 110.2, 108.3, 106.5, 104.9, 102.4, 99.2, 96.9, 93.6, 88.9, 87.0, 83.2, 82.0, 80.7, 78.8, 76.9, 75.8,
        74.0, 72.8, 71.2, 69.8, 67.2, 65.1, 62.9, 61.9, 61.8, 61.0, 59.2, 57.7, 56.9, 56.7, 56.4
    ],
    "재배면적": [
        1156885, 1135812, 1102608, 1055886, 1049556, 1052395, 1058927, 1066203, 1072363, 1083125, 1053186, 1016030,
        1001159, 979717, 955229, 950250, 935766, 924471, 892074, 853823, 849172, 832625, 815506, 799344, 778734,
        754713, 737673, 729814, 726432, 732477, 727054, 708012
    ],
    "생산량": [
        5330826, 4749562, 5059764, 4694956, 5322962, 5449561, 5096879, 5262700, 5290771, 5514796, 4926746, 4451135,
        5000149, 4768368, 4679991, 4407743, 4843478, 4916080, 4295413, 4224019, 4006185, 4230011, 4240739, 4326915,
        4196691, 3972468, 3868045, 3744450, 3506578, 3881601, 3763700, 3702239
    ]
}

# 데이터프레임 생성
df = pd.DataFrame(데이터)

# ARIMA 모델을 위한 소비량 데이터 추출
소비량 = df["소비량"]

# ARIMA 모델 생성 및 학습
# (p, d, q) 값은 기본적으로 (1, 1, 0)으로 설정하여 단순화
모델 = ARIMA(소비량, order=(2, 1, 0))
모델_결과 = 모델.fit()

# 향후 5개년 (2025~2029년) 예측
예측_연도 = list(range(2025, 2030))
예측_소비량 = 모델_결과.forecast(steps=5)

# 예측 결과 출력
print("\n2025~2029년 예측 소비량:")
for 연도, 소비 in zip(예측_연도, 예측_소비량):
    print(f"{연도}년 예측 소비량: {소비:.2f} kg")

# 실제 소비량과 예측 소비량을 시각화
plt.figure(figsize=(10, 6))
plt.plot(df["연도"], 소비량, label="실제 소비량", marker="o")  # 실제 데이터
plt.plot(예측_연도, 예측_소비량, label="예측 소비량", marker="x", linestyle="--", color="orange")  # 예측 데이터
plt.title("1인당 연간 쌀 소비량 예측 (2025~2029년)")
plt.xlabel("연도")
plt.ylabel("1인당 소비량 (kg)")
plt.legend()
plt.grid()
plt.show()

# 잔차 계산 및 시각화
잔차 = 모델_결과.resid
plt.figure(figsize=(10, 6))
plt.plot(잔차, label="잔차")
plt.axhline(0, linestyle="--", color="red")
plt.title("ARIMA 모델 잔차 분석")
plt.xlabel("시간")
plt.ylabel("잔차")
plt.legend()
plt.grid()
plt.show()

# 잔차 분포 확인 (정규성 검정)
import seaborn as sns
sns.histplot(잔차, kde=True)
plt.title("잔차의 분포")
plt.show()

이상치_인덱스 = np.where(잔차 > 100)
print("극단적 잔차 발생 연도:", df.iloc[이상치_인덱스]["연도"].values)
# -> 1992년 모델 초기화와 관련이 있을 가능성

from statsmodels.graphics.tsaplots import plot_acf
plot_acf(모델_결과.resid, lags=20)
plt.title("잔차의 자기상관 분석")
plt.show()
# 첫 번째 시차에서의 높은 자기상관
#  ARIMA 모델의 order=(1, 1, 0)이 사용되었으므로, p와 q 값을 조정하여 모델을 개선
# order=(2, 1, 0) 또는 order=(1, 1, 1)을 시도

print(f"AIC: {모델_결과.aic}")
print(f"BIC: {모델_결과.bic}")

# order=(1, 1, 0)의 경우
# AIC: 93.69459503288292
# BIC: 96.56256944185321
# 이 값이 더 낮은 조합 찾아야 함

# 모델 (2,1,0)으로 수정 
# AIC: 89.72602801721534
# BIC: 94.02798963067079
# 더 낮아짐

# 모델 (2,1,1)로 수정
# AIC: 90.57081006148275
# BIC: 96.30675887942333

# (2,1,0)이 가장 적합
# 초기(1992년)에 큰 잔차가 발생했지만, 이후 잔차가 0 근처에서 안정적인 패턴
# 잔차의 대부분이 0 근처에 몰려 있습니다.
# 오른쪽 끝에 극단적인 값이 하나 존재하지만, 이는 모델 초기화 또는 데이터의 이상치에 의한 영향일 가능성이 높습니다.
# 전반적으로 정규분포를 따르지 않지만, 큰 왜곡은 보이지 않습니다.
# 첫 번째 시차에서 높은 자기상관 계수(약 1)를 보이지만, 이는 데이터의 초기화 과정에서 발생하는 일반적인 현상
# 잔차 자기상관이 거의 없으므로 모델이 적합하게 학습되었음을 의미
# (2,1,0) 모델은 AIC와 BIC 기준에서 최적의 성능을 보였으며, 잔차가 랜덤하게 분포하고 자기상관이 적으므로 데이터에 잘 적합한 모델로 판단

######################################
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.api import VAR
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 데이터 준비
데이터 = {
    "연도": list(range(1992, 2024)),
    "총인구수": [
        44503200, 45001113, 45416339, 45858029, 46266256, 46684069, 46991171, 47335678, 47732558, 48021543, 48229948,
        48386823, 48583805, 48782274, 48991779, 49268928, 49540367, 49773145, 50515666, 50734284, 50948272, 51141463,
        51327916, 51529338, 51696216, 51778544, 51826059, 51849861, 51829023, 51638809, 51439038, 51325329
    ],
    "소비량": [
        112.9, 110.2, 108.3, 106.5, 104.9, 102.4, 99.2, 96.9, 93.6, 88.9, 87.0, 83.2, 82.0, 80.7, 78.8, 76.9, 75.8,
        74.0, 72.8, 71.2, 69.8, 67.2, 65.1, 62.9, 61.9, 61.8, 61.0, 59.2, 57.7, 56.9, 56.7, 56.4
    ],
    "재배면적": [
        1156885, 1135812, 1102608, 1055886, 1049556, 1052395, 1058927, 1066203, 1072363, 1083125, 1053186, 1016030,
        1001159, 979717, 955229, 950250, 935766, 924471, 892074, 853823, 849172, 832625, 815506, 799344, 778734,
        754713, 737673, 729814, 726432, 732477, 727054, 708012
    ],
    "생산량": [
        5330826, 4749562, 5059764, 4694956, 5322962, 5449561, 5096879, 5262700, 5290771, 5514796, 4926746, 4451135,
        5000149, 4768368, 4679991, 4407743, 4843478, 4916080, 4295413, 4224019, 4006185, 4230011, 4240739, 4326915,
        4196691, 3972468, 3868045, 3744450, 3506578, 3881601, 3763700, 3702239
    ]
}

# 데이터프레임 생성
df = pd.DataFrame(데이터)

# 스케일링 (최소-최대 스케일링)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df.drop(columns=["연도"]))
scaled_df = pd.DataFrame(scaled_data, columns=["총인구수", "소비량", "재배면적", "생산량"], index=df["연도"])

# 차분 데이터 생성 (VAR 모델용)
diff_df = scaled_df.diff().dropna()

# AIC/BIC 기반 최적 시차 탐색 (maxlags=5 이하로 제한)
model = VAR(diff_df)
results = model.select_order(maxlags=5)  # maxlags 제한
optimal_lag = results.selected_orders['aic']
print(f"최적 시차 (AIC 기준): {optimal_lag}")

# 최적 시차로 VAR 모델 학습
var_results = model.fit(maxlags=optimal_lag)

# VAR 모델 예측
forecast_steps = 5
forecast_diff = var_results.forecast(diff_df.values[-var_results.k_ar:], steps=forecast_steps)
forecast_diff_cumsum = forecast_diff.cumsum(axis=0)
forecast_original = scaler.inverse_transform(forecast_diff_cumsum)

# 예측 결과 출력
forecast_years = list(range(2025, 2030))
forecast_result = pd.DataFrame(forecast_original, columns=["총인구수", "소비량", "재배면적", "생산량"], index=forecast_years)

print("VAR 모델 예측 결과:")
print(forecast_result)

# 소비량 시각화 (VAR)
plt.figure(figsize=(12, 6))
plt.plot(df["연도"], df["소비량"], label="실제 소비량", marker="o")
plt.plot(forecast_years, forecast_result["소비량"], label="VAR 예측 소비량", marker="x", linestyle="--", color="orange")
plt.title("1인당 연간 쌀 소비량 예측 (VAR)")
plt.xlabel("연도")
plt.ylabel("1인당 소비량 (kg)")
plt.legend()
plt.grid()
plt.show()

#총인구수:
#2025년 약 44.39백만 명에서 2029년 약 41.86백만 명으로 지속적으로 감소.
#과거 데이터의 추세를 적절히 반영.
#소비량:
#2025년 59.63kg에서 2029년 69.72kg로 증가.
#과거 소비량의 감소 추세와는 다른 패턴을 보입니다. 이는 VAR 모델이 최근 변동성을 과도하게 반영했거나 데이터 특성을 제대로 학습하지 못했음을 시사합니다.
#재배면적:
#2025년 약 641,372ha에서 2029년 약 529,425ha로 감소.
#이는 재배면적의 하락 추세와 일관된 결과.
#생산량:
#2025년 약 2.53백만 톤에서 2029년 약 2.11백만 톤으로 감소.
#생산량은 재배면적 감소와 연관된 자연스러운 결과.
#문제점
#소비량 예측의 증가세:
#실제 데이터에서 소비량은 지속적으로 감소했지만, 모델은 2025년 이후 증가세를 보였습니다.
#이는 최근 데이터의 변화나 변동성을 과도하게 반영했을 가능성이 큽니다.
#데이터 안정성:
#차분 데이터를 사용했음에도 불구하고 일부 변수에서 장기적인 안정성이 부족할 수 있습니다.

# LSTM 모델로는 데이터수가 모자라서 제대로 예측이 불가능함

############################################################################

# LSTM 모델로 예측한 생산량 값 사용해서 VAR 다시 해보기
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt

# 기존 데이터 준비
데이터 = {
    "연도": list(range(1992, 2024)),
    "소비량": [
        112.9, 110.2, 108.3, 106.5, 104.9, 102.4, 99.2, 96.9, 93.6, 88.9, 87.0, 83.2, 82.0, 80.7, 78.8, 76.9, 75.8,
        74.0, 72.8, 71.2, 69.8, 67.2, 65.1, 62.9, 61.9, 61.8, 61.0, 59.2, 57.7, 56.9, 56.7, 56.4
    ],
    "생산량": [
        5330826, 4749562, 5059764, 4694956, 5322962, 5449561, 5096879, 5262700, 5290771, 5514796, 4926746, 4451135,
        5000149, 4768368, 4679991, 4407743, 4843478, 4916080, 4295413, 4224019, 4006185, 4230011, 4240739, 4326915,
        4196691, 3972468, 3868045, 3744450, 3506578, 3881601, 3763700, 3702239
    ]
}

# LSTM 생산량 예측 결과 추가 (2025~2029)
추가_생산량 = [3698139.75, 3677770.00, 3697611.25, 3772737.25, 3781820.25]
추가_연도 = list(range(2025, 2030))

# 기존 데이터프레임 생성
df = pd.DataFrame(데이터)

# 2025~2029년 데이터 보강
추가_df = pd.DataFrame({
    "연도": 추가_연도,
    "소비량": [None] * len(추가_연도),  # 소비량은 예측 대상
    "생산량": 추가_생산량
})
df = pd.concat([df, 추가_df], ignore_index=True)

# 스케일링 (최소-최대 스케일링)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df.drop(columns=["연도"]))
scaled_df = pd.DataFrame(scaled_data, columns=["소비량", "생산량"], index=df["연도"])

# 차분 데이터 생성 (VAR 모델용)
diff_df = scaled_df.diff().dropna()

# AIC/BIC 기반 최적 시차 탐색 (maxlags=5 이하로 제한)
model = VAR(diff_df)
results = model.select_order(maxlags=5)  # maxlags 제한
optimal_lag = results.selected_orders['aic']
print(f"최적 시차 (AIC 기준): {optimal_lag}")

# 최적 시차로 VAR 모델 학습
var_results = model.fit(maxlags=optimal_lag)

# VAR 모델 예측 (2025~2029 소비량 예측)
forecast_steps = len(추가_연도)
forecast_diff = var_results.forecast(diff_df.values[-var_results.k_ar:], steps=forecast_steps)
forecast_diff_cumsum = forecast_diff.cumsum(axis=0)
forecast_original = scaler.inverse_transform(forecast_diff_cumsum)

# 예측 결과 출력
forecast_result = pd.DataFrame(forecast_original, columns=["소비량", "생산량"], index=추가_연도)

print("VAR 모델 소비량 예측 결과:")
print(forecast_result["소비량"])

# 소비량 시각화
plt.figure(figsize=(12, 6))
plt.plot(df["연도"][:len(데이터["소비량"])], df["소비량"][:len(데이터["소비량"])], label="실제 소비량", marker="o")
plt.plot(추가_연도, forecast_result["소비량"], label="VAR 예측 소비량", marker="x", linestyle="--", color="orange")
plt.title("1인당 연간 쌀 소비량 예측 (VAR)")
plt.xlabel("연도")
plt.ylabel("1인당 소비량 (kg)")
plt.legend()
plt.show()


#첫 번째 VAR 모델
#특징:

#기존 데이터만 사용하여 소비량, 생산량, 총인구수, 재배면적 등 다변량 시계열 데이터를 포함하여 소비량을 예측.
#모델이 소비량의 증가 추세를 예측하였으며, 이는 과거 데이터의 하락 추세와는 반대.
#과도한 단기 변동성을 반영했을 가능성이 큼.
#문제점:

#과거 소비량 추세와 일치하지 않는 증가 예측.
#과도한 변수 사용으로 인해 오버피팅 위험이 존재.
#2. 두 번째 VAR 모델 (LSTM 생산량 포함)
#특징:

#LSTM 모델을 사용해 생산량을 예측하여 추가 데이터로 활용.
#기존 데이터의 소비량 및 예측된 생산량만 사용하여 간결한 구조로 소비량을 예측.
#소비량의 하락 추세를 유지하면서 더 현실적인 결과를 제공.
#장점:

#과거 소비량의 감소 추세와 일치.
#과도한 변수 사용을 줄이고, 예측의 초점이 명확함.
#단점:

#LSTM의 생산량 예측이 정확하지 않을 경우, VAR 모델 성능에 부정적 영향을 줄 수 있음.

#두 번째 VAR 모델이 더 적합합니다:

#이유:
#소비량의 과거 추세(감소)를 잘 반영하며, 단순한 모델 구조로 인해 과적합 가능성이 낮음.
#LSTM의 생산량 예측 데이터를 포함하여 미래 생산량 변동을 반영하려는 시도가 긍정적.

# ARIMA랑 결과값이 비슷함

# 잔차 검정 (White Noise Test)
from statsmodels.stats.diagnostic import acorr_ljungbox

# 잔차 가져오기
residuals = var_results.resid

# 각 변수의 잔차 검정
for col in residuals.columns:
    print(f"### {col} 잔차 검정 ###")
    lb_test = acorr_ljungbox(residuals[col], lags=[10], return_df=True)
    print(lb_test)

# 루트 안정성 검증
stability = var_results.is_stable()
print(f"VAR 모델의 안정성: {'안정적' if stability else '불안정'}")

# 특성 근 시각화
import matplotlib.pyplot as plt

eigvals = var_results.roots
plt.figure(figsize=(6, 6))
plt.title("VAR 모델 특성 근 분포")
plt.scatter(eigvals.real, eigvals.imag, color='blue', marker='o', label="특성 근")
plt.axhline(y=0, color='k', linestyle='--')
plt.axvline(x=0, color='k', linestyle='--')
plt.gca().add_patch(plt.Circle((0, 0), 1.0, color='r', fill=False, linestyle='--'))
plt.xlabel("실수부")
plt.ylabel("허수부")
plt.legend()
plt.grid()
plt.show()

# 잔차 검정을 통해 소비량과 생산량의 잔차가 백색잡음임을 확인하였으며, 특성 근 분석 결과 모델의 안정성이 입증됨. 