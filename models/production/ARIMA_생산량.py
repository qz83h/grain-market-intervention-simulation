import warnings
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# 경고 제거
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 정리 함수
def preprocess_population_data(file_path):
    """인구 데이터를 정리하고 시계열 형태로 반환"""
    df = pd.read_csv(file_path, header=None, names=['연도', '총인구수'])
    df['총인구수'] = df['총인구수'].str.replace(',', '').astype(int)  # 쉼표 제거 및 숫자 변환
    df['연도'] = pd.to_datetime(df['연도'], format='%Y')  # 연도를 시계열로 변환
    df.set_index('연도', inplace=True)  # DatetimeIndex 설정
    return df

# ARIMA 모델 학습 및 예측
def fit_arima_model(data, order, title, forecast_years=5):
    """ARIMA 모델을 학습하고 예측"""
    # 데이터 확인
    if data.empty or data.ndim != 1:
        raise ValueError(f"{title} 데이터가 비어 있거나 다차원 배열입니다.")

    # ARIMA 모델 생성 및 학습
    model = ARIMA(data, order=order)
    results = model.fit()

    # 모델 평가
    print(results.summary())

    # 예측 수행
    forecast = results.forecast(steps=forecast_years)
    forecast_index = pd.date_range(start=data.index[-1], periods=forecast_years+1, freq='Y')[1:]
    conf_int = results.get_forecast(steps=forecast_years).conf_int()

    # 예측 시각화
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data.values, label='실제값')
    plt.plot(forecast_index, forecast, 'r--', label='예측값')
    plt.fill_between(forecast_index,
                     conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                     color='r', alpha=0.1)
    plt.title(f'{title} 예측 결과 (ARIMA)')
    plt.xlabel('연도')
    plt.ylabel(title)
    plt.legend()
    plt.grid(True)
    plt.show()

    return forecast, conf_int

# 최적 파라미터 탐색 함수
def find_best_arima_params(data, title):
    """최적의 ARIMA 파라미터를 탐색"""
    best_aic = float("inf")
    best_order = None
    for p in range(0, 3):
        for d in range(0, 3):
            for q in range(0, 3):
                try:
                    model = ARIMA(data, order=(p, d, q))
                    results = model.fit()
                    if results.aic < best_aic:
                        best_aic = results.aic
                        best_order = (p, d, q)
                except Exception as e:
                    continue
    print(f"{title} 데이터의 최적 파라미터: {best_order}, AIC: {best_aic}")
    return best_order

# 메인 실행 함수
def main():
    # 파일 경로 설정
    population_file_path = r"C:\\Users\\Owlet\\Desktop\\P프로젝트\\총인구수(1992-2023).csv"

    # 데이터 정리
    population_df = preprocess_population_data(population_file_path)

    # 생산량 및 소비량 데이터 생성
    production_data = {
        "연도": list(range(2010, 2024)),
        "생산량": [4295413, 4224019, 4006185, 4230011, 4240739, 4326915, 
                   4196691, 3972468, 3868045, 3744450, 3506578, 3881601, 
                   3763700, 3702239]
    }
    consumption_data = {
        "연도": list(range(1992, 2024)),
        "소비량": [112.9, 110.2, 108.3, 106.5, 104.9, 102.4, 99.2, 96.9, 
                   93.6, 88.9, 87.0, 83.2, 82.0, 80.7, 78.8, 76.9, 75.8, 
                   74.0, 72.8, 71.2, 69.8, 67.2, 65.1, 62.9, 61.9, 61.8, 
                   61.0, 59.2, 57.7, 56.9, 56.7, 56.4]
    }

    prod_df = pd.DataFrame(production_data)
    cons_df = pd.DataFrame(consumption_data)
    prod_df['연도'] = pd.to_datetime(prod_df['연도'], format='%Y')
    cons_df['연도'] = pd.to_datetime(cons_df['연도'], format='%Y')
    prod_df.set_index('연도', inplace=True)
    cons_df.set_index('연도', inplace=True)

    # 최적 파라미터 탐색 및 설정
    print("\n=== 생산량 데이터 최적 파라미터 탐색 ===")
    prod_best_order = find_best_arima_params(prod_df['생산량'], title='생산량')

    print("\n=== 소비량 데이터 최적 파라미터 탐색 ===")
    cons_best_order = find_best_arima_params(cons_df['소비량'], title='소비량')

    # 생산량 예측
    print("\n=== 생산량 예측 ===")
    prod_forecast, prod_conf = fit_arima_model(
        prod_df['생산량'], 
        order=prod_best_order, 
        title='생산량', 
        forecast_years=5
    )

    # 소비량 예측
    print("\n=== 소비량 예측 ===")
    cons_forecast, cons_conf = fit_arima_model(
        cons_df['소비량'], 
        order=cons_best_order, 
        title='소비량', 
        forecast_years=5
    )

    # 예측 결과 출력
    years = list(range(2024, 2029))
    print("\n=== 향후 5년 예측 결과 ===")
    print("\n생산량 예측:")
    for year, pred, (lower, upper) in zip(years, prod_forecast, prod_conf.values):
        print(f"{year}: {pred:,.0f} ({lower:,.0f} - {upper:,.0f})")
    
    print("\n소비량 예측:")
    for year, pred, (lower, upper) in zip(years, cons_forecast, cons_conf.values):
        print(f"{year}: {pred:.1f} ({lower:.1f} - {upper:.1f})")

if __name__ == "__main__":
    main()