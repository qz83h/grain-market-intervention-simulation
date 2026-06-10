import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import platform
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')
else:
    plt.rc('font', family='NanumGothic')

plt.rc('axes', unicode_minus=False)

def create_initial_dataset():
    """초기 데이터셋 생성"""
    market_data = pd.DataFrame({
        'year': range(2014, 2025),
        'expected_production': [4180000, 4260000, 4022000, 3955000, 3875000, 3780000, 
                              3631000, 3830000, 3857000, 3682000, 3657000],
        'actual_production': [4241000, 4320000, 4197000, 3972000, 3868000, 3744000, 
                            3507000, 3882000, 3764000, 3702000, 3585000],
        'market_isolation': [240000, 200000, 299000, 370000, 0, 0, 0, 200000, 
                           370000, 0, 200000],
        'excess_production': [180000, 350000, 299000, 250000, 0, 0, 0, 270000,
                            155000, 95000, 56000],
        'consumption_per_capita': [65.1, 62.9, 61.9, 61.8, 61.0, 59.2, 
                                 57.7, 56.9, 56.7, 56.4, 55.8]
    })
    
    market_data['isolation_flag'] = np.where(market_data['market_isolation'] > 0, 1, 0)
    
    return market_data

def create_additional_features(data):
    """시계열 특성을 반영한 추가 피처 생성"""
    df = data.copy()
    
    df['Q_G'] = df['actual_production'] - df['market_isolation']
    df['I'] = np.where(df['actual_production'] < df['expected_production'], 1, 0)
    df['prod_ma3'] = df['actual_production'].rolling(window=3, min_periods=1).mean()
    df['prod_change'] = df['actual_production'].pct_change().fillna(0)
    df['consumption_change'] = df['consumption_per_capita'].pct_change().fillna(0)
    df['prod_cons_interaction'] = df['actual_production'] * df['consumption_per_capita']
    df['excess_prod_ratio'] = df['excess_production'] / df['actual_production']
    df['isolation_ratio'] = df['market_isolation'] / df['excess_production'].replace(0, 1)
    
    return df

def extend_dataset(initial_data, future_consumption, future_production):
    """데이터셋 미래 기간 확장"""
    future_years = pd.DataFrame({
        'year': range(2025, 2030),
        'consumption_per_capita': future_consumption,
        'expected_production': future_production,
        'actual_production': future_production,
        'market_isolation': np.nan,
        'excess_production': np.zeros(5),
        'isolation_flag': np.zeros(5)
    })
    
    extended_data = pd.concat([initial_data, future_years], ignore_index=True)
    extended_data = create_additional_features(extended_data)
    
    return extended_data

def plot_full_period_forecast(historical_data, future_data, model, features):
    """전체 기간에 대한 예측 및 시각화"""
    plt.figure(figsize=(15, 8))
    
    full_data = pd.concat([historical_data, future_data], ignore_index=True)
    X_full = full_data[features]
    
    full_predictions = model.predict(X_full)
    
    plt.plot(full_data['year'], full_data['market_isolation'], 
            'b-', label='실제값', linewidth=2)
    plt.plot(full_data['year'], full_predictions, 
            'r--', label='예측값', linewidth=2)
    
    plt.title('연도별 시장격리량 추이', fontsize=14)
    plt.xlabel('연도', fontsize=12)
    plt.ylabel('시장격리량 (톤)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    return full_predictions

def evaluate_model_performance(y_true, y_pred):
    """모델 성능 평가"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # 수정된 MAPE 계산
    non_zero_mask = y_true != 0
    if np.any(non_zero_mask):
        mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / 
                              y_true[non_zero_mask])) * 100
    else:
        mape = np.nan
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape
    }

def create_enhanced_model():
    """향상된 RandomForest 모델"""
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=4,
        min_samples_split=3,
        min_samples_leaf=2,
        random_state=42
    )

def main():
    # 미래 데이터 준비
    future_consumption = [55.726390, 54.835064, 53.709801, 52.461647, 51.112121]
    future_production = [3713416.00, 3710953.50, 3735898.25, 3767419.75, 3776377.75]
    
    # 데이터셋 생성
    initial_data = create_initial_dataset()
    extended_data = extend_dataset(initial_data, future_consumption, future_production)
    
    # 피처 선택
    features = ['actual_production', 'consumption_per_capita', 'I', 'Q_G',
                'prod_ma3', 'prod_change', 'consumption_change', 'prod_cons_interaction',
                'excess_prod_ratio', 'isolation_ratio', 'isolation_flag']
    
    # 데이터 분할
    X_historical = extended_data[extended_data['year'] < 2025][features]
    y_historical = extended_data[extended_data['year'] < 2025]['market_isolation']
    
    # 모델 학습
    rf_model = create_enhanced_model()
    rf_model.fit(X_historical, y_historical)
    
    # 예측 및 시각화
    full_predictions = plot_full_period_forecast(
        extended_data[extended_data['year'] < 2025],
        extended_data[extended_data['year'] >= 2025],
        rf_model,
        features
    )
    
    # 성능 평가
    historical_predictions = full_predictions[:len(y_historical)]
    metrics = evaluate_model_performance(y_historical, historical_predictions)
    
    print("\n모델 성능 평가 지표:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # 미래 예측값 출력
    future_predictions = full_predictions[len(y_historical):]
    print("\n미래 기간 예측값:")
    for year, pred in zip(range(2025, 2030), future_predictions):
        print(f"{year}년: {pred:.0f}톤")
    
    # 변수 중요도 분석
    importance = pd.DataFrame({
        'feature': features,
        'importance': rf_model.feature_importances_
    })
    print("\n변수 중요도 분석:")
    print(importance.sort_values('importance', ascending=False))

if __name__ == "__main__":
    main()
