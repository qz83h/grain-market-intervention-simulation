import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 한글 폰트 설정
plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)

# 1. CSV 파일 불러오기
file_path = r"C:\Users\exist\OneDrive\바탕 화면\시장격리\가격.csv"
prices_df = pd.read_csv(file_path, encoding='cp949', parse_dates=['날짜'])

# 2. 산지쌀값, 소매가 분리 함수
def get_producer_price(df):
    return df[['날짜', '산지쌀값']].copy()

def get_retail_price(df):
    return df[['날짜', '소매가']].copy()

producer_price_df = get_producer_price(prices_df)
retail_price_df = get_retail_price(prices_df)

# 3. 평년 수확기 쌀값 계산 함수 (데이터 부족한 경우 보완 로직 포함)
def calculate_average_past_prices(price_df, year):
    """
    특정 연도의 평년 수확기 쌀값을 계산합니다.
    데이터가 부족한 경우:
    - 2014년: 해당 연도의 평균값 사용
    - 2015년: 2014년의 평균값 사용
    - 2016년: 2014년, 2015년의 평균값 사용
    """
    past_prices = price_df[
        (price_df['날짜'].dt.year < year) & 
        (price_df['날짜'].dt.year >= max(year - 3, price_df['날짜'].dt.year.min())) & 
        (price_df['날짜'].dt.month.isin([10, 11]))
    ]

    if not past_prices.empty:
        return past_prices['산지쌀값'].mean()
    
    # 데이터 부족한 경우 보완 로직
    if year == 2014:
        # 2014년: 해당 연도의 10~11월 평균값 사용
        current_year_prices = price_df[
            (price_df['날짜'].dt.year == year) & 
            (price_df['날짜'].dt.month.isin([10, 11]))
        ]
        return current_year_prices['산지쌀값'].mean()
    elif year == 2015:
        # 2015년: 2014년의 평균값 사용
        return calculate_average_past_prices(price_df, 2014)
    elif year == 2016:
        # 2016년: 2014년, 2015년의 평균값 사용
        avg_2014 = calculate_average_past_prices(price_df, 2014)
        avg_2015 = calculate_average_past_prices(price_df, 2015)
        return np.mean([avg_2014, avg_2015])
    else:
        return np.nan  # 다른 경우는 nan 반환

# 3-1. 각 연도별 평년 수확기 쌀값 출력
def print_average_past_prices(price_df, years):
    """
    각 연도별 평년 수확기 쌀값을 출력합니다.
    """
    print("\n각 연도별 평년 수확기 쌀값:")
    for year in years:
        avg_price = calculate_average_past_prices(price_df, year)
        if np.isnan(avg_price):
            print(f"{year}: 평년 수확기 쌀값 데이터 부족")
        else:
            print(f"{year}: {avg_price:.2f} 원")

# 평년 수확기 쌀값 계산 함수 정의 (위 코드 참조)

# 1차 및 2차 시장격리 데이터 추가
market_df = pd.DataFrame({
    "연도": [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    "예상생산량": [4180000, 4260000, 4022000, 3955000, 3875000, 3780000, 3631000, 3830000, 3857000, 3682000, 3657000],
    "실제생산량": [4241000, 4320000, 4197000, 3972000, 3868000, 3744000, 3507000, 3882000, 3764000, 3702000, 3585000],
    "초과생산량": [180000, 350000, 299000, 250000, 0, 0, 0, 270000, 155000, 95000, 56000],
    "시장격리여부": [1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1],
    "1차시장격리매입중량": [180000, 200000, 299000, 370000, 0, 0, 0, 200000, 370000, 0, 105000]
})

# 평년 수확기 쌀값 계산 및 시장격리 데이터프레임에 추가
market_df['평년수확기쌀값'] = market_df['연도'].apply(lambda year: calculate_average_past_prices(producer_price_df, year))

# 각 연도별 평년 수확기 쌀값 출력
print_average_past_prices(producer_price_df, market_df['연도'])


# 4. 시장격리 판단 함수
def check_market_isolation(row, price_df):
    초과생산률 = (row['초과생산량'] / row['예상생산량']) * 100
    harvest_prices = price_df[
        (price_df['날짜'].dt.year == row['연도']) & 
        (price_df['날짜'].dt.month.isin([10, 11]))
    ]
    수확기_산지쌀값 = harvest_prices['산지쌀값'].mean()
    평년대비_하락률 = ((row['평년수확기쌀값'] - 수확기_산지쌀값) / row['평년수확기쌀값']) * 100
    
    if 초과생산률 >= 3 or 평년대비_하락률 >= 3:
        return "시장격리 필요", 수확기_산지쌀값
    else:
        return "시장격리 불필요", 수확기_산지쌀값

# 5. 시장격리 데이터프레임 설정
market_isolation = {
    "연도": [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    "예상생산량": [4180000, 4260000, 4022000, 3955000, 3875000, 3780000, 3631000, 3830000, 3857000, 3682000, 3657000],
    "실제생산량": [4241000, 4320000, 4197000, 3972000, 3868000, 3744000, 3507000, 3882000, 3764000, 3702000, 3585000],
    "초과생산량": [180000, 350000, 299000, 250000, 0, 0, 0, 270000, 155000, 95000, 56000],
    "시장격리여부": [1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1],
    "1차시장격리매입중량": [180000, 200000, 299000, 370000, 0, 0, 0, 200000, 370000, 0, 105000]
}
market_df = pd.DataFrame(market_isolation)

# 평년 수확기 쌀값 계산
market_df['평년수확기쌀값'] = market_df['연도'].apply(lambda year: calculate_average_past_prices(producer_price_df, year))

# 시장격리 판단 및 수확기 산지쌀값 추가
market_df['시장격리판단'], market_df['수확기산지쌀값'] = zip(
    *market_df.apply(lambda row: check_market_isolation(row, producer_price_df), axis=1)
)

# 6. 1차 및 2차 매입 시점 추가
market_df['1차 매입 시점'] = pd.to_datetime(
    ["2014-10-15", "2015-10-15", "2016-10-15", "2017-10-15", None, None, None, "2022-01-05", "2022-10-15", None, "2024-09-25"]
)
market_df['2차 매입 시점'] = pd.to_datetime(
    ["2014-11-25", None, None, None, None, None, None, None, None, None, "2024-11-15"]
)

# 7. 시장격리 필요 여부 출력 함수
def display_market_isolation_results(df):
    """
    시장격리 판단 결과를 보기 쉽게 표 형식으로 출력합니다.
    """
    display_df = df.copy()
    
    # 매입 중량 없는 경우 '없음' 표시
    display_df['1차시장격리매입중량'] = display_df['1차시장격리매입중량'].apply(lambda x: f"{x:,}" if x > 0 else "없음")
    
    # 1차 및 2차 매입 시점 포맷팅
    display_df['1차 매입 시점'] = display_df['1차 매입 시점'].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else "없음")
    display_df['2차 매입 시점'] = display_df['2차 매입 시점'].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else "없음")
    
    # 출력 컬럼 정리
    output_columns = [
        "연도", "예상생산량", "실제생산량", "초과생산량", "시장격리여부", 
        "1차시장격리매입중량", "평년수확기쌀값", "수확기산지쌀값", "시장격리판단", 
        "1차 매입 시점", "2차 매입 시점"
    ]
    print(display_df[output_columns].to_string(index=False))

# 결과 보기 좋은 형태로 출력
display_market_isolation_results(market_df)

# 8. 산지쌀값과 소매가격 비교 그래프
def plot_prices(producer_df, retail_df):
    plt.figure(figsize=(12, 6))
    plt.plot(producer_df['날짜'], producer_df['산지쌀값'], label="산지쌀값", color='blue')
    plt.plot(retail_df['날짜'], retail_df['소매가'], label="소매가", color='red')
    plt.title("산지쌀값과 소매가 비교")
    plt.xlabel("날짜")
    plt.ylabel("가격")
    plt.legend()
    plt.grid()
    plt.show()

plot_prices(producer_price_df, retail_price_df)

# 9. 연도별 시장격리 표시 그래프
# 시장격리 시행 연도와 미시행 연도 구분
def analyze_market_isolation(market_df, producer_price_df):
    """
    시장격리 시행 여부와 추가 매입 필요 여부를 분석하여 결과 출력.
    """
    print("\n[시장격리를 시행한 연도]")
    for _, row in market_df[market_df['시장격리여부'] == 1].iterrows():
        year = row['연도']
        평년가격 = row['평년수확기쌀값']
        yearly_data = producer_price_df[producer_price_df['날짜'].dt.year == year]
        
        # 1차 매입 이후 분석
        first_date = row['1차 매입 시점']
        second_date = row['2차 매입 시점']
        if pd.notna(first_date):
            post_first = yearly_data[yearly_data['날짜'] > first_date]
            if not post_first.empty and any(post_first['산지쌀값'] < 평년가격 * 0.95):
                추가_매입_시점 = post_first[post_first['산지쌀값'] < 평년가격 * 0.95].iloc[0]['날짜']
                print(f"{year}년 (1차 매입 이후): 추가 매입 시점 추천 - {추가_매입_시점.date()}")
            else:
                print(f"{year}년 (1차 매입 이후): 추가 매입 필요 없음")
        
        # 2차 매입 이후 분석
        if pd.notna(second_date):
            post_second = yearly_data[yearly_data['날짜'] > second_date]
            if not post_second.empty and any(post_second['산지쌀값'] < 평년가격 * 0.95):
                추가_매입_시점 = post_second[post_second['산지쌀값'] < 평년가격 * 0.95].iloc[0]['날짜']
                print(f"{year}년 (2차 매입 이후): 추가 매입 시점 추천 - {추가_매입_시점.date()}")
            else:
                print(f"{year}년 (2차 매입 이후): 추가 매입 필요 없음")
    
    print("\n[시장격리를 시행하지 않은 연도]")
    for _, row in market_df[market_df['시장격리여부'] == 0].iterrows():
        year = row['연도']
        평년가격 = row['평년수확기쌀값']
        yearly_data = producer_price_df[producer_price_df['날짜'].dt.year == year]
        
        if not yearly_data.empty and any(yearly_data['산지쌀값'] < 평년가격 * 0.95):
            추천_시점 = yearly_data[yearly_data['산지쌀값'] < 평년가격 * 0.95].iloc[0]['날짜']
            print(f"{year}년: 시장격리 매입 시점 추천 - {추천_시점.date()}")
        else:
            print(f"{year}년: 시장 격리 필요 없음")

# 결과 분석 및 출력
analyze_market_isolation(market_df, producer_price_df)

def plot_market_isolation_with_recommendation(producer_df, market_df, recommendations):
    """
    각 연도별 산지쌀값 그래프를 그리고 1차/2차 시장격리 시점과 추천 매입 시점을 표시합니다.
    """
    for year in market_df['연도']:
        yearly_data = producer_df[producer_df['날짜'].dt.year == year]
        first_isolation_date = market_df.loc[market_df['연도'] == year, '1차 매입 시점'].values[0]
        second_isolation_date = market_df.loc[market_df['연도'] == year, '2차 매입 시점'].values[0]
        recommendation_date = recommendations.get(year, None)

        plt.figure(figsize=(10, 5))
        plt.plot(yearly_data['날짜'], yearly_data['산지쌀값'], label="산지쌀값", color='blue')
        
        if pd.notna(first_isolation_date):
            plt.axvline(pd.to_datetime(first_isolation_date), color='red', linestyle='--', label="1차 시장격리")
        
        if pd.notna(second_isolation_date):
            plt.axvline(pd.to_datetime(second_isolation_date), color='green', linestyle='--', label="2차 시장격리")
        
        if recommendation_date:
            plt.axvline(pd.to_datetime(recommendation_date), color='purple', linestyle=':', label="추천 매입 시점")
        
        plt.title(f"{year}년 산지쌀값 추이")
        plt.xlabel("날짜")
        plt.ylabel("산지쌀값")
        plt.legend()
        plt.grid()
        plt.show()

# 추천 매입 시점 도출 후 그래프에 추가
recommendations = {}
for _, row in market_df.iterrows():
    year = row['연도']
    평년가격 = row['평년수확기쌀값']
    yearly_data = producer_price_df[producer_price_df['날짜'].dt.year == year]
    
    # 시장격리 여부와 추가 매입 추천
    if row['시장격리여부'] == 1:
        first_date = row['1차 매입 시점']
        second_date = row['2차 매입 시점']
        if pd.notna(second_date):  # 2차 매입 이후 추천
            post_second = yearly_data[yearly_data['날짜'] > second_date]
            if not post_second.empty and any(post_second['산지쌀값'] < 평년가격 * 0.95):
                recommendations[year] = post_second[post_second['산지쌀값'] < 평년가격 * 0.95].iloc[0]['날짜']
        elif pd.notna(first_date):  # 1차 매입 이후 추천
            post_first = yearly_data[yearly_data['날짜'] > first_date]
            if not post_first.empty and any(post_first['산지쌀값'] < 평년가격 * 0.95):
                recommendations[year] = post_first[post_first['산지쌀값'] < 평년가격 * 0.95].iloc[0]['날짜']
    elif row['시장격리여부'] == 0:  # 시장격리 미시행 연도
        if not yearly_data.empty and any(yearly_data['산지쌀값'] < 평년가격 * 0.95):
            recommendations[year] = yearly_data[yearly_data['산지쌀값'] < 평년가격 * 0.95].iloc[0]['날짜']

# 그래프 출력
plot_market_isolation_with_recommendation(producer_price_df, market_df, recommendations)

def recommend_isolation_volume(row, price_df):
    """
    시장격리 격리량을 추천합니다.
    """
    초과생산량 = row['초과생산량']
    평년가격 = row['평년수확기쌀값']
    연도 = row['연도']
    수확기_산지쌀값 = price_df[
        (price_df['날짜'].dt.year == 연도) & 
        (price_df['날짜'].dt.month.isin([10, 11]))
    ]['산지쌀값'].mean()

    # 기본 추천 격리량: 초과생산량의 80%
    recommended_volume = 초과생산량 * 0.8

    # 가격 하락폭에 따른 추가 보정
    if 수확기_산지쌀값 < 평년가격 * 0.95:
        하락폭 = (평년가격 - 수확기_산지쌀값) / 평년가격
        recommended_volume *= (1 + 하락폭)  # 하락폭 비례 추가 격리량

    return round(recommended_volume, -3)  # 천 단위로 반올림

# 시장격리 격리량 추천 추가
market_df['추천격리량'] = market_df.apply(lambda row: recommend_isolation_volume(row, producer_price_df) if row['시장격리여부'] == 1 else "시장격리 미시행", axis=1)

# 결과 출력
print("\n[격리량 추천 결과]")
print(market_df[['연도', '초과생산량', '추천격리량']])


#####################
def evaluate_isolation_effectiveness(producer_df, market_df):
    """
    시장격리 정책의 효과를 검증합니다.
    - 격리 시행 전 3개월과 격리 시행 후 2개월의 가격 변화를 비교합니다.
    - 1차, 2차 시장격리가 모두 시행된 경우, 1차 매입 시점 이전 3개월과 2차 매입 시점 이후 2개월을 비교합니다.
    """
    results = []
    for _, row in market_df.iterrows():
        year = row['연도']
        first_isolation_date = row['1차 매입 시점']
        second_isolation_date = row['2차 매입 시점']

        if pd.notna(first_isolation_date):
            # 1차 격리 전 3개월 평균
            before_isolation = producer_df[
                (producer_df['날짜'] < first_isolation_date) &
                (producer_df['날짜'] >= first_isolation_date - pd.DateOffset(months=3)) &
                (producer_df['날짜'].dt.year == year)
            ]['산지쌀값']

            # 2차 격리 이후 2개월 평균 (있을 경우)
            if pd.notna(second_isolation_date):
                after_isolation = producer_df[
                    (producer_df['날짜'] >= second_isolation_date) &
                    (producer_df['날짜'] < second_isolation_date + pd.DateOffset(months=2)) &
                    (producer_df['날짜'].dt.year == year)
                ]['산지쌀값']
            else:  # 1차 격리 이후 2개월 평균 (2차 격리가 없을 경우)
                after_isolation = producer_df[
                    (producer_df['날짜'] >= first_isolation_date) &
                    (producer_df['날짜'] < first_isolation_date + pd.DateOffset(months=2)) &
                    (producer_df['날짜'].dt.year == year)
                ]['산지쌀값']

            # 평균 값 및 변화율 계산
            if not before_isolation.empty and not after_isolation.empty:
                before_avg = before_isolation.mean()
                after_avg = after_isolation.mean()
                change = (after_avg - before_avg) / before_avg * 100  # 변화율 계산
                results.append({
                    '연도': year,
                    '격리 전 평균 (3개월)': before_avg,
                    '격리 후 평균 (2개월)': after_avg,
                    '가격 변화율(%)': change
                })

    results_df = pd.DataFrame(results)
    print("\n[시장격리 정책 효과 검증]")
    print(results_df.to_string(index=False))
    return results_df

# 결과 분석 및 시각화
isolation_effect_df = evaluate_isolation_effectiveness(producer_price_df, market_df)
from statsmodels.tsa.arima.model import ARIMA

def simulate_price_with_recommended_timing(producer_df, recommendations, isolation_effect=0.03):
    """
    추천 매입 시기에 시장격리 정책이 시행되었을 때 가격 변화를 시뮬레이션합니다.
    - isolation_effect: 매입 정책으로 인한 예상 가격 상승률 (기본값 3%)
    """
    simulated_results = []
    
    for year, rec_date in recommendations.items():
        yearly_data = producer_df[producer_df['날짜'].dt.year == year]
        pre_isolation = yearly_data[yearly_data['날짜'] < rec_date]['산지쌀값']
        
        # ARIMA 모델로 격리 전 데이터 학습
        if not pre_isolation.empty:
            model = ARIMA(pre_isolation, order=(1, 1, 1))
            model_fit = model.fit()
            
            # 격리 시점 이후 30일 예측
            forecast = model_fit.forecast(steps=30)
            
            # 시장격리 정책 효과 적용 (가격 상승)
            adjusted_forecast = forecast * (1 + isolation_effect)
            
            simulated_results.append({
                '연도': year,
                '추천 매입 시점': rec_date,
                '격리 전 평균': pre_isolation.mean(),
                '격리 후 예상 평균': adjusted_forecast.mean(),
                '예상 가격 변화율(%)': ((adjusted_forecast.mean() - pre_isolation.mean()) / pre_isolation.mean()) * 100
            })
    
    simulated_df = pd.DataFrame(simulated_results)
    print("\n[추천 매입 시기 시뮬레이션 결과]")
    print(simulated_df.to_string(index=False))
    return simulated_df

# 추천 매입 시기 검증 실행
simulated_results_df = simulate_price_with_recommended_timing(producer_price_df, recommendations)

# 시각화
plt.figure(figsize=(10, 6))
plt.bar(simulated_results_df['연도'], simulated_results_df['예상 가격 변화율(%)'], color='lightgreen')
plt.axhline(y=0, color='gray', linestyle='--')
plt.title("추천 매입 시기 시뮬레이션 - 예상 가격 변화율")
plt.xlabel("연도")
plt.ylabel("예상 가격 변화율 (%)")
plt.grid()
plt.show()

def evaluate_government_decision(market_df, simulated_df, producer_df, target_price_change=3.0):
    """
    정부의 매입시점 판단 성공 여부를 검증합니다.
    - 격리 전 3개월 평균과 격리 후 2개월 평균을 사용합니다.
    - 1차, 2차 시장격리가 모두 시행된 경우, 1차 매입 시점 이전 3개월 평균과 2차 매입 시점 이후 2개월 평균을 비교합니다.
    - target_price_change: 목표 가격 상승률 (기본값 3%)
    """
    evaluation_results = []
    for _, row in market_df.iterrows():
        year = row['연도']
        first_isolation_date = row['1차 매입 시점']
        second_isolation_date = row['2차 매입 시점']

        if pd.notna(first_isolation_date):
            # 1차 매입 시점 이전 3개월 평균
            before_isolation = producer_df[
                (producer_df['날짜'] < first_isolation_date) &
                (producer_df['날짜'] >= first_isolation_date - pd.DateOffset(months=3)) &
                (producer_df['날짜'].dt.year == year)
            ]['산지쌀값']

            # 2차 매입 시점 이후 2개월 평균 (있을 경우)
            if pd.notna(second_isolation_date):
                after_isolation = producer_df[
                    (producer_df['날짜'] >= second_isolation_date) &
                    (producer_df['날짜'] < second_isolation_date + pd.DateOffset(months=2)) &
                    (producer_df['날짜'].dt.year == year)
                ]['산지쌀값']
            else:  # 1차 매입 시점 이후 2개월 평균 (2차 매입이 없을 경우)
                after_isolation = producer_df[
                    (producer_df['날짜'] >= first_isolation_date) &
                    (producer_df['날짜'] < first_isolation_date + pd.DateOffset(months=2)) &
                    (producer_df['날짜'].dt.year == year)
                ]['산지쌀값']

            # 평균 값 및 변화율 계산
            if not before_isolation.empty and not after_isolation.empty:
                before_avg = before_isolation.mean()
                after_avg = after_isolation.mean()
                price_change = (after_avg - before_avg) / before_avg * 100

                # 정부의 판단 성공 여부
                if price_change >= target_price_change:
                    result = "성공"
                else:
                    result = "실패"

                evaluation_results.append({
                    '연도': year,
                    '1차 매입 시점': first_isolation_date,
                    '2차 매입 시점': second_isolation_date if pd.notna(second_isolation_date) else "없음",
                    '격리 전 평균 (3개월)': before_avg,
                    '격리 후 평균 (2개월)': after_avg,
                    '가격 변화율(%)': price_change,
                    '목표 변화율(%)': target_price_change,
                    '판단': result
                })

    results_df = pd.DataFrame(evaluation_results)
    print("\n[정부의 매입시점 판단 성공 여부 검증]")
    print(results_df.to_string(index=False))
    return results_df

# 정부 판단 검증 실행
evaluation_df = evaluate_government_decision(market_df, simulated_results_df, producer_price_df)

# 시각화
plt.figure(figsize=(10, 6))
plt.bar(evaluation_df['연도'], evaluation_df['가격 변화율(%)'], color='orange', label='예상 가격 변화율')
plt.axhline(y=3.0, color='red', linestyle='--', label='목표 변화율 (3%)')
plt.title("정부 매입량 판단 검증 결과")
plt.xlabel("연도")
plt.ylabel("예상 가격 변화율 (%)")
plt.legend()
plt.grid()
plt.show()



#############################################

######################### 시장격리 시뮬레이션 모델

from sklearn.linear_model import LinearRegression

def calculate_k_value(market_df, producer_df):
    """
    과거 데이터로 격리량 증가율과 가격 변화율 간의 상관관계(k)를 계산합니다.
    """
    results = []
    
    for _, row in market_df.iterrows():
        year = row['연도']
        isolation_date = row['1차 매입 시점']
        isolation_volume = row['1차시장격리매입중량']
        
        # 격리 전후 평균 가격 계산
        before_isolation = producer_df[
            (producer_df['날짜'] < isolation_date) & 
            (producer_df['날짜'] >= isolation_date - pd.DateOffset(months=3))
        ]['산지쌀값']
        
        after_isolation = producer_df[
            (producer_df['날짜'] >= isolation_date) & 
            (producer_df['날짜'] < isolation_date + pd.DateOffset(months=2))
        ]['산지쌀값']
        
        if not before_isolation.empty and not after_isolation.empty and isolation_volume > 0:
            before_avg = before_isolation.mean()
            after_avg = after_isolation.mean()
            price_change_rate = ((after_avg - before_avg) / before_avg) * 100  # 가격 변화율
            
            # 격리량 증가율: 격리량 / 평균 생산량 비율
            production_avg = row['예상생산량']
            volume_increase_rate = (isolation_volume / production_avg) * 100
            
            results.append((volume_increase_rate, price_change_rate))
    
    # 데이터 준비
    if len(results) > 0:
        X = np.array([x[0] for x in results]).reshape(-1, 1)  # 격리량 증가율
        y = np.array([x[1] for x in results])                # 가격 변화율
        
        # 회귀 분석
        model = LinearRegression()
        model.fit(X, y)
        k = model.coef_[0]
        
        print(f"도출된 k 값 (격리량 증가율과 가격 변화율의 상관관계): {k:.4f}")
        return k
    else:
        print("충분한 데이터가 없어 k 값을 계산할 수 없습니다.")
        return None

# k 값 계산 실행
k_value = calculate_k_value(market_df, producer_price_df)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def simulate_market_isolation_sensitivity_fixed(price_df, market_df, excess_rate_values, price_drop_values):
    """
    시장격리 기준을 다양한 조합으로 조정하고, 결과를 초과생산량 비율 및 시장격리 필요 여부로 요약 출력합니다.
    """
    sensitivity_results = []

    for excess_rate_threshold in excess_rate_values:
        for price_drop_threshold in price_drop_values:
            for _, row in market_df.iterrows():
                year = row['연도']
                excess_rate = (row['초과생산량'] / row['예상생산량']) * 100 if row['예상생산량'] > 0 else 0  # 초과생산량 비율 계산

                # 수확기 쌀값 데이터 가져오기
                harvest_prices = price_df[
                    (price_df['날짜'].dt.year == year) & 
                    (price_df['날짜'].dt.month.isin([10, 11]))
                ]
                if not harvest_prices.empty:
                    수확기_산지쌀값 = harvest_prices['산지쌀값'].mean()
                    평년가격 = row['평년수확기쌀값']
                    price_drop = ((평년가격 - 수확기_산지쌀값) / 평년가격) * 100 if 평년가격 > 0 else 0
                else:
                    수확기_산지쌀값, price_drop = np.nan, np.nan

                # 시장격리 필요 여부 판정
                if excess_rate >= excess_rate_threshold or price_drop >= price_drop_threshold:
                    decision = "시장격리 필요"
                else:
                    decision = "시장격리 불필요"

                # 결과 저장
                sensitivity_results.append({
                    '연도': year,
                    '초과생산량 기준(%)': excess_rate_threshold,
                    '가격 하락률 기준(%)': price_drop_threshold,
                    '초과생산량 비율(%)': round(excess_rate, 2),
                    '시장격리 필요 여부': decision
                })

    # 결과 데이터프레임 생성
    sensitivity_df = pd.DataFrame(sensitivity_results)
    print("\n[시장격리 기준 민감도 분석 시뮬레이션 결과]")
    print(sensitivity_df.to_string(index=False))
    return sensitivity_df

# 사용자 설정값 입력
excess_rate_values = [2.0, 3.0, 4.0]  # 초과생산량 비율 기준 리스트
price_drop_values = [3.0, 5.0, 7.0]   # 가격 하락률 기준 리스트

# 민감도 분석 시뮬레이션 실행
sensitivity_analysis_results = simulate_market_isolation_sensitivity_fixed(
    producer_price_df, market_df, 
    excess_rate_values=excess_rate_values, 
    price_drop_values=price_drop_values