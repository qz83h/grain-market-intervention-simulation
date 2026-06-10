산지쌀값<-read.csv("산지쌀값.csv",header = T)
소매<-read.csv("소매가_전처리.csv",header=T)
str(산지쌀값)
head(산지쌀값,20)
str(소매)
head(소매,20)
# 산지쌀값 데이터 전처리
library(tidyr)
library(dplyr)
library(stringr)
# 산지쌀값 데이터 전처리
산지쌀값_long <- 산지쌀값 %>%
  rowwise() %>% # 행 단위로 처리
  mutate(기준일 = as.character(기준일)) %>%
  transmute(
    날짜_5일 = paste0(substr(기준일, 1, 4), "-", substr(기준일, 6, 7), "-05"),
    날짜_15일 = paste0(substr(기준일, 1, 4), "-", substr(기준일, 6, 7), "-15"),
    날짜_25일 = paste0(substr(기준일, 1, 4), "-", substr(기준일, 6, 7), "-25"),
    산지쌀값_5일 = as.numeric(gsub(",", "", `X5일`)),
    산지쌀값_15일 = as.numeric(gsub(",", "", `X15일`)),
    산지쌀값_25일 = as.numeric(gsub(",", "", `X25일`))
  ) %>%
  pivot_longer(
    cols = everything(),
    names_to = c(".value", "구분"),
    names_sep = "_"
  ) %>%
  rename(날짜 = 날짜, 산지쌀값 = 산지쌀값) %>%
  arrange(날짜) %>% # 날짜 기준 정렬 
  select(날짜, 산지쌀값) # 필요한 열만 선택

# 소매 데이터 전처리
소매_long <- 소매 %>%
  pivot_longer(cols = starts_with("X"), 
               names_to = "월", 
               values_to = "소매가") %>%
  mutate(
    월 = gsub("X", "", 월),                 # "X" 제거
    월 = gsub("월", "", 월),               # "월" 제거
    날짜 = paste0(구분, "-", 월, "-", gsub("일", "", 구분.1)), # 날짜 생성
    날짜 = as.Date(날짜, format = "%Y-%m-%d"), # 날짜 형식 변환
    소매가 = as.numeric(gsub(",", "", 소매가)) # 쉼표 제거 및 숫자 변환
  ) %>%
  select(날짜, 소매가) %>% # 필요한 열만 선택
  arrange(날짜) # 날짜 순 정렬

# 결과 확인
head(소매_long, 10)

# 산지쌀값_long의 날짜를 <date> 형식으로 변환
산지쌀값_long <- 산지쌀값_long %>%
  mutate(날짜 = as.Date(날짜, format = "%Y-%m-%d"))

# 산지쌀값과 소매 데이터를 날짜 기준으로 병합
merged_data <- left_join(산지쌀값_long, 소매_long, by = "날짜")

# 결과 확인
head(merged_data, 10)

# 결과 확인
head(merged_data)

# 병합된 데이터 CSV 파일로 저장
write.csv(merged_data, "가격.csv", row.names = FALSE)

# 저장 경로 확인
getwd()  # 현재 작업 디렉토리 확인
