import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# 제목
st.title("🌱 스마트팜 가상 실험실")
st.write("온도, 습도, 물 공급량을 조절하여 작물의 성장 변화를 시뮬레이션해보세요!")

# 사용자 입력 (환경 조건)
st.sidebar.header("환경 조건 설정")
temp = st.sidebar.slider("온도(℃)", 10, 40, 25)        # 최적: 22~28℃
humidity = st.sidebar.slider("습도(%)", 20, 100, 60)    # 최적: 50~70%
water = st.sidebar.slider("물 공급량(단위)", 0, 100, 50) # 최적: 40~60

# 성장 모델 정의
days = np.arange(0, 31)  # 30일간 성장 시뮬레이션

# 환경 적합도 계산 (0~1 사이 값)
def suitability(value, opt_min, opt_max):
    if opt_min <= value <= opt_max:
        return 1.0
    else:
        # 범위를 벗어나면 선형적으로 점수 하락
        dist = min(abs(value - opt_min), abs(value - opt_max))
        return max(0, 1 - dist * 0.05)

score_temp = suitability(temp, 22, 28)
score_hum = suitability(humidity, 50, 70)
score_water = suitability(water, 40, 60)

# 종합 점수 (평균)
growth_factor = (score_temp + score_hum + score_water) / 3

# 성장 곡선 모델 (로지스틱 곡선 응용)
max_height = 100  # 최대 성장치
growth_curve = max_height / (1 + np.exp(-0.2 * (days - 15))) * growth_factor

# 결과 출력
st.subheader("📈 작물 성장 시뮬레이션 결과")
fig, ax = plt.subplots()
ax.plot(days, growth_curve, label="성장 곡선", color="green")
ax.set_xlabel("재배 일수 (일)")
ax.set_ylabel("생장 정도 (%)")
ax.set_title("작물 성장 그래프")
ax.legend()
st.pyplot(fig)

# 환경 평가 출력
st.subheader("🌡 환경 적합도 분석")
st.write(f"- 온도 적합도: {score_temp*100:.1f}%")
st.write(f"- 습도 적합도: {score_hum*100:.1f}%")
st.write(f"- 물 공급 적합도: {score_water*100:.1f}%")
st.write(f"➡️ 종합 성장 가능성: **{growth_factor*100:.1f}%**")
