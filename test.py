import io
import pandas as pd
import numpy as np
import streamlit as st
pip install scikit-learn
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# =============================
# 페이지 설정
# =============================
st.set_page_config(page_title="스마트팜 수확량 예측", page_icon="🌱", layout="wide")
st.title("🌱 스마트팜 수확량 예측 웹앱")
st.write("환경 데이터(온도/습도/CO₂/일사량 등)와 생육/수확량 데이터를 기반으로 간단한 예측 모델을 학습합니다.")

# =============================
# 사이드바: 파일 업로드 & 옵션
# =============================
st.sidebar.header("1) 데이터 업로드")
env_file = st.sidebar.file_uploader("시설원예_일기준_환경정보 엑셀 업로드", type=["xlsx", "xls"])
growth_file = st.sidebar.file_uploader("시설원예_생육정보/수확량 엑셀 업로드", type=["xlsx", "xls"])

st.sidebar.header("2) 전처리 옵션")
rm_desc_rows = st.sidebar.checkbox("환경 데이터 상단 설명행 2줄 제거(권장)", value=True, help="제공된 표준 환경파일에 설명행이 2줄 포함된 경우 사용")

st.sidebar.header("3) 모델 옵션")
model_name = st.sidebar.selectbox("모델 선택", ["LinearRegression", "RandomForest"], index=1)
cv_splits = st.sidebar.slider("TimeSeries CV 분할 수", 3, 10, 5)

test_ratio = st.sidebar.slider("검증용 비율 (최근 기간)", 0.1, 0.4, 0.2, step=0.05)

# =============================
# 유틸 함수
# =============================
DATE_CANDIDATES = ["수집일", "날짜", "Date", "date", "기준일", "측정일"]

def detect_date_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if any(k in str(c) for k in DATE_CANDIDATES):
            return c
    # fallback: try first datetime-like
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    return None


def clean_environment_df(df_raw: pd.DataFrame, remove_top_two: bool = True) -> pd.DataFrame:
    df = df_raw.copy()
    if remove_top_two and len(df) > 2:
        df = df.iloc[2:].copy()
        # 제공된 표준 파일의 예상 컬럼명 재정의 (없으면 기존 유지)
        expected = [
            "수집일", "내부CO2", "내부이슬점온도", "내부습도", "양액지습", "외부일사량", "외부온도",
            "내부온도", "토양온도", "외부풍향", "외부풍속", "적산온도", "24시간평균온도"
        ]
        if df.shape[1] >= len(expected):
            df.columns = expected + [f"col_{i}" for i in range(len(expected), df.shape[1])]
    # 날짜/숫자 변환
    date_col = detect_date_col(df)
    if date_col is None:
        st.warning("환경 데이터에서 날짜 컬럼을 찾지 못했습니다. '수집일/날짜' 등의 이름을 확인해주세요.")
    else:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        df = df.sort_values(date_col)
        df = df.rename(columns={date_col: "날짜"})
    # 숫자형으로 변환 시도
    for c in df.columns:
        if c == "날짜":
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # 중복 날짜 평균 처리
    df = df.groupby("날짜", as_index=False).mean(numeric_only=True)
    return df


def clean_growth_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    # 날짜 컬럼 감지 및 변환
    date_col = detect_date_col(df)
    if date_col is None:
        st.warning("생육/수확량 데이터에서 날짜 컬럼을 찾지 못했습니다. '수집일/날짜' 등의 이름을 확인해주세요.")
    else:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        df = df.sort_values(date_col)
        df = df.rename(columns={date_col: "날짜"})
    # 숫자형 변환
    for c in df.columns:
        if c == "날짜":
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # 중복 날짜 집계: 평균(또는 합계가 더 맞다면 아래를 sum으로 변경)
    df = df.groupby("날짜", as_index=False).mean(numeric_only=True)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["year"] = out["날짜"].dt.year
    out["month"] = out["날짜"].dt.month
    out["day"] = out["날짜"].dt.day
    out["dow"] = out["날짜"].dt.dayofweek
    return out


def add_rolling_features(df: pd.DataFrame, feature_cols: list[str], windows=(3, 7)) -> pd.DataFrame:
    out = df.copy()
    for w in windows:
        for c in feature_cols:
            out[f"{c}_roll{w}"] = out[c].rolling(window=w, min_periods=1).mean()
    return out

# =============================
# 데이터 로딩 & 전처리
# =============================
if env_file is not None and growth_file is not None:
    try:
        env_df_raw = pd.read_excel(env_file)
        growth_df_raw = pd.read_excel(growth_file)
    except Exception as e:
        st.error(f"엑셀 파일을 읽는 중 오류가 발생했습니다: {e}")
        st.stop()

    env_df = clean_environment_df(env_df_raw, remove_top_two=rm_desc_rows)
    growth_df = clean_growth_df(growth_df_raw)

    st.subheader("📄 미리보기")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**환경 데이터 (정리 후 상위 5행)**")
        st.dataframe(env_df.head())
    with c2:
        st.markdown("**생육/수확량 데이터 (정리 후 상위 5행)**")
        st.dataframe(growth_df.head())

    # 타깃 선택 (생육/수확량 데이터의 수치형 컬럼 중 선택)
    numeric_cols_growth = [c for c in growth_df.columns if c != "날짜" and pd.api.types.is_numeric_dtype(growth_df[c])]
    if not numeric_cols_growth:
        st.error("생육/수확량 데이터에 수치형 컬럼이 없습니다. 엑셀 컬럼을 확인해주세요.")
        st.stop()

    target_col = st.selectbox("예측할 타깃(수확량/생육지표) 컬럼 선택", numeric_cols_growth)

    # 피처 후보 (환경 데이터의 수치형 컬럼)
    numeric_cols_env = [c for c in env_df.columns if c != "날짜" and pd.api.types.is_numeric_dtype(env_df[c])]
    default_features = [c for c in numeric_cols_env if c in ["내부온도", "내부습도", "내부CO2", "외부일사량", "외부온도", "토양온도", "24시간평균온도"]]
    if not default_features:
        default_features = numeric_cols_env[: min(6, len(numeric_cols_env))]

    selected_features = st.multiselect(
        "환경 피처 선택 (예: 내부온도/내부습도/CO2/일사량 등)",
        numeric_cols_env,
        default=default_features,
    )

    if not selected_features:
        st.warning("최소 1개 이상의 환경 피처를 선택해주세요.")
        st.stop()

    # 병합 및 피처 엔지니어링
    merged = pd.merge_asof(
        left=growth_df.sort_values("날짜"),
        right=env_df.sort_values("날짜"),
        on="날짜",
        direction="backward",
    )

    # 시계열 파생치 추가
    merged = add_time_features(merged)
    merged = add_rolling_features(merged, selected_features, windows=(3, 7))

    # 모델 입력/타깃 구성
    feature_cols = selected_features + [f"{c}_roll3" for c in selected_features] + [f"{c}_roll7" for c in selected_features] + ["year", "month", "day", "dow"]
    X = merged[feature_cols].copy()
    y = merged[target_col].copy()

    # 누락치 처리
    # (TimeSeriesSplit을 쓰므로 간단히 평균대치 + 스케일링)
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, feature_cols)],
        remainder="drop",
    )

    if model_name == "LinearRegression":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=400, random_state=42)

    pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])

    # 학습/검증 분할 (최근 test_ratio 비율을 테스트로 사용)
    n = len(X)
    test_size = max(1, int(n * test_ratio))
    train_X, test_X = X.iloc[:-test_size], X.iloc[-test_size:]
    train_y, test_y = y.iloc[:-test_size], y.iloc[-test_size:]

    # 교차검증 (시계열)
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    try:
        cv_scores = cross_val_score(pipe, X, y, cv=tscv, scoring="r2")
        cv_msg = f"TimeSeries CV R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}"
    except Exception as e:
        cv_msg = f"교차검증을 수행할 수 없습니다: {e}"

    # 학습 및 평가
    pipe.fit(train_X, train_y)
    pred_train = pipe.predict(train_X)
    pred_test = pipe.predict(test_X)

    r2_tr = r2_score(train_y, pred_train)
    r2_te = r2_score(test_y, pred_test)
    mae_te = mean_absolute_error(test_y, pred_test)
    rmse_te = mean_squared_error(test_y, pred_test, squared=False)

    st.subheader("📈 학습 결과")
    st.write(cv_msg)
    c1, c2, c3 = st.columns(3)
    c1.metric("Train R²", f"{r2_tr:.3f}")
    c2.metric("Test R²", f"{r2_te:.3f}")
    c3.metric("Test RMSE", f"{rmse_te:.3f}")
    st.caption(f"Test MAE: {mae_te:.3f}")

    # 예측 vs 실제 (최근 구간)
    st.subheader("🔎 실제 vs 예측 (최근 검증 구간)")
    compare_df = test_X.copy()
    compare_df["실제값"] = test_y.values
    compare_df["예측값"] = pred_test
    compare_df = compare_df.assign(날짜=merged.iloc[-test_size:]["날짜"].values)[["날짜", "실제값", "예측값"]]
    st.dataframe(compare_df.tail(20))
    st.line_chart(compare_df.set_index("날짜"))

    # 피처 중요도 (트리모델일 때)
    if model_name == "RandomForest":
        try:
            # 파이프 내부에서 학습된 모델 접근을 위해 전처리 후 학습을 별도로 수행
            prep = preprocessor.fit(train_X)
            Xtr = prep.transform(train_X)
            rf = RandomForestRegressor(n_estimators=400, random_state=42)
            rf.fit(Xtr, train_y)
            importances = rf.feature_importances_
            imp_df = pd.DataFrame({"feature": feature_cols, "importance": importances}).sort_values("importance", ascending=False)
            st.subheader("🌟 피처 중요도 (RandomForest)")
            st.dataframe(imp_df.head(20))
        except Exception as e:
            st.info(f"피처 중요도를 계산할 수 없습니다: {e}")

    # =============================
    # 시나리오 예측 (가상 환경 입력)
    # =============================
    st.subheader("🧪 시나리오 예측: 환경 값을 직접 입력하여 타깃 예측")
    with st.expander("환경 값 입력 폼"):
        inputs = {}
        for c in selected_features:
            # 기본값: 학습구간 평균
            default_val = float(np.nanmean(train_X[c])) if c in train_X else 0.0
            inputs[c] = st.number_input(f"{c}", value=default_val)
        # 날짜 파생치 필요
        sim_date = st.date_input("가정 날짜", value=pd.to_datetime(merged["날짜"].iloc[-1]).date())
        sim = {
            "year": sim_date.year,
            "month": sim_date.month,
            "day": sim_date.day,
            "dow": pd.Timestamp(sim_date).dayofweek,
        }
        # 롤링 특성은 간단히 현재값으로 대체(보수적 대치)
        for c in selected_features:
            sim[f"{c}_roll3"] = inputs[c]
            sim[f"{c}_roll7"] = inputs[c]
        # 단일 샘플 DataFrame 구성
        sim_row = {**{k: inputs[k] for k in selected_features}, **sim}
        sim_df = pd.DataFrame([sim_row])[feature_cols]
        pred_sim = pipe.predict(sim_df)[0]
        st.success(f"예상 {target_col}: {pred_sim:.3f}")

else:
    st.info("왼쪽 사이드바에서 환경 엑셀과 생육/수확량 엑셀을 업로드해주세요.")

st.caption("Tip: 예측력이 낮다면 (1) 타깃을 주간/월간 집계로 바꾸거나, (2) 롤링 윈도우를 늘리고, (3) 계절/품종/생육단계 같은 메타정보를 추가해 보세요.")
