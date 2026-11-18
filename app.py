"""
미국 ETF 성과 및 분산투자 대시보드 (app.py)

Twelve Data 수집 데이터(data/etf_prices.csv)를 기반으로
레버리지 ETF의 장기 성과 특성과 다자산 분산투자 효과를 시각화합니다.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ==============================================================================
# 1. 데이터 로딩 및 전처리
# ==============================================================================

@st.cache_data
def load_data() -> pd.DataFrame:
    """
    data/etf_prices.csv를 읽고 종가 피벗 테이블을 생성합니다.
    - datetime을 날짜 형식으로 변환
    - symbol을 컬럼으로 하고 close를 값으로 하는 피벗 테이블 반환
    """
    df = pd.read_csv("data/etf_prices.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])
    pivot_df = df.pivot(index="datetime", columns="symbol", values="close").sort_index()
    return pivot_df


# ==============================================================================
# 2. 공통 계산 함수 (화면 UI와 분리)
# ==============================================================================

def calc_metrics(series: pd.Series, start_dt: pd.Timestamp, end_dt: pd.Timestamp):
    """
    단일 가격 시계열로부터 핵심 4대 지표를 계산합니다.
    - 총수익률: 마지막 종가 / 첫 종가 - 1
    - CAGR: (1 + 총수익률) ** (1 / 연수) - 1 (연수 = 실제 경과일수 / 365.25)
    - 연변동성: 일간수익률 표준편차 * sqrt(252)
    - 최대낙폭(MDD): 종가 / 누적 최고 종가 - 1 의 최솟값
    """
    first_val = series.iloc[0]
    last_val = series.iloc[-1]
    total_return = (last_val / first_val) - 1.0

    days = (end_dt - start_dt).days
    years = days / 365.25
    if years > 0 and (1.0 + total_return) > 0:
        cagr = (1.0 + total_return) ** (1.0 / years) - 1.0
    else:
        cagr = 0.0

    daily_returns = series.pct_change().dropna()
    volatility = daily_returns.std() * np.sqrt(252)

    cummax = series.cummax()
    drawdown = (series / cummax) - 1.0
    mdd = drawdown.min()

    return {
        "total_return": total_return,
        "cagr": cagr,
        "volatility": volatility,
        "mdd": mdd,
    }


def calc_drawdown_series(series: pd.Series) -> pd.Series:
    """
    가격 시계열의 누적 최고점 대비 낙폭(%) 시계열을 계산합니다.
    """
    cummax = series.cummax()
    return ((series / cummax) - 1.0) * 100.0


def calc_portfolio_series(daily_ret: pd.DataFrame, spy_weight: float, first_date: pd.Timestamp) -> pd.Series:
    """
    SPY와 GLD의 일일 리밸런싱 포트폴리오 가격 시계열을 생성합니다 (첫날 기준 100).
    """
    w_spy = spy_weight / 100.0
    w_gld = 1.0 - w_spy
    port_ret = w_spy * daily_ret["SPY"] + w_gld * daily_ret["GLD"]
    port_cum = (1.0 + port_ret).cumprod() * 100.0
    # 첫날 100 추가
    port_series = pd.concat([pd.Series([100.0], index=[first_date]), port_cum])
    return port_series


# ==============================================================================
# 3. 사이드바 렌더링
# ==============================================================================

def render_sidebar(pivot_df: pd.DataFrame):
    """
    사이드바 위젯을 렌더링하고 필터링된 데이터프레임을 반환합니다.
    """
    st.sidebar.header("분석 기간 설정")

    min_date = pivot_df.index.min().date()
    max_date = pivot_df.index.max().date()

    start_date = st.sidebar.date_input("시작일", value=min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("종료일", value=max_date, min_value=min_date, max_value=max_date)

    # 유효성 검사 1: 시작일이 종료일보다 늦은 경우
    if start_date > end_date:
        st.sidebar.error("시작일은 종료일보다 이전이어야 합니다.")
        st.stop()

    filtered_df = pivot_df.loc[str(start_date):str(end_date)]

    # 유효성 검사 2: 데이터 건수 확인
    if len(filtered_df) < 2:
        st.sidebar.error("선택한 기간에 충분한 데이터가 없습니다.")
        st.stop()

    # 유효성 검사 3: 60영업일 미만 경고
    if len(filtered_df) < 60:
        st.sidebar.warning("구간이 짧아 연환산 지표가 불안정합니다.")

    st.sidebar.markdown("---")
    # 배당 미포함 주의 문구 (접거나 숨기지 않고 상시 표시)
    st.sidebar.info("주의: 종가 기준 가격 수익률이며 배당은 포함되지 않았습니다. 실제 성과와 다릅니다.")

    return filtered_df, start_date, end_date


# ==============================================================================
# 4. 탭 1 — 레버리지의 함정
# ==============================================================================

def render_tab1_leverage(filtered_df: pd.DataFrame, start_dt: pd.Timestamp, end_dt: pd.Timestamp):
    """
    탭 1: QQQ(1배), QLD(2배), TQQQ(3배)의 누적 성과, 낙폭, 요약표, 원금 회복 계산기 등을 표시합니다.
    """
    symbols = ["QQQ", "QLD", "TQQQ"]
    sub_df = filtered_df[symbols]

    st.header("레버리지의 함정: 3배 레버리지는 3배 벌어주지 않는다")
    st.caption("비교 대상: QQQ (나스닥100 1배), QLD (나스닥100 2배), TQQQ (나스닥100 3배)")

    # 4-1. 누적 성과 곡선 & 4-3. 낙폭 시계열 그래프 (2단 컬럼 배치)
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader("누적 성과 곡선 (첫날 = 100)")
        is_log = st.checkbox("세로축 로그 스케일", value=False)
        
        # 첫날 기준 100으로 정규화
        norm_df = (sub_df / sub_df.iloc[0]) * 100.0
        fig_perf = px.line(
            norm_df,
            x=norm_df.index,
            y=symbols,
            labels={"datetime": "날짜", "value": "정규화 가격 (첫날=100)", "symbol": "종목"},
            color_discrete_map={"QQQ": "#1f77b4", "QLD": "#ff7f0e", "TQQQ": "#d62728"}
        )
        fig_perf.update_layout(
            yaxis_type="log" if is_log else "linear",
            hovermode="x unified",
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_perf, use_container_width=True)

    with col_chart2:
        st.subheader("낙폭 (고점 대비 하락률)")
        st.caption("성과가 좋아 보여도 깊은 낙폭을 견디지 못하면 손실이 확정됩니다.")
        
        dd_df = pd.DataFrame(index=sub_df.index)
        for s in symbols:
            dd_df[s] = calc_drawdown_series(sub_df[s])

        fig_dd = px.line(
            dd_df,
            x=dd_df.index,
            y=symbols,
            labels={"datetime": "날짜", "value": "낙폭 (%)", "symbol": "종목"},
            color_discrete_map={"QQQ": "#1f77b4", "QLD": "#ff7f0e", "TQQQ": "#d62728"}
        )
        fig_dd.update_layout(
            hovermode="x unified",
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig_dd.update_yaxes(ticksuffix="%")
        st.plotly_chart(fig_dd, use_container_width=True)

    st.markdown("---")

    # 4-2. 성과 요약표
    st.subheader("성과 요약표")
    summary_data = []
    metrics_cache = {}
    for s in symbols:
        m = calc_metrics(sub_df[s], start_dt, end_dt)
        metrics_cache[s] = m
        summary_data.append({
            "종목": s,
            "총수익률": f"{m['total_return'] * 100:+.1f}%",
            "연환산수익률(CAGR)": f"{m['cagr'] * 100:.1f}%",
            "연변동성": f"{m['volatility'] * 100:.1f}%",
            "최대낙폭(MDD)": f"{m['mdd'] * 100:.1f}%"
        })
    summary_df = pd.DataFrame(summary_data)
    st.table(summary_df.set_index("종목"))

    st.caption("• 총수익률: 선택 기간 첫날 종가 대비 마지막 종가의 총 가격 변동률")
    st.caption("• 연환산수익률(CAGR): 실제 경과일수(365.25일 기준)를 반영한 연평균 복리 수익률")
    st.caption("• 연변동성: 일간수익률 표준편차에 252영업일 제곱근을 곱한 연간 가격 변동폭")
    st.caption("• 최대낙폭(MDD): 투자 기간 중 고점 대비 얼마나 빠졌는가를 나타내는 최대 하락률")

    st.markdown("---")

    # 4-4. 원금 회복 계산기 & 4-5. 동적 메시지 (2단 컬럼)
    col_calc, col_msg = st.columns(2)

    with col_calc:
        st.subheader("원금 회복 계산기")
        st.caption("하락폭이 커질수록 원금 회복에 필요한 상승률은 기하급수적으로 커집니다.")
        loss_pct = st.slider("하락률 선택 (%)", min_value=0, max_value=90, value=50, step=5)
        
        loss_val = loss_pct / 100.0
        req_gain = (1.0 / (1.0 - loss_val) - 1.0) * 100.0 if loss_val < 1.0 else 0.0

        st.metric(label="원금 회복에 필요한 상승률", value=f"+{req_gain:.1f}%")
        st.write("-50% 하락은 +100% 상승이 있어야 원금입니다. 하락과 상승은 대칭이 아닙니다.")

    with col_msg:
        st.subheader("선택 기간 핵심 분석")
        days = (end_dt - start_dt).days
        years = days / 365.25
        
        tqqq_tr = metrics_cache["TQQQ"]["total_return"] * 100.0
        qqq_tr = metrics_cache["QQQ"]["total_return"] * 100.0
        tqqq_mdd = metrics_cache["TQQQ"]["mdd"] * 100.0
        
        # 수익률 배수 계산 (0 나눔 방지)
        ratio_str = f"{(tqqq_tr / qqq_tr):.1f}배" if qqq_tr != 0 else "산출 불가"

        s_str = start_dt.strftime("%Y-%m-%d")
        e_str = end_dt.strftime("%Y-%m-%d")

        info_text = (
            f"{s_str} ~ {e_str} 구간에서 TQQQ(3배)는 {tqqq_tr:+.1f}%, QQQ(1배)는 {qqq_tr:+.1f}%였습니다.\n\n"
            f"배수는 3배지만 {years:.1f}년 수익률 차이는 {ratio_str}에 그쳤습니다.\n\n"
            f"그동안 TQQQ의 최대낙폭은 {tqqq_mdd:.1f}%였습니다."
        )
        st.info(info_text)

        # 4-6. 시작일 변경 유도 문구
        st.markdown(
            "**안내:** 사이드바에서 분석 시작일을 **2023-01-03**으로 바꿔 보십시오. 결론이 뒤집힙니다."
        )


# ==============================================================================
# 5. 탭 2 — 분산투자의 힘
# ==============================================================================

def render_tab2_diversification(filtered_df: pd.DataFrame, start_dt: pd.Timestamp, end_dt: pd.Timestamp):
    """
    탭 2: 7종목 상관계수, SPY/GLD 배분 실험, 비중별 위험-수익 산점도, 자산군 성과 비교를 표시합니다.
    """
    all_symbols = ["QQQ", "QLD", "TQQQ", "SPY", "VTI", "SCHD", "GLD"]
    daily_ret = filtered_df[all_symbols].pct_change().dropna()

    st.header("분산투자의 힘: 섞으면 덜 흔들린다")

    # 5-1. 상관계수 히트맵 & 5-2. 히트맵 해석
    st.subheader("1. 7종목 일간수익률 상관계수 행렬")
    
    corr_matrix = daily_ret.corr()

    col_hmap, col_desc = st.columns([1.2, 0.8])

    with col_hmap:
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1.0,
            zmax=1.0,
            labels=dict(color="상관계수")
        )
        fig_corr.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            height=380
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    with col_desc:
        st.markdown("#### 상관계수 해석 포인트")
        spy_vti = corr_matrix.loc["SPY", "VTI"]
        qqq_tqqq = corr_matrix.loc["QQQ", "TQQQ"]
        gld_spy = corr_matrix.loc["GLD", "SPY"]
        gld_schd = corr_matrix.loc["GLD", "SCHD"]

        st.write(
            f"1. **SPY와 VTI의 상관계수는 {spy_vti:.2f}입니다.**\n"
            f"   상관이 1에 매우 가까우므로 두 종목을 함께 담는 것은 분산이 아닙니다."
        )
        st.write(
            f"2. **QQQ와 TQQQ의 상관계수는 {qqq_tqqq:.2f}입니다.**\n"
            f"   배수만 다를 뿐 매일 같은 방향으로 움직이는 사실상 동일한 자산입니다."
        )
        st.write(
            f"3. **GLD(금)와 주식(SPY {gld_spy:.2f}, SCHD {gld_schd:.2f})은 상관이 매우 낮습니다.**\n"
            f"   움직임이 다른 자산을 섞을 때 실질적인 분산투자 효과가 발생합니다."
        )

    st.markdown("---")

    # 5-3. 두 자산 배분 실험 (SPY + GLD)
    st.subheader("2. 두 자산 배분 실험 (SPY + GLD)")
    st.caption("가정: 매일 비중을 다시 맞춘다는 일일 리밸런싱 가정 하에 계산합니다.")

    spy_weight = st.slider("SPY(주식) 비중 (%)", min_value=0, max_value=100, value=50, step=5)
    gld_weight = 100 - spy_weight

    # 포트폴리오 시계열 및 지표 계산
    first_date = filtered_df.index[0]
    port_series = calc_portfolio_series(daily_ret, spy_weight, first_date)
    port_m = calc_metrics(port_series, start_dt, end_dt)

    # SPY 100% 기준 지표
    spy_m = calc_metrics(filtered_df["SPY"], start_dt, end_dt)

    # 3개 메트릭 표시 (SPY 100% 대비 차이)
    col_m1, col_m2, col_m3 = st.columns(3)

    delta_tr = (port_m["total_return"] - spy_m["total_return"]) * 100.0
    delta_vol = (port_m["volatility"] - spy_m["volatility"]) * 100.0
    delta_mdd = (port_m["mdd"] - spy_m["mdd"]) * 100.0

    with col_m1:
        st.metric(
            label=f"포트폴리오 총수익률 (SPY {spy_weight}% / GLD {gld_weight}%)",
            value=f"{port_m['total_return'] * 100:+.1f}%",
            delta=f"{delta_tr:+.1f}%p (SPY 100% 대비)"
        )
    with col_m2:
        st.metric(
            label="연변동성 (위험도)",
            value=f"{port_m['volatility'] * 100:.1f}%",
            delta=f"{delta_vol:+.1f}%p (SPY 100% 대비)",
            delta_color="inverse"
        )
    with col_m3:
        st.metric(
            label="최대낙폭 (MDD)",
            value=f"{port_m['mdd'] * 100:.1f}%",
            delta=f"{delta_mdd:+.1f}%p (SPY 100% 대비)",
            delta_color="normal"
        )

    st.markdown("---")

    # 5-4. 비중별 위험-수익 산점도 & 5-5. 자산군 성과 비교 (2단 컬럼)
    col_scatter, col_assets = st.columns(2)

    with col_scatter:
        st.subheader("비중별 위험-수익 산점도")
        st.caption("SPY 비중을 0%부터 100%까지 10% 간격으로 배치한 결과입니다.")

        scatter_points = []
        for w in range(0, 101, 10):
            p_s = calc_portfolio_series(daily_ret, float(w), first_date)
            pm = calc_metrics(p_s, start_dt, end_dt)
            scatter_points.append({
                "비중": f"SPY {w}%",
                "연변동성": pm["volatility"] * 100.0,
                "총수익률": pm["total_return"] * 100.0,
                "spy_w": w
            })
        scatter_df = pd.DataFrame(scatter_points)

        # 현재 슬라이더로 선택된 비중의 수치
        cur_vol = port_m["volatility"] * 100.0
        cur_tr = port_m["total_return"] * 100.0

        fig_sc = go.Figure()

        # 10% 단위 기본 포인트들
        fig_sc.add_trace(go.Scatter(
            x=scatter_df["연변동성"],
            y=scatter_df["총수익률"],
            mode="markers+text",
            text=scatter_df["비중"],
            textposition="top center",
            marker=dict(size=9, color="#1f77b4"),
            name="10% 단위 비중"
        ))

        # 현재 선택된 비중 포인트 강조
        fig_sc.add_trace(go.Scatter(
            x=[cur_vol],
            y=[cur_tr],
            mode="markers+text",
            text=[f"선택: SPY {spy_weight}%"],
            textposition="bottom center",
            marker=dict(size=14, color="#d62728", symbol="diamond"),
            name="선택된 비중"
        ))

        fig_sc.update_layout(
            xaxis_title="연변동성 (%)",
            yaxis_title="총수익률 (%)",
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_sc, use_container_width=True)

        st.info(
            "이 그래프의 가장 왼쪽(최저 변동성 지점)은 주식 100%도 금 100%도 아닌 두 자산을 섞은 구간입니다. "
            "각각의 자산보다 적절히 섞은 포트폴리오가 덜 흔들립니다."
        )

    with col_assets:
        st.subheader("주요 자산군 누적 성과 비교")
        st.caption("SPY (미국 대형주), GLD (금 현물), SCHD (미국 배당성장)")

        asset_symbols = ["SPY", "GLD", "SCHD"]
        norm_asset_df = (filtered_df[asset_symbols] / filtered_df[asset_symbols].iloc[0]) * 100.0

        fig_assets = px.line(
            norm_asset_df,
            x=norm_asset_df.index,
            y=asset_symbols,
            labels={"datetime": "날짜", "value": "정규화 가격 (첫날=100)", "symbol": "종목"},
            color_discrete_map={"SPY": "#1f77b4", "GLD": "#bcbd22", "SCHD": "#2ca02c"}
        )
        fig_assets.update_layout(
            hovermode="x unified",
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_assets, use_container_width=True)

        st.caption("주: SCHD는 배당이 제외된 종가 기준이므로 실제 성과보다 과소평가되어 있습니다.")


# ==============================================================================
# 6. 메인 실행 함수
# ==============================================================================

def main():
    st.set_page_config(
        page_title="미국 ETF 성과 및 분산투자 대시보드",
        layout="wide"
    )

    st.title("미국 ETF 성과 및 분산투자 대시보드")

    # 데이터 로드
    pivot_df = load_data()

    # 사이드바 렌더링
    filtered_df, start_date, end_date = render_sidebar(pivot_df)

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    # 탭 구성
    tab1, tab2 = st.tabs(["레버리지의 함정", "분산투자의 힘"])

    with tab1:
        render_tab1_leverage(filtered_df, start_dt, end_dt)

    with tab2:
        render_tab2_diversification(filtered_df, start_dt, end_dt)


if __name__ == "__main__":
    main()
