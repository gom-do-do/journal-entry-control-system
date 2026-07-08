import re

import numpy as np
import pandas as pd
import streamlit as st


NIGHT_START_HOUR = 20
NIGHT_END_HOUR = 6
BENFORD_WARNING_THRESHOLD_PCT = 8.0

DEFAULT_RISK_KEYWORDS = [
    "임원 대여",
    "손실 보전",
    "현금 인출",
    "정산 보류",
    "대표이사 개인",
    "자문료 증빙 미비",
]


def benford_expected_probabilities() -> pd.DataFrame:
    digits = np.arange(1, 10)
    expected = np.log10(1 + 1 / digits)
    return pd.DataFrame(
        {
            "FIRST_DIGIT": digits,
            "EXPECTED_PCT": expected * 100,
        }
    )


def generate_benford_amounts(rng: np.random.Generator, rows: int) -> np.ndarray:
    expected = benford_expected_probabilities()["EXPECTED_PCT"].to_numpy() / 100
    first_digits = rng.choice(np.arange(1, 10), size=rows, p=expected)
    exponents = rng.integers(4, 7, size=rows)

    amounts = []
    for digit, exponent in zip(first_digits, exponents):
        scale = 10 ** exponent
        low = digit * scale
        high = (digit + 1) * scale
        amount = rng.integers(low, high)
        amounts.append((amount // 100) * 100)
    return np.array(amounts, dtype=int)


@st.cache_data
def create_mock_journal_entries(seed: int = 42, rows: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    accounts = [
        "현금",
        "보통예금",
        "매출",
        "외상매출금",
        "매입",
        "외상매입금",
        "급여",
        "복리후생비",
        "접대비",
        "소모품비",
        "지급수수료",
        "감가상각비",
    ]
    normal_descriptions = [
        "월중 정상 매출 전표",
        "거래처 대금 입금",
        "사무용품 구매",
        "직원 급여 지급",
        "월말 비용 정산",
        "카드 수수료 인식",
        "감가상각비 계상",
        "거래처 외상대 지급",
        "복리후생비 정산",
        "일반 운영비 처리",
    ]
    risk_descriptions = [
        "임원 대여 관련 가지급금 처리",
        "손실 보전 목적의 비용 대체",
        "긴급 현금 인출 후 사후 정산",
        "거래처 정산 보류 건 임시 처리",
        "대표이사 개인 비용 대납",
        "자문료 증빙 미비 건 비용 인식",
        "현금 인출 및 접대비 정산",
        "손실 보전 관련 조정 전표",
        "임원 대여 회수 지연",
        "대표이사 개인 카드 사용분",
    ]

    business_days = pd.bdate_range("2026-01-01", "2026-03-31")
    posting_dates = pd.Series(rng.choice(business_days, size=rows, replace=True))

    input_hours = rng.integers(9, 18, size=rows)
    input_minutes = rng.integers(0, 60, size=rows)
    input_seconds = rng.integers(0, 60, size=rows)

    weekend_size = min(8, rows)
    night_size = min(7, max(rows - weekend_size, 0))
    keyword_size = min(12, rows)
    manipulation_size = min(14, rows)

    weekend_idx = rng.choice(rows, size=weekend_size, replace=False)
    remaining_idx = np.setdiff1d(np.arange(rows), weekend_idx)
    night_idx = rng.choice(remaining_idx, size=night_size, replace=False)

    high_risk_base_idx = np.concatenate([weekend_idx[:2], night_idx[:3]])
    keyword_pool = np.setdiff1d(np.arange(rows), high_risk_base_idx)
    extra_keyword_size = max(keyword_size - len(high_risk_base_idx), 0)
    extra_keyword_idx = rng.choice(keyword_pool, size=extra_keyword_size, replace=False)
    risk_keyword_idx = np.unique(np.concatenate([high_risk_base_idx, extra_keyword_idx]))

    manipulation_pool = np.setdiff1d(np.arange(rows), risk_keyword_idx)
    manipulation_idx = rng.choice(manipulation_pool, size=manipulation_size, replace=False)

    weekend_dates = pd.to_datetime(
        rng.choice(
            pd.date_range("2026-01-01", "2026-03-31", freq="D")
            .to_series()
            .loc[lambda s: s.dt.weekday >= 5]
            .to_numpy(),
            size=len(weekend_idx),
            replace=True,
        )
    )
    posting_dates.iloc[weekend_idx] = weekend_dates

    input_hours[night_idx[:4]] = 2
    input_hours[night_idx[4:]] = rng.choice(
        [20, 21, 22, 23, 0, 1, 5],
        size=max(len(night_idx) - 4, 0),
    )

    base_amounts = generate_benford_amounts(rng, rows)
    is_debit = rng.random(rows) < 0.75
    debit = np.where(is_debit, base_amounts, 0)
    credit = np.where(is_debit, 0, base_amounts)

    account_names = rng.choice(accounts, size=rows).astype(object)
    descriptions = rng.choice(normal_descriptions, size=rows).astype(object)
    descriptions[risk_keyword_idx] = rng.choice(risk_descriptions, size=len(risk_keyword_idx), replace=True)

    manipulated_amount_pattern = np.array(
        [7_100_000, 7_450_000, 7_820_000, 7_930_000, 8_120_000, 8_560_000, 8_930_000]
    )
    manipulated_amounts = np.resize(manipulated_amount_pattern, len(manipulation_idx))
    rng.shuffle(manipulated_amounts)
    account_names[manipulation_idx] = rng.choice(["소모품비", "지급수수료"], size=len(manipulation_idx))
    descriptions[manipulation_idx] = rng.choice(
        ["대량 구매 비용 정산", "일괄 수수료 정산", "분기 비용 일괄 계상"],
        size=len(manipulation_idx),
        replace=True,
    )
    debit[manipulation_idx] = manipulated_amounts
    credit[manipulation_idx] = 0

    df = pd.DataFrame(
        {
            "JE_NUM": [f"JE-2026-{i:04d}" for i in range(1, rows + 1)],
            "POSTING_DATE": posting_dates.dt.strftime("%Y-%m-%d"),
            "INPUT_TIME": [
                f"{h:02d}:{m:02d}:{s:02d}"
                for h, m, s in zip(input_hours, input_minutes, input_seconds)
            ],
            "ACCOUNT_NAME": account_names,
            "DEBIT": debit,
            "CREDIT": credit,
            "DESCRIPTION": descriptions,
        }
    )

    return df.sort_values(["POSTING_DATE", "INPUT_TIME", "JE_NUM"]).reset_index(drop=True)


def detect_bypass_entries(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    posting_dt = pd.to_datetime(result["POSTING_DATE"])
    input_time = pd.to_datetime(result["INPUT_TIME"], format="%H:%M:%S").dt.time
    input_hour = pd.to_datetime(result["INPUT_TIME"], format="%H:%M:%S").dt.hour

    result["IS_WEEKEND"] = posting_dt.dt.weekday >= 5
    result["IS_NIGHT"] = (input_hour >= NIGHT_START_HOUR) | (input_hour < NIGHT_END_HOUR)
    result["BYPASS_FLAG"] = result["IS_WEEKEND"] | result["IS_NIGHT"]

    result["DETECTION_REASON"] = np.select(
        [
            result["IS_WEEKEND"] & result["IS_NIGHT"],
            result["IS_WEEKEND"],
            result["IS_NIGHT"],
        ],
        [
            "주말 및 야간 입력",
            "주말 입력",
            "야간 입력",
        ],
        default="정상 시간대",
    )
    result["INPUT_TIME"] = input_time.astype(str)
    return result


def parse_custom_keywords(keyword_text: str) -> list[str]:
    if not keyword_text:
        return []
    return [
        keyword.strip()
        for keyword in re.split(r"[,;\n]", keyword_text)
        if keyword.strip()
    ]


def detect_keyword_risk_entries(df: pd.DataFrame, keywords: list[str]) -> pd.DataFrame:
    result = df.copy()
    clean_keywords = list(dict.fromkeys(keyword.strip() for keyword in keywords if keyword.strip()))

    if not clean_keywords:
        result["RISK_KEYWORD_FLAG"] = False
        result["MATCHED_KEYWORDS"] = ""
        result["OVERALL_RISK_LEVEL"] = np.where(result["BYPASS_FLAG"], "중위험", "정상")
        return result

    def find_matches(description: str) -> str:
        text = str(description)
        matches = [keyword for keyword in clean_keywords if keyword.lower() in text.lower()]
        return ", ".join(matches)

    result["MATCHED_KEYWORDS"] = result["DESCRIPTION"].apply(find_matches)
    result["RISK_KEYWORD_FLAG"] = result["MATCHED_KEYWORDS"].ne("")
    result["OVERALL_RISK_LEVEL"] = np.select(
        [
            result["BYPASS_FLAG"] & result["RISK_KEYWORD_FLAG"],
            result["RISK_KEYWORD_FLAG"],
            result["BYPASS_FLAG"],
        ],
        [
            "고위험",
            "키워드 위험",
            "중위험",
        ],
        default="정상",
    )
    return result


def first_digit(value: float) -> int | None:
    amount = abs(int(value))
    if amount <= 0:
        return None
    return int(str(amount)[0])


def analyze_benford_debit(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    debit_df = df.loc[df["DEBIT"] > 0].copy()
    debit_df["FIRST_DIGIT"] = debit_df["DEBIT"].apply(first_digit)

    actual_counts = (
        debit_df["FIRST_DIGIT"]
        .value_counts()
        .reindex(range(1, 10), fill_value=0)
        .rename_axis("FIRST_DIGIT")
        .reset_index(name="ACTUAL_COUNT")
    )
    total_debit_count = int(actual_counts["ACTUAL_COUNT"].sum())

    analysis_df = benford_expected_probabilities().merge(actual_counts, on="FIRST_DIGIT")
    analysis_df["ACTUAL_PCT"] = np.where(
        total_debit_count > 0,
        analysis_df["ACTUAL_COUNT"] / total_debit_count * 100,
        0,
    )
    analysis_df["DIFF_PCT"] = analysis_df["ACTUAL_PCT"] - analysis_df["EXPECTED_PCT"]
    analysis_df["ABS_DIFF_PCT"] = analysis_df["DIFF_PCT"].abs()
    analysis_df["BENFORD_FLAG"] = analysis_df["ABS_DIFF_PCT"] >= BENFORD_WARNING_THRESHOLD_PCT

    return analysis_df, debit_df


def format_amount_columns(df: pd.DataFrame):
    return df.style.format({"DEBIT": "{:,.0f}", "CREDIT": "{:,.0f}"})


def format_percent_columns(df: pd.DataFrame):
    return df.style.format(
        {
            "EXPECTED_PCT": "{:.1f}%",
            "ACTUAL_PCT": "{:.1f}%",
            "DIFF_PCT": "{:+.1f}%p",
            "ABS_DIFF_PCT": "{:.1f}%p",
        }
    )


def render_table(df: pd.DataFrame, columns: list[str]) -> None:
    st.dataframe(
        format_amount_columns(df[columns]),
        width="stretch",
        hide_index=True,
    )


def main() -> None:
    st.set_page_config(page_title="Journal Entry Control", layout="wide")

    st.title("분개장 이상징후 탐지기")
    st.caption("Control 1: 우회 입력 통제 | Control 2: 적요란 위험 키워드 | Control 3: 벤포드의 법칙 분석")

    with st.sidebar:
        st.header("테스트 설정")
        seed = st.number_input("Mock data seed", min_value=1, max_value=9999, value=42, step=1)
        rows = st.slider("전표 건수", min_value=50, max_value=300, value=100, step=10)

        st.header("위험 키워드 설정")
        selected_keywords = st.multiselect(
            "기본 위험 키워드",
            options=DEFAULT_RISK_KEYWORDS,
            default=DEFAULT_RISK_KEYWORDS,
        )
        custom_keyword_text = st.text_area(
            "추가 키워드",
            placeholder="쉼표, 세미콜론 또는 줄바꿈으로 구분",
            height=90,
        )
        show_all = st.toggle("전체 전표 보기", value=False)

    custom_keywords = parse_custom_keywords(custom_keyword_text)
    active_keywords = list(dict.fromkeys(selected_keywords + custom_keywords))

    journal_df = create_mock_journal_entries(seed=seed, rows=rows)
    controlled_df = detect_bypass_entries(journal_df)
    controlled_df = detect_keyword_risk_entries(controlled_df, active_keywords)
    benford_df, debit_sample_df = analyze_benford_debit(controlled_df)
    benford_flags_df = benford_df[benford_df["BENFORD_FLAG"]].copy()

    bypass_df = controlled_df[controlled_df["BYPASS_FLAG"]].copy()
    keyword_df = controlled_df[controlled_df["RISK_KEYWORD_FLAG"]].copy()
    high_risk_df = controlled_df[
        controlled_df["BYPASS_FLAG"] & controlled_df["RISK_KEYWORD_FLAG"]
    ].copy()

    total_count = len(controlled_df)
    weekend_count = int(controlled_df["IS_WEEKEND"].sum())
    night_count = int(controlled_df["IS_NIGHT"].sum())
    bypass_count = int(controlled_df["BYPASS_FLAG"].sum())
    keyword_count = int(controlled_df["RISK_KEYWORD_FLAG"].sum())
    high_risk_count = len(high_risk_df)
    benford_flag_count = len(benford_flags_df)
    high_risk_rate = high_risk_count / total_count if total_count else 0

    metric_cols = st.columns(7)
    metric_cols[0].metric("전체 전표", f"{total_count:,}건")
    metric_cols[1].metric("주말 입력", f"{weekend_count:,}건")
    metric_cols[2].metric("야간 입력", f"{night_count:,}건")
    metric_cols[3].metric("우회 입력 탐지", f"{bypass_count:,}건")
    metric_cols[4].metric("위험 키워드 탐지", f"{keyword_count:,}건")
    metric_cols[5].metric("고위험군", f"{high_risk_count:,}건", f"{high_risk_rate:.1%}")
    metric_cols[6].metric("벤포드 경고 숫자", f"{benford_flag_count:,}개")

    display_cols = [
        "JE_NUM",
        "POSTING_DATE",
        "INPUT_TIME",
        "ACCOUNT_NAME",
        "DEBIT",
        "CREDIT",
        "DESCRIPTION",
        "DETECTION_REASON",
        "MATCHED_KEYWORDS",
        "OVERALL_RISK_LEVEL",
    ]

    st.subheader("고위험군: 우회 입력 + 위험 키워드")
    if high_risk_df.empty:
        st.success("현재 기준으로 우회 입력과 위험 키워드가 동시에 탐지된 전표는 없습니다.")
    else:
        st.error(f"우회 입력과 위험 키워드가 동시에 확인된 고위험 전표 {high_risk_count:,}건이 있습니다.")
        render_table(high_risk_df, display_cols)

    tab_bypass, tab_keyword, tab_benford, tab_summary = st.tabs(
        ["우회 입력 통제", "위험 키워드 스크리닝", "벤포드의 법칙 분석", "요약 분석"]
    )

    with tab_bypass:
        st.subheader("우회 입력 통제 결과")
        if bypass_df.empty:
            st.success("현재 기준으로 탐지된 우회 입력 전표가 없습니다.")
        else:
            st.warning(f"영업시간 외 입력 전표 {bypass_count:,}건이 탐지되었습니다.")
            render_table(bypass_df, display_cols)

    with tab_keyword:
        st.subheader("적요란 위험 키워드 탐지 결과")
        if not active_keywords:
            st.info("사이드바에서 탐지할 위험 키워드를 선택하거나 입력해 주세요.")
        elif keyword_df.empty:
            st.success("현재 선택한 키워드로 탐지된 전표가 없습니다.")
        else:
            st.warning(f"위험 키워드 포함 전표 {keyword_count:,}건이 탐지되었습니다.")
            render_table(keyword_df, display_cols)

    with tab_benford:
        st.subheader("DEBIT 금액 첫째 자리 숫자 분포")
        st.caption(
            f"차변 금액이 0보다 큰 {len(debit_sample_df):,}건을 대상으로 분석합니다. "
            f"이론 분포와 실제 분포의 차이가 {BENFORD_WARNING_THRESHOLD_PCT:.1f}%p 이상이면 경고합니다."
        )

        chart_df = benford_df[["FIRST_DIGIT", "EXPECTED_PCT", "ACTUAL_PCT"]].rename(
            columns={
                "FIRST_DIGIT": "첫째 자리",
                "EXPECTED_PCT": "벤포드 이론",
                "ACTUAL_PCT": "실제 분포",
            }
        )
        chart_df = chart_df.set_index("첫째 자리")
        st.line_chart(chart_df, height=360)

        if benford_flags_df.empty:
            st.success("현재 기준으로 벤포드 분포에서 크게 벗어난 첫째 자리 숫자는 없습니다.")
        else:
            flagged_digits = ", ".join(benford_flags_df["FIRST_DIGIT"].astype(str))
            st.warning(f"벤포드 이론 분포와 차이가 큰 숫자가 확인되었습니다: {flagged_digits}")

        benford_display = benford_df[
            [
                "FIRST_DIGIT",
                "EXPECTED_PCT",
                "ACTUAL_PCT",
                "DIFF_PCT",
                "ACTUAL_COUNT",
                "BENFORD_FLAG",
            ]
        ].rename(
            columns={
                "FIRST_DIGIT": "첫째 자리",
                "EXPECTED_PCT": "이론 비율",
                "ACTUAL_PCT": "실제 비율",
                "DIFF_PCT": "차이",
                "ACTUAL_COUNT": "실제 건수",
                "BENFORD_FLAG": "경고",
            }
        )
        st.dataframe(benford_display, width="stretch", hide_index=True)

        suspect_digits = benford_flags_df["FIRST_DIGIT"].tolist()
        suspect_debit_df = debit_sample_df[debit_sample_df["FIRST_DIGIT"].isin(suspect_digits)].copy()
        if not suspect_debit_df.empty:
            st.subheader("벤포드 경고 숫자로 시작하는 차변 전표")
            benford_cols = display_cols + ["FIRST_DIGIT"]
            render_table(suspect_debit_df, benford_cols)

    with tab_summary:
        left, right = st.columns([1, 2])
        with left:
            st.subheader("탐지 사유별 건수")
            reason_summary = (
                controlled_df.loc[controlled_df["BYPASS_FLAG"], "DETECTION_REASON"]
                .value_counts()
                .rename_axis("탐지 사유")
                .reset_index(name="건수")
            )
            st.dataframe(reason_summary, width="stretch", hide_index=True)

            st.subheader("위험 수준별 건수")
            risk_summary = (
                controlled_df["OVERALL_RISK_LEVEL"]
                .value_counts()
                .rename_axis("위험 수준")
                .reset_index(name="건수")
            )
            st.dataframe(risk_summary, width="stretch", hide_index=True)

        with right:
            st.subheader("월별 입력 현황")
            chart_df = controlled_df.assign(
                MONTH=pd.to_datetime(controlled_df["POSTING_DATE"]).dt.to_period("M").astype(str),
                STATUS=np.where(controlled_df["BYPASS_FLAG"], "우회 입력", "정상 시간대"),
            )
            monthly_summary = (
                chart_df.groupby(["MONTH", "STATUS"]).size().reset_index(name="COUNT")
            )
            st.bar_chart(monthly_summary, x="MONTH", y="COUNT", color="STATUS")

            st.subheader("벤포드 차이 상위 숫자")
            top_benford = benford_df.sort_values("ABS_DIFF_PCT", ascending=False).head(3)
            st.dataframe(
                format_percent_columns(
                    top_benford[
                        ["FIRST_DIGIT", "EXPECTED_PCT", "ACTUAL_PCT", "DIFF_PCT", "ABS_DIFF_PCT"]
                    ]
                ),
                width="stretch",
                hide_index=True,
            )

    st.subheader("감사 실무 관점의 기준")
    st.markdown(
        """
- 위험 키워드 탐지는 부정 확정이 아니라 조사 우선순위 선정을 위한 1차 스크리닝으로 보는 것이 적절합니다.
- 야간·주말 입력과 위험 키워드가 동시에 발견된 전표는 입력 우회 가능성과 거래 실질 위험이 결합되므로 고위험군으로 분리하는 기준이 실무적으로 유의미합니다.
- 벤포드 분석은 거래가 자연 발생적이고 표본 수가 충분할수록 유용합니다. 급여, 고정 수수료, 정액 상각처럼 금액이 인위적으로 정해지는 모집단에는 별도 기준이 필요합니다.
- 특정 숫자에서 큰 차이가 발생하면 해당 숫자로 시작하는 대형 전표, 특정 계정과목, 입력자, 승인자, 결산일 근접 여부를 함께 확인하는 방식이 실무적으로 적절합니다.
"""
    )

    if show_all:
        st.subheader("전체 Mock 전표 데이터")
        render_table(controlled_df, display_cols)


if __name__ == "__main__":
    main()
