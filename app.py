import re
from datetime import datetime, time
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st


NIGHT_START_HOUR = 20
NIGHT_END_HOUR = 6
BENFORD_WARNING_THRESHOLD_PCT = 8.0

REQUIRED_COLUMNS = [
    "JE_NUM",
    "POSTING_DATE",
    "INPUT_TIME",
    "ACCOUNT_NAME",
    "DEBIT",
    "CREDIT",
    "DESCRIPTION",
]

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
    return pd.DataFrame({"FIRST_DIGIT": digits, "EXPECTED_PCT": expected * 100})


def generate_benford_amounts(rng: np.random.Generator, rows: int) -> np.ndarray:
    expected = benford_expected_probabilities()["EXPECTED_PCT"].to_numpy() / 100
    first_digits = rng.choice(np.arange(1, 10), size=rows, p=expected)
    exponents = rng.integers(4, 7, size=rows)

    amounts = []
    for digit, exponent in zip(first_digits, exponents):
        scale = 10**exponent
        amount = rng.integers(digit * scale, (digit + 1) * scale)
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


def read_uploaded_journal_file(uploaded_file) -> pd.DataFrame:
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        for encoding in ["utf-8-sig", "cp949", "euc-kr"]:
            try:
                uploaded_file.seek(0)
                return pd.read_csv(uploaded_file, encoding=encoding)
            except UnicodeDecodeError:
                continue
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file)

    if file_name.endswith((".xlsx", ".xls")):
        uploaded_file.seek(0)
        return pd.read_excel(uploaded_file)

    raise ValueError("CSV 또는 Excel 파일만 업로드할 수 있습니다.")


def validate_required_columns(df: pd.DataFrame) -> list[str]:
    normalized_columns = {str(col).strip() for col in df.columns}
    return [col for col in REQUIRED_COLUMNS if col not in normalized_columns]


def guess_column_mapping(columns: list[str]) -> dict[str, str | None]:
    alias_map = {
        "JE_NUM": ["JE_NUM", "전표번호", "전표 번호", "Document No", "Document Number", "Slip No", "Voucher No"],
        "POSTING_DATE": ["POSTING_DATE", "전표일자", "전표 일자", "Posting Date", "Document Date", "Date"],
        "INPUT_TIME": ["INPUT_TIME", "입력시간", "입력 시간", "Input Time", "Entry Time", "Created Time"],
        "ACCOUNT_NAME": ["ACCOUNT_NAME", "계정과목", "계정 과목", "Account", "Account Name", "GL Account"],
        "DEBIT": ["DEBIT", "차변", "차변금액", "Debit", "Debit Amount", "Dr"],
        "CREDIT": ["CREDIT", "대변", "대변금액", "Credit", "Credit Amount", "Cr"],
        "DESCRIPTION": ["DESCRIPTION", "적요", "Description", "Narration", "Memo", "Text"],
    }
    normalized_lookup = {str(col).strip().lower(): str(col).strip() for col in columns}
    mapping = {}
    for required_col, aliases in alias_map.items():
        mapped_col = None
        for alias in aliases:
            if alias.strip().lower() in normalized_lookup:
                mapped_col = normalized_lookup[alias.strip().lower()]
                break
        mapping[required_col] = mapped_col
    return mapping


def apply_column_mapping(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    mapped = pd.DataFrame()
    for required_col in REQUIRED_COLUMNS:
        mapped[required_col] = df[mapping[required_col]]
    return mapped


def normalize_time_value(value):
    if pd.isna(value):
        return pd.NA
    if isinstance(value, pd.Timestamp):
        return value.strftime("%H:%M:%S")
    if isinstance(value, datetime):
        return value.strftime("%H:%M:%S")
    if isinstance(value, time):
        return value.strftime("%H:%M:%S")
    if isinstance(value, (int, float, np.integer, np.floating)) and 0 <= float(value) < 1:
        total_seconds = int(round(float(value) * 24 * 60 * 60)) % (24 * 60 * 60)
        hour = total_seconds // 3600
        minute = (total_seconds % 3600) // 60
        second = total_seconds % 60
        return f"{hour:02d}:{minute:02d}:{second:02d}"

    text = str(value).strip()
    if re.fullmatch(r"\d{1,2}:\d{2}(:\d{2})?", text):
        parsed = pd.to_datetime(text, format="%H:%M:%S" if text.count(":") == 2 else "%H:%M", errors="coerce")
    else:
        parsed = pd.to_datetime(text, errors="coerce")

    if pd.isna(parsed):
        return pd.NA
    return parsed.strftime("%H:%M:%S")


def normalize_amount_series(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(r"[^\d\-.]", "", regex=True)
        .replace("", np.nan)
    )
    return pd.to_numeric(cleaned, errors="coerce").fillna(0)


def prepare_journal_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared.columns = [str(col).strip() for col in prepared.columns]
    prepared = prepared[REQUIRED_COLUMNS].copy()

    prepared["JE_NUM"] = prepared["JE_NUM"].astype(str)
    prepared["POSTING_DATE"] = pd.to_datetime(prepared["POSTING_DATE"], errors="coerce").dt.strftime("%Y-%m-%d")
    prepared["INPUT_TIME"] = prepared["INPUT_TIME"].apply(normalize_time_value)
    prepared["ACCOUNT_NAME"] = prepared["ACCOUNT_NAME"].fillna("").astype(str)
    prepared["DEBIT"] = normalize_amount_series(prepared["DEBIT"])
    prepared["CREDIT"] = normalize_amount_series(prepared["CREDIT"])
    prepared["DESCRIPTION"] = prepared["DESCRIPTION"].fillna("").astype(str)

    return prepared.reset_index(drop=True)


def profile_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    posting_dt = pd.to_datetime(df["POSTING_DATE"], errors="coerce")
    input_dt = pd.to_datetime(df["INPUT_TIME"], format="%H:%M:%S", errors="coerce")
    debit = pd.to_numeric(df["DEBIT"], errors="coerce").fillna(0)
    credit = pd.to_numeric(df["CREDIT"], errors="coerce").fillna(0)

    checks = [
        {
            "CHECK_NAME": "전체 행 수",
            "ISSUE_COUNT": len(df),
            "SEVERITY": "Info",
            "DESCRIPTION": "분석 대상 전체 전표 라인 수",
        },
        {
            "CHECK_NAME": "전표일자 파싱 실패",
            "ISSUE_COUNT": int(posting_dt.isna().sum()),
            "SEVERITY": "High",
            "DESCRIPTION": "POSTING_DATE가 날짜로 해석되지 않는 건",
        },
        {
            "CHECK_NAME": "입력시간 파싱 실패",
            "ISSUE_COUNT": int(input_dt.isna().sum()),
            "SEVERITY": "Medium",
            "DESCRIPTION": "INPUT_TIME이 HH:MM:SS로 해석되지 않는 건",
        },
        {
            "CHECK_NAME": "계정과목 누락",
            "ISSUE_COUNT": int(df["ACCOUNT_NAME"].astype(str).str.strip().eq("").sum()),
            "SEVERITY": "Medium",
            "DESCRIPTION": "ACCOUNT_NAME이 비어 있는 건",
        },
        {
            "CHECK_NAME": "적요 누락",
            "ISSUE_COUNT": int(df["DESCRIPTION"].astype(str).str.strip().eq("").sum()),
            "SEVERITY": "Low",
            "DESCRIPTION": "DESCRIPTION이 비어 있는 건",
        },
        {
            "CHECK_NAME": "차변/대변 모두 0",
            "ISSUE_COUNT": int(((debit == 0) & (credit == 0)).sum()),
            "SEVERITY": "High",
            "DESCRIPTION": "금액 정보가 없는 전표 라인",
        },
        {
            "CHECK_NAME": "차변/대변 동시 입력",
            "ISSUE_COUNT": int(((debit > 0) & (credit > 0)).sum()),
            "SEVERITY": "Medium",
            "DESCRIPTION": "단일 라인에 차변과 대변이 모두 입력된 건",
        },
        {
            "CHECK_NAME": "음수 금액",
            "ISSUE_COUNT": int(((debit < 0) | (credit < 0)).sum()),
            "SEVERITY": "Medium",
            "DESCRIPTION": "차변 또는 대변 금액이 음수인 건",
        },
    ]
    return pd.DataFrame(checks)


def detect_bypass_entries(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    posting_dt = pd.to_datetime(result["POSTING_DATE"], errors="coerce")
    input_dt = pd.to_datetime(result["INPUT_TIME"], format="%H:%M:%S", errors="coerce")
    input_hour = input_dt.dt.hour

    result["IS_WEEKEND"] = posting_dt.dt.weekday >= 5
    result["IS_NIGHT"] = (input_hour >= NIGHT_START_HOUR) | (input_hour < NIGHT_END_HOUR)
    result["BYPASS_FLAG"] = result["IS_WEEKEND"].fillna(False) | result["IS_NIGHT"].fillna(False)

    result["DETECTION_REASON"] = np.select(
        [
            result["IS_WEEKEND"] & result["IS_NIGHT"],
            result["IS_WEEKEND"],
            result["IS_NIGHT"],
        ],
        ["주말 및 야간 입력", "주말 입력", "야간 입력"],
        default="정상 시간대",
    )
    result["INPUT_TIME"] = input_dt.dt.strftime("%H:%M:%S").fillna("")
    return result


def parse_custom_keywords(keyword_text: str) -> list[str]:
    if not keyword_text:
        return []
    return [keyword.strip() for keyword in re.split(r"[,;\n]", keyword_text) if keyword.strip()]


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
        ["고위험", "키워드 위험", "중위험"],
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


def analyze_account_benford(df: pd.DataFrame, min_debit_count: int = 5) -> pd.DataFrame:
    rows = []
    for account_name, account_df in df[df["DEBIT"] > 0].groupby("ACCOUNT_NAME"):
        analysis_df, debit_df = analyze_benford_debit(account_df)
        if len(debit_df) < min_debit_count:
            continue
        top_gap = analysis_df.sort_values("ABS_DIFF_PCT", ascending=False).iloc[0]
        rows.append(
            {
                "ACCOUNT_NAME": account_name,
                "DEBIT_COUNT": len(debit_df),
                "TOP_GAP_DIGIT": int(top_gap["FIRST_DIGIT"]),
                "TOP_GAP_PCT": float(top_gap["ABS_DIFF_PCT"]),
                "FLAGGED_DIGITS": ", ".join(analysis_df.loc[analysis_df["BENFORD_FLAG"], "FIRST_DIGIT"].astype(str)),
            }
        )
    return pd.DataFrame(rows).sort_values("TOP_GAP_PCT", ascending=False) if rows else pd.DataFrame()


def calculate_risk_scores(
    df: pd.DataFrame,
    benford_flags_df: pd.DataFrame,
    manual_high_amount_threshold: float | None = None,
) -> pd.DataFrame:
    result = df.copy()
    amount_abs = result[["DEBIT", "CREDIT"]].abs().max(axis=1)
    positive_amounts = amount_abs[amount_abs > 0]
    auto_threshold = positive_amounts.quantile(0.95) if not positive_amounts.empty else np.inf
    high_amount_threshold = manual_high_amount_threshold if manual_high_amount_threshold else auto_threshold

    posting_dt = pd.to_datetime(result["POSTING_DATE"], errors="coerce")
    month_end = posting_dt + pd.offsets.MonthEnd(0)
    days_to_month_end = (month_end - posting_dt).dt.days

    result["FIRST_DIGIT"] = result["DEBIT"].apply(first_digit)
    benford_flag_digits = set(benford_flags_df["FIRST_DIGIT"].astype(int).tolist())
    result["BENFORD_DIGIT_FLAG"] = result["FIRST_DIGIT"].isin(benford_flag_digits) & result["DEBIT"].gt(0)
    result["HIGH_AMOUNT_FLAG"] = amount_abs >= high_amount_threshold
    result["MONTH_END_FLAG"] = days_to_month_end.between(0, 3).fillna(False)
    result["HIGH_AMOUNT_THRESHOLD"] = high_amount_threshold

    result["RISK_SCORE"] = 0
    result.loc[result["IS_WEEKEND"], "RISK_SCORE"] += 20
    result.loc[result["IS_NIGHT"], "RISK_SCORE"] += 20
    result.loc[result["RISK_KEYWORD_FLAG"], "RISK_SCORE"] += 30
    result.loc[result["BENFORD_DIGIT_FLAG"], "RISK_SCORE"] += 20
    result.loc[result["HIGH_AMOUNT_FLAG"], "RISK_SCORE"] += 10
    result.loc[result["MONTH_END_FLAG"] & result["BYPASS_FLAG"], "RISK_SCORE"] += 15

    factor_map = [
        ("IS_WEEKEND", "주말 입력"),
        ("IS_NIGHT", "야간 입력"),
        ("RISK_KEYWORD_FLAG", "위험 키워드"),
        ("BENFORD_DIGIT_FLAG", "벤포드 경고 숫자"),
        ("HIGH_AMOUNT_FLAG", "고액 전표"),
        ("MONTH_END_FLAG", "월말 3일 이내"),
    ]

    def collect_factors(row: pd.Series) -> str:
        factors = [label for column, label in factor_map if bool(row.get(column, False))]
        return ", ".join(factors) if factors else "정상"

    result["RISK_FACTORS"] = result.apply(collect_factors, axis=1)
    result["AUDIT_PRIORITY"] = np.select(
        [
            result["RISK_SCORE"] >= 70,
            result["RISK_SCORE"] >= 40,
            result["RISK_SCORE"] >= 20,
        ],
        ["높음", "보통", "낮음"],
        default="정상",
    )
    return result


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def create_excel_workpaper(
    controlled_df: pd.DataFrame,
    bypass_df: pd.DataFrame,
    keyword_df: pd.DataFrame,
    score_high_risk_df: pd.DataFrame,
    benford_df: pd.DataFrame,
    account_benford_df: pd.DataFrame,
    data_quality_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary_df.to_excel(writer, index=False, sheet_name="Summary")
        score_high_risk_df.to_excel(writer, index=False, sheet_name="High Risk")
        bypass_df.to_excel(writer, index=False, sheet_name="Bypass Entries")
        keyword_df.to_excel(writer, index=False, sheet_name="Keyword Risk")
        benford_df.to_excel(writer, index=False, sheet_name="Benford")
        account_benford_df.to_excel(writer, index=False, sheet_name="Account Benford")
        data_quality_df.to_excel(writer, index=False, sheet_name="Data Quality")
        controlled_df.to_excel(writer, index=False, sheet_name="Full Data")
    return output.getvalue()


def format_amount_columns(df: pd.DataFrame):
    formatters = {}
    if "DEBIT" in df.columns:
        formatters["DEBIT"] = "{:,.0f}"
    if "CREDIT" in df.columns:
        formatters["CREDIT"] = "{:,.0f}"
    if "RISK_SCORE" in df.columns:
        formatters["RISK_SCORE"] = "{:,.0f}"
    return df.style.format(formatters)


def format_percent_columns(df: pd.DataFrame):
    return df.style.format(
        {
            "EXPECTED_PCT": "{:.1f}%",
            "ACTUAL_PCT": "{:.1f}%",
            "DIFF_PCT": "{:+.1f}%p",
            "ABS_DIFF_PCT": "{:.1f}%p",
            "TOP_GAP_PCT": "{:.1f}%p",
        }
    )


def render_table(df: pd.DataFrame, columns: list[str]) -> None:
    visible_columns = [col for col in columns if col in df.columns]
    st.dataframe(format_amount_columns(df[visible_columns]), width="stretch", hide_index=True)


def main() -> None:
    st.set_page_config(page_title="Journal Entry Control", layout="wide")

    st.title("분개장 이상징후 탐지기")
    st.caption("Control 1: 우회 입력 통제 | Control 2: 적요란 위험 키워드 | Control 3: 벤포드의 법칙 분석")

    uploaded_file = None
    upload_error = None
    data_source = "Mock Data"
    raw_uploaded_df = None
    column_mapping = {}

    with st.sidebar:
        st.header("데이터 설정")
        uploaded_file = st.file_uploader(
            "실제 분개장 파일 업로드",
            type=["csv", "xlsx", "xls"],
            help="필수 컬럼: JE_NUM, POSTING_DATE, INPUT_TIME, ACCOUNT_NAME, DEBIT, CREDIT, DESCRIPTION",
        )
        st.caption("파일 업로드 시 필수 컬럼이 포함되어 있어야 합니다.")

        if uploaded_file is not None:
            try:
                raw_uploaded_df = read_uploaded_journal_file(uploaded_file)
                raw_columns = [str(col).strip() for col in raw_uploaded_df.columns]
                guessed_mapping = guess_column_mapping(raw_columns)

                st.subheader("컬럼 매핑")
                st.caption("업로드 파일의 컬럼을 분석 표준 컬럼에 연결해 주세요.")
                mapping_options = ["선택 안 함"] + raw_columns
                for required_col in REQUIRED_COLUMNS:
                    guessed_col = guessed_mapping.get(required_col)
                    default_index = mapping_options.index(guessed_col) if guessed_col in mapping_options else 0
                    selected_col = st.selectbox(
                        required_col,
                        options=mapping_options,
                        index=default_index,
                        key=f"mapping_{required_col}",
                    )
                    column_mapping[required_col] = None if selected_col == "선택 안 함" else selected_col
            except Exception as exc:
                upload_error = f"업로드 파일을 읽는 중 오류가 발생했습니다: {exc}"

        seed = st.number_input("Mock data seed", min_value=1, max_value=9999, value=42, step=1)
        rows = st.slider("Mock 전표 건수", min_value=50, max_value=300, value=100, step=10)

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

        st.header("위험 점수 설정")
        manual_threshold = st.number_input(
            "고액 전표 기준 금액",
            min_value=0,
            value=0,
            step=100000,
            help="0이면 전체 금액 상위 5%를 자동 기준으로 사용합니다.",
        )
        show_all = st.toggle("전체 전표 보기", value=False)

    mock_df = create_mock_journal_entries(seed=seed, rows=rows)
    journal_df = mock_df.copy()

    mock_csv = dataframe_to_csv_bytes(mock_df)
    st.download_button(
        label="테스트용 가상 데이터 다운로드",
        data=mock_csv,
        file_name="mock_journal_entry.csv",
        mime="text/csv",
        width="content",
    )

    if uploaded_file is not None:
        try:
            if raw_uploaded_df is None:
                raw_uploaded_df = read_uploaded_journal_file(uploaded_file)

            missing_mappings = [col for col in REQUIRED_COLUMNS if not column_mapping.get(col)]
            duplicated_mappings = [
                col for col in set(column_mapping.values())
                if col is not None and list(column_mapping.values()).count(col) > 1
            ]

            if missing_mappings:
                upload_error = (
                    "파일 업로드 시 필수 컬럼(JE_NUM, POSTING_DATE, INPUT_TIME, "
                    "ACCOUNT_NAME, DEBIT, CREDIT, DESCRIPTION)에 대한 컬럼 매핑이 필요합니다. "
                    f"미매핑 컬럼: {', '.join(missing_mappings)}"
                )
            elif duplicated_mappings:
                upload_error = (
                    "하나의 원본 컬럼을 여러 표준 컬럼에 중복 매핑할 수 없습니다. "
                    f"중복 선택 컬럼: {', '.join(duplicated_mappings)}"
                )
            else:
                mapped_uploaded_df = apply_column_mapping(raw_uploaded_df, column_mapping)
                journal_df = prepare_journal_dataframe(mapped_uploaded_df)
                data_source = f"업로드 파일: {uploaded_file.name}"
        except Exception as exc:
            upload_error = f"업로드 파일을 읽는 중 오류가 발생했습니다: {exc}"

    if upload_error:
        st.error(upload_error)
        st.info("업로드 파일은 적용하지 않고 Mock Data를 기준으로 분석합니다.")
    elif uploaded_file is not None:
        st.success(f"{uploaded_file.name} 파일을 기준으로 분석합니다.")
    else:
        st.info("현재 Mock Data를 기준으로 분석합니다. 사이드바에서 CSV 또는 Excel 파일을 업로드할 수 있습니다.")

    custom_keywords = parse_custom_keywords(custom_keyword_text)
    active_keywords = list(dict.fromkeys(selected_keywords + custom_keywords))

    data_quality_df = profile_data_quality(journal_df)
    controlled_df = detect_bypass_entries(journal_df)
    controlled_df = detect_keyword_risk_entries(controlled_df, active_keywords)
    benford_df, debit_sample_df = analyze_benford_debit(controlled_df)
    benford_flags_df = benford_df[benford_df["BENFORD_FLAG"]].copy()
    controlled_df = calculate_risk_scores(
        controlled_df,
        benford_flags_df,
        manual_high_amount_threshold=float(manual_threshold) if manual_threshold else None,
    )
    _, debit_sample_df = analyze_benford_debit(controlled_df)
    account_benford_df = analyze_account_benford(controlled_df)

    bypass_df = controlled_df[controlled_df["BYPASS_FLAG"]].copy()
    keyword_df = controlled_df[controlled_df["RISK_KEYWORD_FLAG"]].copy()
    rule_high_risk_df = controlled_df[
        controlled_df["BYPASS_FLAG"] & controlled_df["RISK_KEYWORD_FLAG"]
    ].copy()
    score_high_risk_df = controlled_df[controlled_df["AUDIT_PRIORITY"].eq("높음")].copy()
    benford_suspect_df = controlled_df[controlled_df["BENFORD_DIGIT_FLAG"]].copy()

    total_count = len(controlled_df)
    weekend_count = int(controlled_df["IS_WEEKEND"].sum())
    night_count = int(controlled_df["IS_NIGHT"].sum())
    bypass_count = int(controlled_df["BYPASS_FLAG"].sum())
    keyword_count = int(controlled_df["RISK_KEYWORD_FLAG"].sum())
    rule_high_risk_count = len(rule_high_risk_df)
    score_high_risk_count = len(score_high_risk_df)
    benford_flag_count = len(benford_flags_df)
    high_risk_rate = rule_high_risk_count / total_count if total_count else 0
    data_quality_issue_count = int(
        data_quality_df.loc[
            data_quality_df["SEVERITY"].ne("Info") & data_quality_df["ISSUE_COUNT"].gt(0),
            "ISSUE_COUNT",
        ].sum()
    )

    summary_df = pd.DataFrame(
        [
            {"항목": "분석 데이터", "값": data_source},
            {"항목": "전체 전표", "값": total_count},
            {"항목": "주말 입력", "값": weekend_count},
            {"항목": "야간 입력", "값": night_count},
            {"항목": "우회 입력 탐지", "값": bypass_count},
            {"항목": "위험 키워드 탐지", "값": keyword_count},
            {"항목": "우회+키워드 고위험", "값": rule_high_risk_count},
            {"항목": "점수 고위험", "값": score_high_risk_count},
            {"항목": "벤포드 경고 숫자", "값": benford_flag_count},
            {"항목": "데이터 품질 이슈", "값": data_quality_issue_count},
        ]
    )

    st.caption(f"현재 분석 데이터: {data_source}")

    metric_cols = st.columns(9)
    metric_cols[0].metric("전체 전표", f"{total_count:,}건")
    metric_cols[1].metric("주말 입력", f"{weekend_count:,}건")
    metric_cols[2].metric("야간 입력", f"{night_count:,}건")
    metric_cols[3].metric("우회 입력 탐지", f"{bypass_count:,}건")
    metric_cols[4].metric("위험 키워드 탐지", f"{keyword_count:,}건")
    metric_cols[5].metric("고위험군", f"{rule_high_risk_count:,}건", f"{high_risk_rate:.1%}")
    metric_cols[6].metric("벤포드 경고 숫자", f"{benford_flag_count:,}개")
    metric_cols[7].metric("점수 고위험", f"{score_high_risk_count:,}건")
    metric_cols[8].metric("품질 이슈", f"{data_quality_issue_count:,}건")

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
        "RISK_SCORE",
        "AUDIT_PRIORITY",
        "RISK_FACTORS",
    ]

    st.subheader("탐지 결과 다운로드")
    excel_bytes = create_excel_workpaper(
        controlled_df,
        bypass_df,
        keyword_df,
        score_high_risk_df,
        benford_df,
        account_benford_df,
        data_quality_df,
        summary_df,
    )
    st.download_button(
        "Excel 감사 워크페이퍼 다운로드",
        data=excel_bytes,
        file_name="journal_entry_control_workpaper.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch",
    )

    download_cols = st.columns(5)
    download_cols[0].download_button(
        "전체 분석 결과",
        data=dataframe_to_csv_bytes(controlled_df),
        file_name="journal_entry_control_all_results.csv",
        mime="text/csv",
        width="stretch",
    )
    download_cols[1].download_button(
        "우회 입력 결과",
        data=dataframe_to_csv_bytes(bypass_df),
        file_name="bypass_entry_results.csv",
        mime="text/csv",
        width="stretch",
        disabled=bypass_df.empty,
    )
    download_cols[2].download_button(
        "키워드 결과",
        data=dataframe_to_csv_bytes(keyword_df),
        file_name="keyword_risk_results.csv",
        mime="text/csv",
        width="stretch",
        disabled=keyword_df.empty,
    )
    download_cols[3].download_button(
        "고위험군 결과",
        data=dataframe_to_csv_bytes(score_high_risk_df),
        file_name="high_risk_scored_results.csv",
        mime="text/csv",
        width="stretch",
        disabled=score_high_risk_df.empty,
    )
    download_cols[4].download_button(
        "벤포드 결과",
        data=dataframe_to_csv_bytes(benford_suspect_df),
        file_name="benford_suspect_results.csv",
        mime="text/csv",
        width="stretch",
        disabled=benford_suspect_df.empty,
    )

    st.subheader("고위험군: 우회 입력 + 위험 키워드")
    if rule_high_risk_df.empty:
        st.success("현재 기준으로 우회 입력과 위험 키워드가 동시에 탐지된 전표는 없습니다.")
    else:
        st.error(f"우회 입력과 위험 키워드가 동시에 확인된 고위험 전표 {rule_high_risk_count:,}건이 있습니다.")
        render_table(rule_high_risk_df, display_cols)

    st.subheader("감사 우선순위: 위험 점수 고위험")
    if score_high_risk_df.empty:
        st.success("현재 위험 점수 기준 고위험 전표는 없습니다.")
    else:
        st.warning(f"위험 점수 70점 이상 전표 {score_high_risk_count:,}건을 우선 검토 대상으로 분류했습니다.")
        render_table(score_high_risk_df.sort_values("RISK_SCORE", ascending=False), display_cols)

    tab_bypass, tab_keyword, tab_benford, tab_account_benford, tab_quality, tab_summary = st.tabs(
        ["우회 입력 통제", "위험 키워드 스크리닝", "벤포드의 법칙 분석", "계정별 벤포드", "데이터 품질", "요약 분석"]
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
        st.line_chart(chart_df.set_index("첫째 자리"), height=360)

        if benford_flags_df.empty:
            st.success("현재 기준으로 벤포드 분포에서 크게 벗어난 첫째 자리 숫자는 없습니다.")
        else:
            flagged_digits = ", ".join(benford_flags_df["FIRST_DIGIT"].astype(str))
            st.warning(f"벤포드 이론 분포와 차이가 큰 숫자가 확인되었습니다: {flagged_digits}")

        benford_display = benford_df[
            ["FIRST_DIGIT", "EXPECTED_PCT", "ACTUAL_PCT", "DIFF_PCT", "ACTUAL_COUNT", "BENFORD_FLAG"]
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
            render_table(suspect_debit_df, display_cols + ["FIRST_DIGIT"])

    with tab_account_benford:
        st.subheader("계정과목별 벤포드 이상 차이 요약")
        if account_benford_df.empty:
            st.info("차변 건수가 충분한 계정과목이 없어 계정별 벤포드 요약을 표시하지 않습니다.")
        else:
            st.dataframe(
                format_percent_columns(account_benford_df),
                width="stretch",
                hide_index=True,
            )

    with tab_quality:
        st.subheader("업로드/분석 데이터 품질 체크")
        issue_df = data_quality_df[
            (data_quality_df["SEVERITY"].ne("Info")) & (data_quality_df["ISSUE_COUNT"].gt(0))
        ].copy()
        if issue_df.empty:
            st.success("현재 데이터 품질 체크에서 주요 이슈가 발견되지 않았습니다.")
        else:
            high_issue_count = int(issue_df.loc[issue_df["SEVERITY"].eq("High"), "ISSUE_COUNT"].sum())
            if high_issue_count:
                st.error(f"High 등급 데이터 품질 이슈 {high_issue_count:,}건이 있습니다.")
            else:
                st.warning("일부 데이터 품질 이슈가 있습니다. 분석 결과 해석 전에 확인해 주세요.")
        st.dataframe(data_quality_df, width="stretch", hide_index=True)

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

            st.subheader("감사 우선순위별 건수")
            priority_summary = (
                controlled_df["AUDIT_PRIORITY"]
                .value_counts()
                .rename_axis("감사 우선순위")
                .reset_index(name="건수")
            )
            st.dataframe(priority_summary, width="stretch", hide_index=True)

        with right:
            st.subheader("월별 입력 현황")
            monthly_chart_df = controlled_df.assign(
                MONTH=pd.to_datetime(controlled_df["POSTING_DATE"], errors="coerce")
                .dt.to_period("M")
                .astype(str),
                STATUS=np.where(controlled_df["BYPASS_FLAG"], "우회 입력", "정상 시간대"),
            )
            monthly_summary = monthly_chart_df.groupby(["MONTH", "STATUS"]).size().reset_index(name="COUNT")
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
- Excel 워크페이퍼는 감사조서 초안처럼 Summary, High Risk, Bypass, Keyword, Benford, Full Data 시트로 구성했습니다.
- 위험 점수는 표본추출의 우선순위를 정하기 위한 실무적 랭킹이며, 부정 확정 판단은 아닙니다.
- 고액 전표 기준은 기본적으로 상위 5% 금액을 쓰되, 사이드바에서 회사 규모에 맞는 금액 기준을 직접 입력할 수 있습니다.
- 계정과목별 벤포드는 전체 분포보다 특정 비용 계정의 이상 패턴을 더 잘 보여줄 수 있습니다.
"""
    )

    if show_all:
        st.subheader("전체 전표 데이터")
        render_table(controlled_df, display_cols)


if __name__ == "__main__":
    main()
