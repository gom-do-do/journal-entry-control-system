
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

    st.caption(f"현재 분석 데이터: {data_source}")

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
                MONTH=pd.to_datetime(controlled_df["POSTING_DATE"], errors="coerce")
                .dt.to_period("M")
                .astype(str),
                STATUS=np.where(controlled_df["BYPASS_FLAG"], "우회 입력", "정상 시간대"),
            )
            monthly_summary = chart_df.groupby(["MONTH", "STATUS"]).size().reset_index(name="COUNT")
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
- 업로드 파일은 필수 컬럼 구조가 맞을 때만 분석에 반영합니다. 컬럼명이 다르면 원천 시스템의 필드 매핑표를 먼저 맞추는 것이 좋습니다.
- 위험 키워드 탐지는 부정 확정이 아니라 조사 우선순위 선정을 위한 1차 스크리닝으로 보는 것이 적절합니다.
- 야간·주말 입력과 위험 키워드가 동시에 발견된 전표는 입력 우회 가능성과 거래 실질 위험이 결합되므로 고위험군으로 분리하는 기준이 실무적으로 유의미합니다.
- 벤포드 분석은 거래가 자연 발생적이고 표본 수가 충분할수록 유용합니다. 급여, 고정 수수료, 정액 상각처럼 금액이 인위적으로 정해지는 모집단에는 별도 기준이 필요합니다.
"""
    )

    if show_all:
        st.subheader("전체 전표 데이터")
        render_table(controlled_df, display_cols)


if __name__ == "__main__":
    main()
