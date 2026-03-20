import pickle
from datetime import datetime

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ml_pipeline import MODEL_FEATURES, build_model_features


MONTH_OPTIONS = [
    (1, "January"),
    (2, "February"),
    (3, "March"),
    (4, "April"),
    (5, "May"),
    (6, "June"),
    (7, "July"),
    (8, "August"),
    (9, "September"),
    (10, "October"),
    (11, "November"),
    (12, "December"),
]
MONTH_LABEL_TO_NUMBER = {label: number for number, label in MONTH_OPTIONS}


st.set_page_config(
    page_title="IPO Success Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def load_css():
    """Load custom CSS from external file."""
    try:
        with open("style.css", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("style.css file not found. Using default styling.")


@st.cache_resource
def load_model_objects():
    """Load all required model artifacts."""
    try:
        model = joblib.load("ipo_success_model.pkl")
        scaler = joblib.load("scaler.pkl")
        imputer = joblib.load("imputer.pkl")

        with open("feature_columns.pkl", "rb") as f:
            feature_columns = pickle.load(f)

        with open("model_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

        return model, scaler, imputer, feature_columns, metadata
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None, None, None, None


def predict_ipo_success(ipo_data, model, scaler, imputer, feature_columns):
    """Predict IPO success for new data."""
    try:
        input_df = pd.DataFrame([ipo_data])
        model_input = build_model_features(input_df)

        expected_cols = feature_columns if feature_columns else MODEL_FEATURES
        model_input = model_input.reindex(columns=expected_cols)

        prediction = model.predict(model_input)[0]
        probability = model.predict_proba(model_input)[0]

        return {
            "prediction": "Success" if prediction == 1 else "Failure",
            "probability_failure": float(probability[0]),
            "probability_success": float(probability[1]),
            "confidence": float(max(probability)),
        }
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None


def _render_model_comparison(metadata):
    comparison = metadata.get("model_comparison", []) if metadata else []
    if not comparison:
        return

    chart_df = pd.DataFrame(comparison)
    fig = px.bar(
        chart_df,
        x="name",
        y="test_accuracy",
        text="test_accuracy",
        title="Model Accuracy Comparison",
    )
    fig.update_traces(texttemplate="%{text:.2%}", textposition="outside")
    fig.update_layout(
        xaxis_title="Model",
        yaxis_title="Test Accuracy",
        yaxis_tickformat=".0%",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)


def _normalize_sector_value(value) -> str:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def _get_sector_options(metadata) -> list[str]:
    sector_values = []

    if metadata:
        sector_values.extend(metadata.get("sectors", []))

    try:
        dataset = pd.read_csv("synthetic_ipo_dataset.csv", usecols=["Sector"])
        sector_values.extend(dataset["Sector"].dropna().tolist())
    except Exception:
        pass

    cleaned = [_normalize_sector_value(value) for value in sector_values]
    cleaned = [value for value in cleaned if value]
    options = sorted(set(cleaned))
    return options if options else ["Unknown"]


def _get_default_month_label() -> str:
    current_month = datetime.now().month
    return MONTH_OPTIONS[current_month - 1][1]


def _initialize_selection_state(sectors: list[str]) -> None:
    if "selected_month_label" not in st.session_state:
        st.session_state.selected_month_label = _get_default_month_label()

    if "selected_sector" not in st.session_state:
        st.session_state.selected_sector = sectors[0]

    if st.session_state.selected_month_label not in MONTH_LABEL_TO_NUMBER:
        st.session_state.selected_month_label = _get_default_month_label()

    if st.session_state.selected_sector not in sectors:
        st.session_state.selected_sector = sectors[0]


def _render_investor_demand_charts(qib: float, hni: float, rii: float) -> None:
    total = qib + hni + rii
    shares = [qib, hni, rii]
    categories = ["QIB", "HNI", "RII"]
    colors = ["#0F766E", "#EA580C", "#1D4ED8"]

    share_df = pd.DataFrame(
        {
            "Category": categories,
            "Shares": shares,
            "Share_pct": [((value / total) * 100) if total > 0 else 0 for value in shares],
        }
    )

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        fig_volume = go.Figure(
            go.Bar(
                x=share_df["Category"],
                y=share_df["Shares"],
                marker_color=colors,
                text=[f"{v:,.0f}" for v in share_df["Shares"]],
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>Shares: %{y:,.0f}<extra></extra>",
            )
        )
        fig_volume.update_layout(
            title="Investor Category Demand Volume",
            xaxis_title="Investor Category",
            yaxis_title="Requested Shares",
            yaxis_tickformat=",",
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=50, b=10),
            height=360,
        )
        fig_volume.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.25)")
        fig_volume.update_xaxes(showgrid=False)
        st.plotly_chart(fig_volume, use_container_width=True)

    with chart_col2:
        fig_mix = go.Figure(
            go.Pie(
                labels=share_df["Category"],
                values=share_df["Shares"],
                hole=0.55,
                marker=dict(colors=colors),
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>Shares: %{value:,.0f}<br>Share: %{percent}<extra></extra>",
                sort=False,
            )
        )
        fig_mix.update_layout(
            title="Investor Participation Mix",
            annotations=[
                dict(
                    text=f"Total<br><b>{total:,.0f}</b>",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=14),
                )
            ],
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=50, b=10),
            height=360,
        )
        st.plotly_chart(fig_mix, use_container_width=True)


def main():
    load_css()

    st.markdown(
        """
        <div class="header-container">
            <h1 class="header-title">📈 IPO Success Analyzer</h1>
            <p class="header-subtitle">Fresh Multi-Model Pipeline On synthetic_ipo_dataset.csv</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model, scaler, imputer, feature_columns, metadata = load_model_objects()
    if model is None:
        st.error("Failed to load model files. Run retrain_model.py first.")
        return

    sectors = _get_sector_options(metadata)
    _initialize_selection_state(sectors)

    tab1, tab2, tab3 = st.tabs(["Prediction", "Batch Analysis", "About"])

    with tab1:
        st.markdown("## Enter IPO Details")

        col1, col2, col3 = st.columns(3)
        with col1:
            offer_price = st.number_input("Offer Price", min_value=1.0, value=500.0, step=10.0)
            total_shares = st.number_input("Total Shares", min_value=1.0, value=1_000_000.0, step=50_000.0)
            proceeds_total = st.number_input("Proceeds Total", min_value=1.0, value=500_000_000.0, step=10_000_000.0)

        with col2:
            year = st.number_input("Year", min_value=2010, max_value=2035, value=datetime.now().year, step=1)
            month_label = st.selectbox(
                "Month",
                options=[label for _, label in MONTH_OPTIONS],
                index=[label for _, label in MONTH_OPTIONS].index(st.session_state.selected_month_label),
                key="selected_month_label",
            )
            month = MONTH_LABEL_TO_NUMBER[month_label]
            quarter = (month - 1) // 3 + 1
            sector = st.selectbox(
                "Sector",
                options=sectors,
                index=sectors.index(st.session_state.selected_sector),
                key="selected_sector",
                help="Sector list is auto-loaded from model metadata and dataset.",
            )

            st.caption(f"Selected Month: {month_label} | Selected Sector: {sector}")

        with col3:
            qib = st.number_input("QIB Shares", min_value=0.0, value=450_000.0, step=10_000.0)
            hni = st.number_input("HNI Shares", min_value=0.0, value=300_000.0, step=10_000.0)
            rii = st.number_input("RII Shares", min_value=0.0, value=250_000.0, step=10_000.0)

        total_demand = qib + hni + rii
        demand_multiple = (total_demand / total_shares) if total_shares > 0 else 0.0
        st.info(
            f"Derived Demand Multiple (Total Subscription): {demand_multiple:.2f}x | "
            f"Quarter: Q{int(quarter)}"
        )

        if st.button("Predict IPO Success", use_container_width=True):
            ipo_data = {
                "Offer_Price": offer_price,
                "Total_Shares": total_shares,
                "QIB": qib,
                "HNI": hni,
                "RII": rii,
                "Proceeds_Total": proceeds_total,
                "Sector": sector,
                "Year": year,
                "Month": month,
                "Quarter": quarter,
            }

            with st.spinner("Running prediction..."):
                result = predict_ipo_success(ipo_data, model, scaler, imputer, feature_columns)

            if result:
                if result["prediction"] == "Success":
                    st.markdown(
                        f"""
                        <div class="success-card">
                            <div class="prediction-title">SUCCESS PREDICTED</div>
                            <div class="prediction-subtitle">Likely positive listing outcome</div>
                            <div class="confidence-badge">Confidence: {result['confidence']:.1%}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="failure-card">
                            <div class="prediction-title">CAUTION ADVISED</div>
                            <div class="prediction-subtitle">Likely negative listing outcome</div>
                            <div class="confidence-badge">Confidence: {result['confidence']:.1%}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                colp1, colp2 = st.columns(2)
                with colp1:
                    st.metric("Success Probability", f"{result['probability_success']:.1%}")
                with colp2:
                    st.metric("Failure Probability", f"{result['probability_failure']:.1%}")

                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=result["probability_success"] * 100,
                        title={"text": "Success Probability"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "steps": [
                                {"range": [0, 40], "color": "#fca5a5"},
                                {"range": [40, 70], "color": "#fde68a"},
                                {"range": [70, 100], "color": "#86efac"},
                            ],
                            "threshold": {"line": {"color": "black", "width": 3}, "value": 50},
                        },
                    )
                )
                fig.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig, use_container_width=True)

                _render_investor_demand_charts(qib, hni, rii)

    with tab2:
        st.markdown("## Batch Analysis")
        st.write("Upload CSV with new schema columns used by the fresh model.")
        st.caption(
            "Required columns: Offer_Price, Total_Shares, QIB, HNI, RII, Proceeds_Total, Sector and either "
            "(Year, Month, Quarter) or Listing_Date"
        )

        uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])
        if uploaded_file is not None:
            try:
                df_batch = pd.read_csv(uploaded_file)
                st.success(f"Loaded {len(df_batch)} records")
                st.dataframe(df_batch.head(), use_container_width=True)

                if st.button("Run Batch Prediction", use_container_width=True):
                    with st.spinner("Processing predictions..."):
                        predictions = []
                        for _, row in df_batch.iterrows():
                            result = predict_ipo_success(row.to_dict(), model, scaler, imputer, feature_columns)
                            if result:
                                predictions.append(
                                    {
                                        "Prediction": result["prediction"],
                                        "Success_Probability": result["probability_success"],
                                        "Failure_Probability": result["probability_failure"],
                                        "Confidence": result["confidence"],
                                    }
                                )

                        pred_df = pd.DataFrame(predictions)
                        if len(pred_df) != len(df_batch):
                            st.error("Some rows could not be predicted due to input issues.")
                            return

                        df_results = pd.concat([df_batch.reset_index(drop=True), pred_df], axis=1)
                        st.dataframe(df_results, use_container_width=True)

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            success_rate = (df_results["Prediction"] == "Success").mean()
                            st.metric("Success Rate", f"{success_rate:.1%}")
                        with col2:
                            st.metric("Avg Confidence", f"{df_results['Confidence'].mean():.1%}")
                        with col3:
                            st.metric("Rows", str(len(df_results)))

                        csv_data = df_results.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="Download Predictions",
                            data=csv_data,
                            file_name=f"ipo_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )
            except Exception as e:
                st.error(f"Error processing file: {e}")

    with tab3:
        st.markdown("## About The Model")

        if metadata:
            st.markdown(
                f"""
                <div class="info-card">
                    <h3>Model Details</h3>
                    <p><strong>Best Model:</strong> {metadata.get('model_name', 'Unknown')}</p>
                    <p><strong>Test Accuracy:</strong> {metadata.get('accuracy', 0):.2%}</p>
                    <p><strong>Cross-Validation Accuracy:</strong> {metadata.get('cv_accuracy', 0):.2%} (+/- {metadata.get('cv_std', 0):.2%})</p>
                    <p><strong>Features:</strong> {len(metadata.get('features', []))}</p>
                    <p><strong>Training Date:</strong> {metadata.get('training_date', 'N/A')}</p>
                    <p><strong>Dataset Rows:</strong> {metadata.get('dataset_rows', 'N/A')}</p>
                    <p><strong>Target:</strong> {metadata.get('target_definition', 'N/A')}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            _render_model_comparison(metadata)
        else:
            st.warning("No metadata found.")

        st.markdown(
            """
            <div class="alert-warning">
                <h3>Important Disclaimer</h3>
                <p>This model is for informational and educational use only. It is not financial advice.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
