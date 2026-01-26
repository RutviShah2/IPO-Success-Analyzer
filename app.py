import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="IPO Success Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load custom CSS
def load_css():
    """Load custom CSS from external file"""
    try:
        with open('style.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("⚠️ style.css file not found. Using default styling.")

# Load model and preprocessing objects
@st.cache_resource
def load_model_objects():
    """Load all required model objects"""
    try:
        model = joblib.load('ipo_success_model.pkl')
        scaler = joblib.load('scaler.pkl')
        imputer = joblib.load('imputer.pkl')
        
        with open('feature_columns.pkl', 'rb') as f:
            feature_columns = pickle.load(f)
        
        with open('model_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        return model, scaler, imputer, feature_columns, metadata
    except Exception as e:
        st.error(f"Error loading model files: {str(e)}")
        return None, None, None, None, None

# Prediction function
def predict_ipo_success(ipo_data, model, scaler, imputer, feature_columns):
    """Predict IPO success for new data"""
    try:
        # Create DataFrame from input
        input_df = pd.DataFrame([ipo_data])
        
        # Calculate derived features
        input_df['QIB_to_Total_Ratio'] = input_df['QIB(Qualified Institutional Buyers)'] / input_df['Total Subscription']
        input_df['HNI_to_Total_Ratio'] = input_df['HNI(High Net-Worth Individuals)'] / input_df['Total Subscription']
        input_df['RII_to_Total_Ratio'] = input_df['RII(Retail Individual Investors)'] / input_df['Total Subscription']
        input_df['Strong_Subscription'] = (input_df['Total Subscription'] > 50).astype(int)
        input_df['Very_Strong_Subscription'] = (input_df['Total Subscription'] > 100).astype(int)
        
        threshold = 10
        input_df['Balanced_Subscription'] = (
            (input_df['QIB(Qualified Institutional Buyers)'] > threshold) &
            (input_df['HNI(High Net-Worth Individuals)'] > threshold) &
            (input_df['RII(Retail Individual Investors)'] > threshold)
        ).astype(int)
        
        # Select only required features
        input_features = input_df[feature_columns]
        
        # Handle inf/nan
        input_features = input_features.replace([np.inf, -np.inf], np.nan)
        input_features = pd.DataFrame(imputer.transform(input_features), columns=feature_columns)
        
        # Scale features
        input_scaled = scaler.transform(input_features)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        return {
            'prediction': 'Success' if prediction == 1 else 'Failure',
            'probability_failure': probability[0],
            'probability_success': probability[1],
            'confidence': max(probability)
        }
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

# Main app
def main():
    # Load CSS
    load_css()
    
    # Header
    st.markdown("""
        <div class="header-container">
            <h1 class="header-title">📈 IPO Success Analyzer</h1>
            <p class="header-subtitle">AI-Powered IPO Performance Prediction System</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model, scaler, imputer, feature_columns, metadata = load_model_objects()
    
    if model is None:
        st.error("⚠️ Failed to load model files. Please ensure all model files are in the correct directory.")
        return
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🎯 Prediction", "📊 Batch Analysis", "ℹ️ About"])
    
    with tab1:
        st.markdown("<h2 style='text-align: center; color: #003087;'>Enter IPO Details</h2>", unsafe_allow_html=True)
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 💰 Financial Details", unsafe_allow_html=False)
            issue_size = st.number_input(
                "Issue Size (Crores)",
                min_value=0.0,
                value=1000.0,
                step=100.0,
                help="Total issue size in crores"
            )
            
            offer_price = st.number_input(
                "Offer Price (₹)",
                min_value=0.0,
                value=500.0,
                step=10.0,
                help="Offer price per share"
            )
        
        with col2:
            st.markdown("### 📅 Timeline", unsafe_allow_html=False)
            year = st.number_input(
                "Year",
                min_value=2010,
                max_value=2030,
                value=datetime.now().year,
                step=1
            )
            
            month = st.selectbox(
                "Month",
                options=list(range(1, 13)),
                index=datetime.now().month - 1,
                format_func=lambda x: datetime(2000, x, 1).strftime('%B')
            )
            
            quarter = (month - 1) // 3 + 1
        
        with col3:
            st.markdown("### 📊 Subscription Data", unsafe_allow_html=False)
            total_subscription = st.number_input(
                "Total Subscription (x)",
                min_value=0.0,
                value=50.0,
                step=5.0,
                help="Overall subscription times"
            )
        
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        st.markdown("### 👥 Category-wise Subscription Details", unsafe_allow_html=False)
        
        col4, col5, col6 = st.columns(3)
        
        with col4:
            st.markdown("#### 🏢 QIB", unsafe_allow_html=False)
            qib = st.number_input(
                "QIB Subscription (x)",
                min_value=0.0,
                value=100.0,
                step=5.0,
                help="Qualified Institutional Buyers subscription"
            )
        
        with col5:
            st.markdown("#### 💼 HNI", unsafe_allow_html=False)
            hni = st.number_input(
                "HNI Subscription (x)",
                min_value=0.0,
                value=50.0,
                step=5.0,
                help="High Net-Worth Individuals subscription"
            )
        
        with col6:
            st.markdown("#### 👨‍👩‍👧‍👦 RII", unsafe_allow_html=False)
            rii = st.number_input(
                "RII Subscription (x)",
                min_value=0.0,
                value=20.0,
                step=5.0,
                help="Retail Individual Investors subscription"
            )
        
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        
        # Predict button
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            predict_button = st.button("🔮 Predict IPO Success", use_container_width=True)
        
        if predict_button:
            # Prepare input data
            ipo_data = {
                'Issue_Size(crores)': issue_size,
                'QIB(Qualified Institutional Buyers)': qib,
                'HNI(High Net-Worth Individuals)': hni,
                'RII(Retail Individual Investors)': rii,
                'Total Subscription': total_subscription,
                'Offer Price': offer_price,
                'Year': year,
                'Month': month,
                'Quarter': quarter
            }
            
            # Make prediction
            with st.spinner('🔄 Analyzing IPO data...'):
                result = predict_ipo_success(ipo_data, model, scaler, imputer, feature_columns)
            
            if result:
                st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
                
                # Display result
                if result['prediction'] == 'Success':
                    st.markdown(f"""
                        <div class="success-card">
                            <div class="prediction-title">✅ SUCCESS PREDICTED</div>
                            <div class="prediction-subtitle">This IPO is likely to perform well!</div>
                            <div class="confidence-badge">
                                Confidence: {result['confidence']:.1%}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="failure-card">
                            <div class="prediction-title">⚠️ CAUTION ADVISED</div>
                            <div class="prediction-subtitle">This IPO may underperform</div>
                            <div class="confidence-badge">
                                Confidence: {result['confidence']:.1%}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Probability breakdown
                st.markdown("### 📊 Probability Breakdown")
                col7, col8 = st.columns(2)
                
                with col7:
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{result['probability_success']:.1%}</div>
                            <div class="metric-label">Success Probability</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col8:
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{result['probability_failure']:.1%}</div>
                            <div class="metric-label">Failure Probability</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=result['probability_success'] * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Success Probability", 'font': {'size': 24}},
                    delta={'reference': 50, 'increasing': {'color': "green"}},
                    gauge={
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "darkblue"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 30], 'color': '#ffcdd2'},
                            {'range': [30, 70], 'color': '#fff9c4'},
                            {'range': [70, 100], 'color': '#c8e6c9'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                
                fig.update_layout(
                    height=400,
                    margin=dict(l=20, r=20, t=60, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    font={'color': "darkblue", 'family': "Arial"}
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Subscription analysis
                st.markdown("### 📈 Subscription Analysis")
                
                subscription_data = pd.DataFrame({
                    'Category': ['QIB', 'HNI', 'RII'],
                    'Subscription': [qib, hni, rii]
                })
                
                fig2 = px.bar(
                    subscription_data,
                    x='Category',
                    y='Subscription',
                    color='Subscription',
                    color_continuous_scale='Viridis',
                    title='Category-wise Subscription Pattern'
                )
                
                fig2.update_layout(
                    height=400,
                    showlegend=False,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig2, use_container_width=True)
                
                # Key insights
                st.markdown("### 💡 Key Insights")
                
                col9, col10, col11 = st.columns(3)
                
                with col9:
                    qib_ratio = (qib / total_subscription * 100) if total_subscription > 0 else 0
                    st.markdown(f"""
                        <div class="feature-box">
                            <h4>🏢 QIB Participation</h4>
                            <p><strong>{qib_ratio:.1f}%</strong> of total subscription</p>
                            <p>{'Strong' if qib > 100 else 'Moderate' if qib > 50 else 'Weak'} institutional interest</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col10:
                    st.markdown(f"""
                        <div class="feature-box">
                            <h4>📊 Overall Demand</h4>
                            <p><strong>{total_subscription:.1f}x</strong> subscribed</p>
                            <p>{'Excellent' if total_subscription > 100 else 'Good' if total_subscription > 50 else 'Average'} demand</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col11:
                    balanced = (qib > 10 and hni > 10 and rii > 10)
                    st.markdown(f"""
                        <div class="feature-box">
                            <h4>⚖️ Balance</h4>
                            <p>{'Balanced' if balanced else 'Imbalanced'} subscription</p>
                            <p>{'All categories show interest' if balanced else 'Uneven participation'}</p>
                        </div>
                    """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("<h2 style='text-align: center; color: #003087;'>Batch Analysis</h2>", unsafe_allow_html=True)
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        
        st.markdown("""
            <div class="info-card">
                <h3>📤 Upload CSV File</h3>
                <p>Upload a CSV file with multiple IPO records for batch prediction.</p>
                <p><strong>Required columns:</strong> Issue_Size(crores), QIB(Qualified Institutional Buyers), 
                HNI(High Net-Worth Individuals), RII(Retail Individual Investors), Total Subscription, 
                Offer Price, Year, Month, Quarter</p>
            </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df_batch = pd.read_csv(uploaded_file)
                st.success(f"✅ Successfully loaded {len(df_batch)} records")
                
                st.markdown("### 📋 Preview of Uploaded Data")
                st.dataframe(df_batch.head(), use_container_width=True)
                
                if st.button("🚀 Run Batch Prediction", use_container_width=True):
                    with st.spinner('🔄 Processing batch predictions...'):
                        predictions = []
                        
                        for idx, row in df_batch.iterrows():
                            ipo_data = row.to_dict()
                            result = predict_ipo_success(ipo_data, model, scaler, imputer, feature_columns)
                            if result:
                                predictions.append({
                                    'Prediction': result['prediction'],
                                    'Success_Probability': result['probability_success'],
                                    'Confidence': result['confidence']
                                })
                        
                        df_results = pd.concat([df_batch, pd.DataFrame(predictions)], axis=1)
                        
                        st.markdown("### 📊 Batch Prediction Results")
                        st.dataframe(df_results, use_container_width=True)
                        
                        # Summary statistics
                        col12, col13, col14 = st.columns(3)
                        
                        with col12:
                            success_rate = (df_results['Prediction'] == 'Success').mean() * 100
                            st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-value">{success_rate:.1f}%</div>
                                    <div class="metric-label">Success Rate</div>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        with col13:
                            avg_confidence = df_results['Confidence'].mean() * 100
                            st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-value">{avg_confidence:.1f}%</div>
                                    <div class="metric-label">Avg Confidence</div>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        with col14:
                            total_records = len(df_results)
                            st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-value">{total_records}</div>
                                    <div class="metric-label">Total Records</div>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        # Download results
                        csv = df_results.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name=f"ipo_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
            
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
    
    with tab3:
        st.markdown("<h2 style='text-align: center; color: #003087;'>About the Model</h2>", unsafe_allow_html=True)
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        
        col15, col16 = st.columns(2)
        
        with col15:
            st.markdown("""
                <div class="info-card">
                    <h3>🎯 Model Overview</h3>
                    <p>Our IPO Success Analyzer uses advanced machine learning algorithms to predict 
                    the performance of Initial Public Offerings based on subscription patterns and market dynamics.</p>
                    
                    <h4>Key Features:</h4>
                    <ul>
                        <li>Institutional participation analysis</li>
                        <li>Subscription pattern recognition</li>
                        <li>Market timing considerations</li>
                        <li>Historical performance trends</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        
        with col16:
            st.markdown(f"""
                <div class="info-card">
                    <h3>📊 Technical Details</h3>
                    <p><strong>Algorithm:</strong> {metadata['model_name']}</p>
                    <p><strong>Training Accuracy:</strong> {metadata['accuracy']:.2%}</p>
                    <p><strong>Number of Features:</strong> {len(metadata['features'])}</p>
                    <p><strong>Model Version:</strong> 1.0</p>
                    <p><strong>Last Updated:</strong> {metadata['training_date']}</p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="info-card">
                <h3>📈 Features Used in Prediction</h3>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 1rem;">
        """, unsafe_allow_html=True)
        
        for feature in metadata['features']:
            st.markdown(f"""
                <div class="feature-box">
                    <p style="margin: 0; font-size: 0.9rem;">• {feature}</p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div></div>", unsafe_allow_html=True)
        
        st.markdown("""
            <div class="alert-warning">
                <h3>⚠️ Important Disclaimer</h3>
                <p>This tool provides predictions based on historical data and machine learning models. 
                The predictions are for informational and educational purposes only and should not be 
                considered as financial advice. Always consult with a qualified financial advisor and 
                conduct your own research before making investment decisions.</p>
                
                <p><strong>Past performance does not guarantee future results.</strong></p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="info-card" style="text-align: center; margin-top: 2rem;">
                <h3>👨‍💻 Developed with ❤️</h3>
                <p>IPO Success Analyzer v1.0</p>
                <p>Powered by Machine Learning & Streamlit</p>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()