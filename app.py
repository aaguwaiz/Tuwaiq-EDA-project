import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

st.set_page_config(
    page_title="Saudi Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    h1 { color: #1E3A8A; font-family: 'Helvetica Neue', sans-serif; }
    .stButton>button {
        background-color: #1E3A8A;
        color: white;
        border-radius: 8px;
        width: 100%;
        font-weight: bold;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    model = joblib.load('car_price_model.pkl')
    return model

@st.cache_data
def load_data():
    df = pd.read_csv('saudi_cars_cleaned_for_powerbi.csv')
    return df

try:
    rf_model = load_assets()
    df = load_data()
except Exception as e:
    st.error(f"Error loading assets: {e}")
    st.stop()

st.sidebar.image("https://img.icons8.com/clouds/200/car.png", width=150)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Predict Car Price", "Market Analytics Dashboard"])

brands = sorted(df['Brand'].dropna().unique())
trims = sorted(df['Trim'].dropna().unique())

if page == "Predict Car Price":
    st.title("Predict Car Price with Machine Learning")
    st.write("Enter the vehicle specifications below to estimate its current fair market value in Saudi Arabia.")
    st.write("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_brand = st.selectbox("Brand", brands)
        available_models = sorted(df[df['Brand'] == selected_brand]['Model'].dropna().unique())
        selected_model = st.selectbox("Model", available_models)
        selected_trim = st.selectbox("Trim (Specification Level)", trims)
        
    with col2:
        year = st.number_input("Year of Manufacture", min_value=1980, max_value=2027, value=2020, step=1)
        mileage = st.number_input("Mileage (in km)", min_value=0, max_value=1000000, value=50000, step=5000)
        engine_size = st.number_input("Engine Size (Liters)", min_value=0.5, max_value=10.0, value=2.0, step=0.1)
        is_new = st.checkbox("Is the car brand new?")

    st.write("---")
    
    if st.button("Estimate Price"):
        with st.spinner("Processing specifications and estimating price..."):
            
            try:
                model_features = rf_model.feature_names_in_
            except AttributeError:
                st.error("Model features not compatible.")
                st.stop()
                
            input_df = pd.DataFrame(0.0, index=[0], columns=model_features, dtype=np.float64)
            
            if 'Year' in input_df.columns: input_df.at[0, 'Year'] = float(year)
            if 'Mileage' in input_df.columns: input_df.at[0, 'Mileage'] = float(mileage)
            if 'Engine_Size' in input_df.columns: input_df.at[0, 'Engine_Size'] = float(engine_size)
            if 'Is_New' in input_df.columns: input_df.at[0, 'Is_New'] = 1.0 if is_new else 0.0
            
            for col_name, val in [('Brand', selected_brand), ('Model', selected_model), ('Trim', selected_trim)]:
                specific_col = f"{col_name}_{val}"
                if specific_col in input_df.columns:
                    input_df.at[0, specific_col] = 1.0
            
            try:
                predicted_price = rf_model.predict(input_df)[0]
                
                st.balloons()
                st.markdown(f"""
                    <div style="background-color: #E0F2FE; border-left: 10px solid #0284C7; padding: 25px; border-radius: 8px; text-align: center;">
                        <h2 style="color: #0369A1; margin: 0;">Estimated Market Value:</h2>
                        <h1 style="color: #0369A1; font-size: 50px; margin: 10px 0;">{predicted_price:,.2f} SAR</h1>
                        <p style="color: #0284C7; font-size: 14px; margin: 0;">*This prediction is based on current Saudi car market data analysis.</p>
                    </div>
                """, unsafe_allow_html=True)
            except Exception as predict_error:
                st.error(f"Error during prediction: {predict_error}")

elif page == "Market Analytics Dashboard":
    st.title("Saudi Car Market Dashboard")
    st.write("Visual and interactive market insights extracted from our cleaned local car database.")
    st.write("---")
    
    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: #6B7280; margin: 0;">Total Listed Cars</h4>
                <h2 style="color: #1E3A8A; margin: 10px 0;">{len(df):,}</h2>
            </div>
        """, unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: #6B7280; margin: 0;">Average Car Price</h4>
                <h2 style="color: #10B981; margin: 10px 0;">{df['Price'].mean():,.0f} SAR</h2>
            </div>
        """, unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: #6B7280; margin: 0;">Most Popular Brand</h4>
                <h2 style="color: #F59E0B; margin: 10px 0;">{df['Brand'].mode()[0]}</h2>
            </div>
        """, unsafe_allow_html=True)
        
    st.write("---")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.subheader("Top 10 Most Common Brands")
        brand_counts = df['Brand'].value_counts().head(10).reset_index()
        brand_counts.columns = ['Brand', 'Count']
        fig1 = px.bar(brand_counts, x='Brand', y='Count', color='Count',
                      color_continuous_scale='Blues', labels={'Count': 'Car Count', 'Brand': 'Brand'})
        st.plotly_chart(fig1, use_container_width=True)
        
    with chart_col2:
        st.subheader("Average Price by Origin")
        origin_prices = df.groupby('Origin')['Price'].mean().reset_index().sort_values(by='Price', ascending=False)
        fig2 = px.pie(origin_prices, names='Origin', values='Price', color_discrete_sequence=px.colors.sequential.Blues_r,
                      hole=0.4, labels={'Price': 'Average Price', 'Origin': 'Origin'})
        st.plotly_chart(fig2, use_container_width=True)
        
    st.write("---")

    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        st.subheader("Engine Size Distribution")
        fig3 = px.histogram(df, x='Engine_Size', nbins=20, color_discrete_sequence=['#1E3A8A'],
                            labels={'Engine_Size': 'Engine Size (Liters)'})
        st.plotly_chart(fig3, use_container_width=True)

    with chart_col4:
        st.subheader("Top 10 Most Popular Car Colors")
        color_counts = df['Color'].value_counts().head(10).reset_index()
        color_counts.columns = ['Color', 'Count']
        fig4 = px.bar(color_counts, x='Count', y='Color', orientation='h',
                      color='Count', color_continuous_scale='Blues',
                      labels={'Count': 'Car Count', 'Color': 'Color'})
        fig4.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig4, use_container_width=True)
