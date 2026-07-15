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
    df = pd.read_csv('saudi_cars_cleaned.csv')
    return df

try:
    rf_model = load_assets()
    df = load_data()
except Exception as e:
    st.error(f"⚠️ Error: {e}")
    st.stop()

st.sidebar.image("https://img.icons8.com/clouds/200/car.png", width=150)
st.sidebar.title("🎮 لوحة التحكم")
page = st.sidebar.radio("انتقل إلى:", ["🔮 توقع سعر سيارتك", "📊 داشبورد تحليل السوق"])

brands = sorted(df['Brand'].dropna().unique())
regions = sorted(df['Region'].dropna().unique())
colors = sorted(df['Color'].dropna().unique())
origins = sorted(df['Origin'].dropna().unique())
trims = sorted(df['Trim'].dropna().unique())

if page == "🔮 توقع سعر سيارتك":
    st.title("🔮 توقع سعر السيارة بالذكاء الاصطناعي")
    st.write("أدخل مواصفات سيارتك بدقة لمعرفة قيمتها العادلة في السوق السعودي حالياً.")
    st.write("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_brand = st.selectbox("الماركة (Brand)", brands)
        available_models = sorted(df[df['Brand'] == selected_brand]['Model'].dropna().unique())
        selected_model = st.selectbox("الموديل (Model)", available_models)
        year = st.number_input("سنة الصنع (Year)", min_value=1980, max_value=2027, value=2020, step=1)
        
    with col2:
        mileage = st.number_input("الممشى بالكيلومتر (Mileage)", min_value=0, max_value=1000000, value=50000, step=5000)
        engine_size = st.number_input("سعة المحرك باللتر (Engine Size)", min_value=0.5, max_value=10.0, value=2.0, step=0.1)
        region = st.selectbox("المنطقة (Region)", regions)

    with col3:
        origin = st.selectbox("المنشأ (Origin)", origins)
        color = st.selectbox("اللون (Color)", colors)
        trim = st.selectbox("الفئة (Trim)", trims)
        
    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        is_new = st.checkbox("هل السيارة جديدة كلياً؟ (Is New)")
    with col_opt2:
        negotiable = st.checkbox("هل السعر قابل للتفاوض؟ (Negotiable)")

    st.write("---")
    
    if st.button("احسب السعر المتوقع 🚀"):
        with st.spinner("جاري حساب السعر..."):
            
            # هنا السحر! نجلب الـ 438 عمود الأصلية مباشرة من داخل الموديل المدرب
            try:
                model_features = rf_model.feature_names_in_
            except AttributeError:
                st.error("⚠️ الموديل غير متوافق مع جلب الأعمدة تلقائياً. تأكد من حفظه بعد التدريب مباشرة.")
                st.stop()
                
            input_df = pd.DataFrame(0, index=[0], columns=model_features)
            
            # تعبئة القيم الرقمية الصريحة
            if 'Year' in input_df.columns: input_df.at[0, 'Year'] = year
            if 'Mileage' in input_df.columns: input_df.at[0, 'Mileage'] = mileage
            if 'Engine_Size' in input_df.columns: input_df.at[0, 'Engine_Size'] = engine_size
            if 'Is_New' in input_df.columns: input_df.at[0, 'Is_New'] = 1 if is_new else 0
            if 'Negotiable' in input_df.columns: input_df.at[0, 'Negotiable'] = 1 if negotiable else 0
            
            # تشفير المدخلات النصية ديناميكياً لتطابق الموديل
            for col_name, val in [('Brand', selected_brand), ('Model', selected_model), 
                                  ('Region', region), ('Origin', origin), 
                                  ('Color', color), ('Trim', trim)]:
                specific_col = f"{col_name}_{val}"
                if specific_col in input_df.columns:
                    input_df.at[0, specific_col] = 1
            
            try:
                predicted_price = rf_model.predict(input_df)[0]
                
                st.balloons()
                st.markdown(f"""
                    <div style="background-color: #E0F2FE; border-left: 10px solid #0284C7; padding: 25px; border-radius: 8px; text-align: center;">
                        <h2 style="color: #0369A1; margin: 0;">السعر المتوقع التقريبي لسيارتك هو:</h2>
                        <h1 style="color: #0369A1; font-size: 50px; margin: 10px 0;">{predicted_price:,.2f} ر.س</h1>
                        <p style="color: #0284C7; font-size: 14px; margin: 0;">*هذا السعر تقريبي ومبني على تحليل السوق المحلي السعودي الحالي.</p>
                    </div>
                """, unsafe_allow_html=True)
            except Exception as predict_error:
                st.error(f"⚠️ Error during prediction: {predict_error}")

elif page == "📊 داشبورد تحليل السوق":
    st.title("📊 داشبورد تحليل سوق السيارات السعودي")
    st.write("نظرة عامة وإحصائيات تفاعلية مستخرجة مباشرة من قاعدة بيانات السيارات.")
    st.write("---")
    
    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: #6B7280; margin: 0;">إجمالي السيارات</h4>
                <h2 style="color: #1E3A8A; margin: 10px 0;">{len(df):,} سيارة</h2>
            </div>
        """, unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: #6B7280; margin: 0;">متوسط الأسعار بالداتا</h4>
                <h2 style="color: #10B981; margin: 10px 0;">{df['Price'].mean():,.0f} ر.س</h2>
            </div>
        """, unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: #6B7280; margin: 0;">أكثر ماركة طلباً وشهرة</h4>
                <h2 style="color: #F59E0B; margin: 10px 0;">{df['Brand'].mode()[0]}</h2>
            </div>
        """, unsafe_allow_html=True)
        
    st.write("---")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.subheader("📈 أكثر 10 ماركات سيارات معروضة في السوق")
        brand_counts = df['Brand'].value_counts().head(10).reset_index()
        brand_counts.columns = ['Brand', 'Count']
        fig1 = px.bar(brand_counts, x='Brand', y='Count', color='Count',
                      color_continuous_scale='Blues', labels={'Count': 'عدد السيارات', 'Brand': 'الماركة'})
        st.plotly_chart(fig1, use_container_width=True)
        
    with chart_col2:
        st.subheader("💰 متوسط الأسعار حسب بلد المنشأ (Origin)")
        origin_prices = df.groupby('Origin')['Price'].mean().reset_index().sort_values(by='Price', ascending=False)
        fig2 = px.pie(origin_prices, names='Origin', values='Price', color_discrete_sequence=px.colors.sequential.Blues_r,
                      hole=0.4, labels={'Price': 'متوسط السعر', 'Origin': 'المنشأ'})
        st.plotly_chart(fig2, use_container_width=True)
        
    st.write("---")
    
    st.subheader("🚗 العلاقة بين ممشى السيارة (Mileage) وسعرها")
    sample_size = min(2000, len(df))
    scatter_data = df.sample(n=sample_size, random_state=42)
    fig3 = px.scatter(scatter_data, x='Mileage', y='Price', color='Brand',
                      trendline="ols", trendline_color_override="red")
    st.plotly_chart(fig3, use_container_width=True)
