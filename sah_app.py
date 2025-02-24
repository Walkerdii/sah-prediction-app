import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

# -------------------- 配置 --------------------
DATA_URL = "https://raw.githubusercontent.com/Walkerdii/sah-prediction-app/main/Merged_Data.xlsx"
AGE_GROUPS = ['15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49']

# -------------------- 数据处理（优化版本）--------------------
@st.cache_data(ttl=3600)  # 缓存1小时
def load_data(_progress):
    """带进度提示的数据加载"""
    try:
        # 第一阶段：加载数据 (25%)
        _progress.progress(25, text="📥 Downloading dataset...")
        df = pd.read_excel(DATA_URL, engine='openpyxl')
        
        # 第二阶段：特征编码 (50%)
        _progress.progress(50, text="🔧 Processing features...")
        df['age_code'] = df['age_name'].map({age: idx for idx, age in enumerate(AGE_GROUPS)})
        df['sex_code'] = df['sex_name'].map({'female': 0, 'male': 1})
        
        # 第三阶段：训练模型 (75%)
        _progress.progress(75, text="🤖 Training models...")
        features = df[['age_code', 'sex_code', 'year', 'log_population']]
        
        # 使用轻量级模型参数
        model_params = {
            'n_estimators': 50,
            'max_depth': 3,
            'learning_rate': 0.1
        }
        
        models = {
            'DALYs': XGBRegressor(**model_params).fit(features, df['DALYs']),
            'Incidence': XGBRegressor(**model_params).fit(features, df['Incidence']),
            'Prevalence': XGBRegressor(**model_params).fit(features, df['Prevalence'])
        }
        
        _progress.progress(100)
        return models
        
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        st.stop()

# -------------------- 界面 --------------------
st.set_page_config(page_title="SAH Prediction System", layout="wide")
st.title("Subarachnoid Hemorrhage Risk Prediction")

# 初始化加载状态
if 'loaded' not in st.session_state:
    with st.container():
        st.markdown("""
        <style>
            .stProgress > div > div > div {
                background-color: #1f77b4;
            }
        </style>
        """, unsafe_allow_html=True)
        
        progress_bar = st.progress(0, text="🚀 Initializing system...")
        models = load_data(progress_bar)
        st.session_state.models = models
        st.session_state.loaded = True
        progress_bar.empty()
else:
    models = st.session_state.models

# 侧边栏输入
with st.sidebar:
    st.header("⚙️ Prediction Parameters")
    age = st.selectbox("Age Group", AGE_GROUPS)
    sex = st.radio("Sex", ['Female', 'Male'])
    year = st.slider("Year", 1990, 2050, 2023)  # 扩展到2050年
    population = st.number_input(
        "Population (Millions)", 
        min_value=1,
        value=10,
        help="Actual population = input value × 1,000,000"
    )
    log_pop = np.log(population * 1_000_000)

# 准备输入数据
input_data = pd.DataFrame([[
    AGE_GROUPS.index(age),
    0 if sex == 'Female' else 1,
    year,
    log_pop
]], columns=['age_code', 'sex_code', 'year', 'log_population'])

# 执行预测
try:
    predictions = {
        'DALYs': models['DALYs'].predict(input_data)[0],
        'Incidence': models['Incidence'].predict(input_data)[0],
        'Prevalence': models['Prevalence'].predict(input_data)[0]
    }
except Exception as e:
    st.error(f"Prediction error: {str(e)}")
    st.stop()

# 显示结果
col1, col2, col3 = st.columns(3)
col1.metric("DALYs", f"{predictions['DALYs']:,.1f}")
col2.metric("Incidence Rate", f"{predictions['Incidence']:.2f}%")
col3.metric("Prevalence Rate", f"{predictions['Prevalence']:.2f}%")

# 预测年份提示
if year > 2030:
    st.info("ℹ️ Note: Predictions beyond 2030 are extrapolations and should be interpreted with caution.")

# SHAP解释
st.divider()
st.header("Model Interpretation")

try:
    explainer = shap.TreeExplainer(models['DALYs'])
    shap_values = explainer.shap_values(input_data)
    
    plt.switch_backend('agg')
    fig, ax = plt.subplots()
    shap.force_plot(
        explainer.expected_value,
        shap_values[0],
        input_data.iloc[0],
        feature_names=['Age Group', 'Sex', 'Year', 'Log Population'],
        matplotlib=True,
        show=False
    )
    st.pyplot(fig)
    
except Exception as e:
    st.warning(f"SHAP visualization unavailable: {str(e)}")
