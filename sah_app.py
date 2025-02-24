import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

# -------------------- 配置 --------------------
DATA_URL = "https://raw.githubusercontent.com/Walkerdii/sah-prediction-app/main/Merged_Data.xlsx"
AGE_GROUPS = ['15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49']

# -------------------- 数据处理 --------------------
@st.cache_data
def load_data():
    """加载数据并训练模型"""
    try:
        df = pd.read_excel(DATA_URL, engine='openpyxl')
        
        # 特征编码
        df['age_code'] = df['age_name'].map({age: idx for idx, age in enumerate(AGE_GROUPS)})
        df['sex_code'] = df['sex_name'].map({'female': 0, 'male': 1})
        
        # 训练模型
        features = df[['age_code', 'sex_code', 'year', 'log_population']]
        return {
            'DALYs': XGBRegressor().fit(features, df['DALYs']),
            'Incidence': XGBRegressor().fit(features, df['Incidence']),
            'Prevalence': XGBRegressor().fit(features, df['Prevalence'])
        }, df
        
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        st.stop()

# -------------------- 界面 --------------------
st.set_page_config(page_title="SAH Prediction System", layout="wide")
st.title("Subarachnoid Hemorrhage Risk Prediction")
st.caption("Data Source: GBD Database | Developer: Walkerdii")

# 侧边栏输入
with st.sidebar:
    st.header("⚙️ Prediction Parameters")
    age = st.selectbox("Age Group", AGE_GROUPS)
    sex = st.radio("Sex", ['Female', 'Male'])
    year = st.slider("Year", 1990, 2050, 2023)
    population = st.number_input(
        "Population (Millions)", 
        min_value=1,
        value=10,
        help="Actual population = input value × 1,000,000"
    )
    log_pop = np.log(population * 1_000_000)

# -------------------- 模型预测 --------------------
with st.spinner('Loading data and training models...'):
    models, _ = load_data()

input_data = pd.DataFrame([[
    AGE_GROUPS.index(age),
    0 if sex == 'Female' else 1,
    year,
    log_pop
]], columns=['age_code', 'sex_code', 'year', 'log_population'])

try:
    predictions = {
        'DALYs': models['DALYs'].predict(input_data)[0],
        'Incidence': models['Incidence'].predict(input_data)[0],
        'Prevalence': models['Prevalence'].predict(input_data)[0]
    }
except Exception as e:
    st.error(f"Prediction error: {str(e)}")
    st.stop()

# 调整列宽比例解决遮挡问题
col1, col2, col3 = st.columns([2, 2.5, 2])  # 增加中间列的宽度
with col1:
    st.metric("DALYs (Disability-Adjusted Life Years)", 
            f"{predictions['DALYs']:,.1f}",
            help="Measure of overall disease burden")
with col2:
    st.metric("Incidence Rate", 
            f"{predictions['Incidence']:.2f}%",
            help="New cases per 100,000 population")
with col3:
    st.metric("Prevalence Rate", 
            f"{predictions['Prevalence']:.2f}%",
            help="Total cases per 100,000 population")

# -------------------- 改进的SHAP模型解释 --------------------
st.divider()
st.header("Model Interpretation")

try:
    explainer = shap.TreeExplainer(models['DALYs'])
    shap_values = explainer(input_data)
    
    # 使用summary_plot替代force_plot
    plt.switch_backend('agg')
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.summary_plot(
        shap_values.values, 
        input_data,
        feature_names=['Age Group', 'Sex', 'Year', 'Log Population'],
        plot_type="bar",
        show=False
    )
    plt.tight_layout()
    
    st.pyplot(fig)
    
    with st.expander("How to interpret this plot?"):
        st.markdown("""
        - **Bar length**: Shows feature importance magnitude
        - **Color**: Indicates impact direction (red=positive, blue=negative)
        - Values show actual contribution to prediction
        """)
        
except Exception as e:
    st.warning(f"SHAP visualization unavailable: {str(e)}")
