import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

# -------------------- 配置 --------------------
AGE_GROUPS = ['15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49']

# -------------------- 数据处理 --------------------
@st.cache_data
def load_data():
    """加载并处理数据"""
    try:
        # 从本地文件读取数据（或修改为URL）
        df = pd.read_excel("Merged_Data.xlsx", sheet_name="Merged_Data", engine='openpyxl')
        
        # 数据清洗：处理年龄组格式（去除' years'）
        df['age_group'] = df['age_name'].str.replace(' years', '').str.strip()
        
        # 验证年龄组
        invalid_age = ~df['age_group'].isin(AGE_GROUPS)
        if invalid_age.any():
            st.error(f"Invalid age groups detected: {df.loc[invalid_age, 'age_group'].unique()}")
            st.stop()
            
        # 处理性别（统一为小写）
        df['sex'] = df['sex_name'].str.lower().str.strip()
        valid_sex = df['sex'].isin(['female', 'male'])
        if not valid_sex.all():
            invalid_sex = df.loc[~valid_sex, 'sex'].unique()
            st.error(f"Invalid gender values: {invalid_sex}")
            st.stop()
        
        # 特征编码
        df['age_code'] = df['age_group'].map({age: idx for idx, age in enumerate(AGE_GROUPS)})
        df['sex_code'] = df['sex'].map({'female': 0, 'male': 1})
        
        # 训练模型：使用对应特征进行训练
        features = df[['age_code', 'sex_code', 'year', 'log_population']]
        models = {
            'DALYs': XGBRegressor().fit(features, df['DALYs']),
            'Incidence': XGBRegressor().fit(features, df['Incidence']),
            'Prevalence': XGBRegressor().fit(features, df['Prevalence'])
        }
        
        return models, df
        
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        st.stop()

# -------------------- 页面设置 --------------------
st.set_page_config(
    page_title="SAH Predictor: Interactive Burden Forecasting with Explainable AI",
    layout="wide"
)
st.title("SAH Predictor: Interactive Burden Forecasting with Explainable AI")
st.subheader("An XGBoost-based prediction system for subarachnoid hemorrhage outcomes with SHAP interpretation (1990-2050)")
st.caption("Data Source: GBD Database | Developer: Walkerdii")

# -------------------- 侧边栏输入 --------------------
with st.sidebar:
    st.header("⚙️ Prediction Parameters")
    age = st.selectbox("Age Group", AGE_GROUPS)
    sex = st.radio("Gender", ['Female', 'Male'])
    year = st.slider("Year", 1990, 2050, 2023)
    population = st.number_input(
        "Population (Millions)", 
        min_value=1,
        value=10,
        help="Actual population = Input value × 1,000,000"
    )
    log_pop = np.log(population * 1_000_000)

# -------------------- 模型预测 --------------------
with st.spinner('Loading data and training models...'):
    models, df = load_data()

# 构造输入数据
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

# -------------------- 结果展示 --------------------
col1, col2, col3 = st.columns(3)
col1.metric("Disability-Adjusted Life Years (DALYs)", f"{predictions['DALYs']:,.1f}", help="Measure of overall disease burden")
col2.metric("Incidence Rate", f"{predictions['Incidence']:.2f}%", help="New cases per 100,000 population")
col3.metric("Prevalence Rate", f"{predictions['Prevalence']:.2f}%", help="Existing cases per 100,000 population")

# -------------------- SHAP解释模块 --------------------
st.divider()
st.header("🧠 Model Interpretation")

try:
    explainer = shap.Explainer(models['DALYs'])
    shap_values = explainer(input_data)
    
    plt.figure(figsize=(10, 4))
    shap.plots.bar(shap_values[0], show=False)
    plt.title("Feature Impact Analysis", fontsize=10)
    plt.xlabel("SHAP Value (Impact on DALYs)", fontsize=8)
    st.pyplot(plt.gcf())
    
    # 数值表格展示详细的SHAP影响值
    st.subheader("Detailed Impact Values")
    df_impact = pd.DataFrame({
        'Feature': ['Age Group', 'Gender', 'Year', 'Log Population'],
        'SHAP Value': shap_values.values[0].tolist(),
        'Impact Direction': ['Risk Increase' if x > 0 else 'Risk Decrease' for x in shap_values.values[0]]
    })
    st.dataframe(df_impact.style.format({'SHAP Value': '{:.4f}'}))
    
except Exception as e:
    st.error(f"SHAP interpretation failed: {str(e)}")

# -------------------- 调试信息 --------------------
with st.expander("🔍 Data Validation Info"):
    st.write("Data Sample:", df[['age_group', 'sex', 'year', 'log_population']].head(2))
    st.write("Age Distribution:", df['age_group'].value_counts())
    st.write("Gender Distribution:", df['sex'].value_counts())
