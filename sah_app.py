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
        
        # 特征工程
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
        st.error(f"数据加载失败: {str(e)}")
        st.stop()

# -------------------- 界面 --------------------
st.set_page_config(page_title="SAH预测系统", layout="wide")
st.title("蛛网膜下腔出血风险预测")
st.caption("数据来源: GBD数据库 | 开发者: Walkerdii")

# 侧边栏输入
with st.sidebar:
    st.header("⚙️ 预测参数")
    age = st.selectbox("年龄组", AGE_GROUPS)
    sex = st.radio("性别", ['女性', '男性'])
    year = st.slider("年份", 1990, 2030, 2023)
    population = st.number_input("人口数量（百万）", 1, 1000, 10)
    log_pop = np.log(population * 1_000_000)  # 自动转换对数人口

# -------------------- 模型预测 --------------------
models, raw_data = load_data()

# 准备输入数据
input_data = pd.DataFrame([[
    AGE_GROUPS.index(age),
    0 if sex == '女性' else 1,
    year,
    log_pop
]], columns=['age_code', 'sex_code', 'year', 'log_population'])

# 执行预测
predictions = {
    'DALYs': models['DALYs'].predict(input_data)[0],
    'Incidence': models['Incidence'].predict(input_data)[0],
    'Prevalence': models['Prevalence'].predict(input_data)[0]
}

# 显示结果
col1, col2, col3 = st.columns(3)
col1.metric("伤残生命年 (DALYs)", f"{predictions['DALYs']:,.1f}")
col2.metric("发病率", f"{predictions['Incidence']:.2f}%")
col3.metric("患病率", f"{predictions['Prevalence']:.2f}%")

# -------------------- 模型解释 --------------------
st.divider()
st.header("模型解释")

try:
    explainer = shap.TreeExplainer(models['DALYs'])
    shap_values = explainer.shap_values(input_data)
    
    fig, ax = plt.subplots()
    shap.force_plot(
        explainer.expected_value,
        shap_values[0],
        input_data.iloc[0],
        feature_names=['年龄组', '性别', '年份', '对数人口'],
        matplotlib=True,
        show=False
    )
    st.pyplot(fig)
    
except Exception as e:
    st.warning(f"SHAP解释不可用: {str(e)}")

# -------------------- 数据分布 --------------------
st.divider()
st.header("数据分布可视化")

tab1, tab2 = st.tabs(["年龄分布", "人口分布"])
with tab1:
    fig, ax = plt.subplots(figsize=(10, 4))
    raw_data['age_name'].value_counts().plot.bar(ax=ax)
    ax.set_title("各年龄组数据量")
    st.pyplot(fig)

with tab2:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(raw_data['log_population'], bins=30)
    ax.set_title("对数人口分布")
    st.pyplot(fig)
