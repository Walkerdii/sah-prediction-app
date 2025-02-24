import os
import numpy as np
import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

# -------------------- 常量配置 --------------------
GITHUB_DATA_URL = "https://raw.githubusercontent.com/Walkerdii/sah-prediction-app/main/Merged_Data.xlsx"
AGE_GROUPS = ['15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49']

# -------------------- 数据加载与预处理 --------------------
@st.cache_data
def load_data():
    """从GitHub加载数据并训练模型"""
    try:
        # 从GitHub原始数据地址加载
        data = pd.read_excel(GITHUB_DATA_URL, engine='openpyxl')
        
        # 检查必要列是否存在
        required_columns = ['age_name', 'sex_name', 'year', 'population', 
                           'DALYs', 'Incidence', 'Prevalence']
        missing = [col for col in required_columns if col not in data.columns]
        if missing:
            raise ValueError(f"数据缺失必要列: {missing}")

        # 数据预处理
        data['log_population'] = np.log(data['population'])
        data['age_code'] = data['age_name'].map({age: idx for idx, age in enumerate(AGE_GROUPS)})
        data['sex_code'] = data['sex_name'].map({'female': 0, 'male': 1})

        # 训练模型
        features = data[['age_code', 'sex_code', 'year', 'log_population']]
        targets = data[['DALYs', 'Incidence', 'Prevalence']]
        
        models = {
            'DALYs': XGBRegressor().fit(features, targets['DALYs']),
            'Incidence': XGBRegressor().fit(features, targets['Incidence']),
            'Prevalence': XGBRegressor().fit(features, targets['Prevalence'])
        }
        
        return models, data
        
    except Exception as e:
        st.error(f"数据加载失败: {str(e)}")
        st.stop()

# -------------------- 用户界面 --------------------
st.set_page_config(page_title="SAH预测系统", layout="wide")
st.title("蛛网膜下腔出血风险预测系统")
st.markdown("""[![GitHub](https://img.shields.io/badge/源代码-GitHub-blue?logo=github)](https://github.com/Walkerdii/sah-prediction-app)""")

# 侧边栏输入
with st.sidebar:
    st.header("⚙️ 预测参数")
    age_group = st.selectbox("年龄组", options=AGE_GROUPS)
    sex = st.radio("性别", options=['女性', '男性'], format_func=lambda x: '♀️' if x == '女性' else '♂️')
    year = st.slider("预测年份", 2000, 2030, 2023)
    
    # 新增人口输入转换
    population = st.number_input(
        "人口数量", 
        min_value=1,  # 避免0或负数
        value=100000,
        help="输入实际人口数量，系统会自动计算对数人口"
    )
    log_population = np.log(population)

# -------------------- 模型预测 --------------------
try:
    models, raw_data = load_data()
    
    # 转换输入格式
    input_features = pd.DataFrame([[
        AGE_GROUPS.index(age_group),
        0 if sex == '女性' else 1,
        year,
        log_population
    ]], columns=['age_code', 'sex_code', 'year', 'log_population'])
    
    # 执行预测
    predictions = {
        'DALYs': models['DALYs'].predict(input_features)[0],
        'Incidence': models['Incidence'].predict(input_features)[0],
        'Prevalence': models['Prevalence'].predict(input_features)[0]
    }
    
    # 展示结果
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("伤残调整生命年 (DALYs)", f"{predictions['DALYs']:,.2f}")
    with col2:
        st.metric("发病率", f"{predictions['Incidence']:.2%}") 
    with col3:
        st.metric("患病率", f"{predictions['Prevalence']:.2%}")

    # -------------------- 数据分布可视化 --------------------
    st.divider()
    st.header("📊 数据分布")
    
    tab1, tab2 = st.tabs(["年龄分布", "人口分布"])
    
    with tab1:
        fig, ax = plt.subplots()
        raw_data['age_name'].value_counts().sort_index().plot.bar(ax=ax)
        ax.set_xlabel("年龄组")
        ax.set_ylabel("样本数量")
        st.pyplot(fig)
        
    with tab2:
        fig, ax = plt.subplots()
        ax.hist(raw_data['population'], bins=30, edgecolor='k')
        ax.set_xlabel("人口数量")
        ax.set_ylabel("频次")
        st.pyplot(fig)
        
    # -------------------- SHAP解释 --------------------
    st.divider()
    st.header("🔍 模型解释")
    
    explainer = shap.TreeExplainer(models['DALYs'])
    shap_values = explainer.shap_values(input_features)
    
    fig, ax = plt.subplots()
    shap.force_plot(
        explainer.expected_value,
        shap_values[0], 
        input_features.iloc[0],
        feature_names=['年龄组', '性别', '年份', '对数人口'],
        matplotlib=True,
        show=False
    )
    st.pyplot(fig)
    
except Exception as e:
    st.error(f"系统错误: {str(e)}")
    st.write("请检查输入参数或联系开发人员")
