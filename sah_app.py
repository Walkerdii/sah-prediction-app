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
        # 从URL读取数据
        df = pd.read_excel("Merged_Data.xlsx", sheet_name="Merged_Data", engine='openpyxl')
        
        # 数据清洗
        # 1. 处理年龄组格式（去除'years'）
        df['age_group'] = df['age_name'].str.replace(' years', '').str.strip()
        
        # 2. 验证年龄组
        invalid_age = ~df['age_group'].isin(AGE_GROUPS)
        if invalid_age.any():
            st.error(f"发现无效年龄组: {df.loc[invalid_age, 'age_group'].unique()}")
            st.stop()
            
        # 3. 处理性别（统一为小写）
        df['sex'] = df['sex_name'].str.lower().str.strip()
        valid_sex = df['sex'].isin(['female', 'male'])
        if not valid_sex.all():
            invalid_sex = df.loc[~valid_sex, 'sex'].unique()
            st.error(f"发现无效性别: {invalid_sex}")
            st.stop()
        
        # 特征编码
        df['age_code'] = df['age_group'].map({age: idx for idx, age in enumerate(AGE_GROUPS)})
        df['sex_code'] = df['sex'].map({'female': 0, 'male': 1})
        
        # 验证特征列
        features = df[['age_code', 'sex_code', 'year', 'log_population']]
        targets = ['DALYs', 'Incidence', 'Prevalence']
        
        # 训练模型
        models = {
            'DALYs': XGBRegressor().fit(features, df['DALYs']),
            'Incidence': XGBRegressor().fit(features, df['Incidence']),
            'Prevalence': XGBRegressor().fit(features, df['Prevalence'])
        }
        
        return models, df
        
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
    year = st.slider("年份", 1990, 2050, 2023)
    population = st.number_input(
        "人口数量 (百万)", 
        min_value=1,
        value=10,
        help="实际人口 = 输入值 × 1,000,000"
    )
    log_pop = np.log(population * 1_000_000)

# -------------------- 模型预测 --------------------
with st.spinner('正在加载数据和训练模型...'):
    models, df = load_data()

# 构造输入数据
input_data = pd.DataFrame([[
    AGE_GROUPS.index(age),
    0 if sex == '女性' else 1,
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
    st.error(f"预测错误: {str(e)}")
    st.stop()

# -------------------- 结果展示 --------------------
col1, col2, col3 = st.columns(3)
col1.metric("伤残调整生命年 (DALYs)", f"{predictions['DALYs']:,.1f}", help="总体疾病负担")
col2.metric("发病率", f"{predictions['Incidence']:.2f}%", help="每10万人口新增病例")
col3.metric("患病率", f"{predictions['Prevalence']:.2f}%", help="每10万人口现存病例")

# -------------------- SHAP解释模块 --------------------
st.divider()
st.header("模型解释")

try:
    explainer = shap.Explainer(models['DALYs'])
    shap_values = explainer(input_data)
    
    plt.figure(figsize=(10, 4))
    shap.plots.bar(shap_values[0], show=False)
    plt.title("特征影响分析", fontsize=14)
    plt.xlabel("SHAP值 (对DALYs的影响)", fontsize=12)
    st.pyplot(plt.gcf())
    
    # 数值表格
    st.subheader("详细影响值")
    df_impact = pd.DataFrame({
        '特征': ['年龄组', '性别', '年份', '人口对数'],
        'SHAP值': shap_values.values[0].tolist(),
        '影响方向': ['增加风险' if x > 0 else '降低风险' for x in shap_values.values[0]]
    })
    st.dataframe(df_impact.style.format({'SHAP值': '{:.4f}'}))
    
except Exception as e:
    st.error(f"SHAP解释失败: {str(e)}")

# 调试信息
with st.expander("数据验证信息"):
    st.write("数据样例:", df[['age_group', 'sex', 'year', 'log_population']].head(2))
    st.write("年龄分布:", df['age_group'].value_counts())
    st.write("性别分布:", df['sex'].value_counts())
