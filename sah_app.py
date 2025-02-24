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
        
        # 特征编码验证
        assert set(df['sex_name'].unique()) == {'female', 'male'}, "性别数据异常"
        assert set(df['age_name']) == set(AGE_GROUPS), "年龄组数据异常"
        
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
    models, _ = load_data()

# 输入数据验证
input_data = pd.DataFrame([[
    AGE_GROUPS.index(age),
    0 if sex == '女性' else 1,
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
    st.error(f"预测错误: {str(e)}")
    st.stop()

# -------------------- 结果展示 --------------------
col1, col2, col3 = st.columns([1.2, 1, 1])  # 优化列宽比例
col1.metric("伤残调整生命年 (DALYs)", 
           f"{predictions['DALYs']:,.1f}",
           help="总体疾病负担的衡量指标")
col2.metric("发病率", 
           f"{predictions['Incidence']:.2f}%",
           help="每10万人口新增病例数")
col3.metric("患病率", 
           f"{predictions['Prevalence']:.2f}%",
           help="每10万人口现存病例数")

# -------------------- SHAP解释模块 --------------------
st.divider()
st.header("模型解释")

try:
    # 版本兼容性处理
    plt.switch_backend('agg')
    plt.figure(figsize=(10, 4))
    
    # 使用最新SHAP API
    explainer = shap.Explainer(models['DALYs'])
    shap_values = explainer(input_data)
    
    # 可视化
    shap.plots.bar(shap_values[0], show=False)
    plt.title("特征影响分析", fontsize=14)
    plt.xlabel("SHAP值 (对DALYs的影响)", fontsize=12)
    st.pyplot(plt.gcf())
    
    # 数值表格
    st.subheader("详细影响值")
    df_impact = pd.DataFrame({
        '特征': ['年龄组', '性别', '年份', '人口对数'],
        'SHAP值': shap_values.values[0].tolist(),
        '影响方向': ['风险增加' if x > 0 else '风险降低' for x in shap_values.values[0]]
    })
    st.dataframe(df_impact.style.format({'SHAP值': '{:.4f}'}))
    
    # 动态解释
    with st.expander("解读指南"):
        st.markdown(f"""
        ### 当前参数分析
        - **年龄组**: {age} → 贡献值: `{shap_values.values[0][0]:.4f}`
        - **性别**: {sex} → 贡献值: `{shap_values.values[0][1]:.4f}`
        - **年份**: {year} → 贡献值: `{shap_values.values[0][2]:.4f}`
        - **人口基数**: {population}百万 → 贡献值: `{shap_values.values[0][3]:.4f}`
        
        ### 颜色说明
        - 🔴 正值：增加疾病风险
        - 🔵 负值：降低疾病风险
        """)
        
except Exception as e:
    st.error(f"""
    SHAP可视化失败: {str(e)}
    
    **常见解决方法**
    1. 升级SHAP库: `pip install --upgrade shap`
    2. 检查输入数据格式:
       - 年龄代码: {input_data['age_code'].values}
       - 性别代码: {input_data['sex_code'].values}
       - 年份: {year}
       - 人口对数: {log_pop:.2f}
    3. 验证模型特征是否匹配
    """)
