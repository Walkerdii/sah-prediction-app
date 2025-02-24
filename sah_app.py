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
    # 使用DALYs模型作为示例
    explainer = shap.TreeExplainer(models['DALYs'])
    
    # 计算完整SHAP值（包含基值）
    shap_values = explainer.shap_values(input_data)
    
    # 调试输出原始SHAP值
    st.write("Raw SHAP values:", shap_values)
    
    # 创建可视化容器
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 第一子图：条形图显示绝对影响
    shap.summary_plot(
        shap_values, 
        input_data,
        feature_names=['Age Group', 'Sex', 'Year', 'Log Population'],
        plot_type="bar",
        max_display=10,  # 强制显示所有特征
        show=False,
        color_bar=False,
        ax=ax1
    )
    ax1.set_title("Feature Importance (Absolute Impact)")
    
    # 第二子图：小提琴图显示方向性影响
    shap.summary_plot(
        shap_values,
        input_data,
        feature_names=['Age Group', 'Sex', 'Year', 'Log Population'],
        plot_type="violin",
        show=False,
        ax=ax2
    )
    ax2.set_title("Feature Impact Direction")
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # 添加数值表格展示
    st.subheader("Detailed SHAP Values")
    shap_table = pd.DataFrame({
        'Feature': ['Age Group', 'Sex', 'Year', 'Log Population'],
        'SHAP Value': shap_values[0],
        'Impact Direction': ['Positive' if val > 0 else 'Negative' for val in shap_values[0]]
    })
    st.dataframe(shap_table.style.format({'SHAP Value': '{:.4f}'}))
    
    # 添加动态解释
    age_impact = shap_values[0][0]
    sex_impact = shap_values[0][1]
    
    with st.expander("Interpretation Guidance"):
        st.markdown(f"""
        ### 特征影响分析：
        - **年龄组**贡献值：`{age_impact:.4f}`
          - 当前选择：{age}
          - 影响方向：{'增加风险' if age_impact > 0 else '降低风险'}
        
        - **性别**贡献值：`{sex_impact:.4f}`
          - 当前选择：{sex}
          - 影响方向：{'增加风险' if sex_impact > 0 else '降低风险'}
        
        ### 解读原则：
        1. 正值（红色）表示提升风险指标
        2. 负值（蓝色）表示降低风险指标
        3. 绝对值越大表示影响越显著
        """)
    
except Exception as e:
    st.warning(f"SHAP visualization error: {str(e)}")
