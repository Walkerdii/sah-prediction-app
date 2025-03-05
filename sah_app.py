import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import os
from xgboost import XGBRegressor
from datetime import datetime

# -------------------- 全局配置 --------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "sah_models")  # 使用绝对路径
AGE_GROUPS = ['15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49']
os.makedirs(MODEL_PATH, exist_ok=True)  # 自动创建模型目录

# -------------------- 模型加载 --------------------
@st.cache_resource
def load_models():
    """加载预训练模型"""
    try:
        # 验证模型文件存在性
        required_models = ['DALYs.json', 'Incidence.json', 'Prevalence.json']
        missing = [m for m in required_models if not os.path.exists(os.path.join(MODEL_PATH, m))]
        
        if missing:
            st.error(f"缺少模型文件: {missing}\n请先执行 model_trainer.py 进行训练")
            st.stop()

        return {
            'DALYs': XGBRegressor().load_model(os.path.join(MODEL_PATH, 'DALYs.json')),
            'Incidence': XGBRegressor().load_model(os.path.join(MODEL_PATH, 'Incidence.json')),
            'Prevalence': XGBRegressor().load_model(os.path.join(MODEL_PATH, 'Prevalence.json'))
        }
    except Exception as e:
        st.error(f"模型加载失败: {str(e)}")
        st.stop()

# -------------------- 数据处理 --------------------
@st.cache_data
def load_dataset():
    """加载并验证数据集"""
    try:
        df = pd.read_excel("Merged_Data.xlsx", 
                          sheet_name="Merged_Data", 
                          engine='openpyxl',
                          dtype={'year': int})
        
        # 数据验证
        age_validation = df['age_name'].str.replace(' years', '').isin(AGE_GROUPS)
        if not age_validation.all():
            invalid_age = df.loc[~age_validation, 'age_name'].unique()
            st.error(f"无效年龄组: {invalid_age}")
            st.stop()
            
        gender_validation = df['sex_name'].str.lower().isin(['female', 'male'])
        if not gender_validation.all():
            invalid_gender = df.loc[~gender_validation, 'sex_name'].unique()
            st.error(f"无效性别: {invalid_gender}")
            st.stop()
            
        # 特征编码（必须与训练时一致）
        df['age_code'] = df['age_name'].str.replace(' years', '').map(
            {age: idx for idx, age in enumerate(AGE_GROUPS)}
        )
        df['sex_code'] = df['sex_name'].str.lower().map({'female': 0, 'male': 1})
        
        return df[['age_code', 'sex_code', 'year', 'log_population']]
    
    except FileNotFoundError:
        st.error("找不到数据文件 Merged_Data.xlsx")
        st.stop()
    except Exception as e:
        st.error(f"数据加载失败: {str(e)}")
        st.stop()

# -------------------- 主界面 --------------------
st.set_page_config(
    page_title="SAH Prediction System", 
    layout="wide",
    page_icon="🧠"
)

st.title("Subarachnoid Hemorrhage Risk Prediction")
st.markdown("""
**An XGBoost-based prediction system with SHAP interpretation (1990-2050)**  
*Data Source: GBD Database | Developer: Walkerdii*
""")

# -------------------- 侧边栏输入 --------------------
with st.sidebar:
    st.header("⚙️ Prediction Parameters")
    
    # 年龄组选择
    age = st.selectbox(
        "Age Group", 
        AGE_GROUPS,
        index=3,
        help="Select age group between 15-49"
    )
    
    # 性别选择
    sex = st.radio(
        "Gender", 
        ['Female', 'Male'],
        index=0,
        horizontal=True
    )
    
    # 年份选择（带动态范围限制）
    current_year = datetime.now().year
    year = st.slider(
        "Year", 
        min_value=1990,
        max_value=2050,
        value=current_year,
        help=f"Valid range: 1990-2050 (Current year: {current_year})"
    )
    
    # 人口输入（带边界检查）
    population = st.number_input(
        "Population (Millions)",
        min_value=0.1,
        max_value=5000.0,
        value=10.0,
        step=0.1,
        format="%.1f",
        help="Actual population = Input × 1,000,000"
    )
    log_pop = np.log(population * 1_000_000)

# -------------------- 模型加载 --------------------
models = load_models()
df = load_dataset()

# -------------------- 预测执行 --------------------
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
    st.error(f"预测执行失败: {str(e)}")
    st.stop()

# -------------------- 结果展示 --------------------
col1, col2, col3 = st.columns(3)
metric_config = {
    'DALYs': ("Disability-Adjusted Life Years", "Overall disease burden measure"),
    'Incidence': ("Incidence Rate", "New cases per 100k population"),
    'Prevalence': ("Prevalence Rate", "Existing cases per 100k population")
}

for col, (key, (title, help_text)) in zip([col1, col2, col3], metric_config.items()):
    col.metric(
        title,
        f"{predictions[key]:,.2f}",
        help=help_text
    )

# -------------------- SHAP解释增强版 --------------------
st.divider()
st.header("🔍 Model Interpretation")

# 选择分析目标
analysis_target = st.selectbox(
    "Select Analysis Target",
    ['DALYs', 'Incidence', 'Prevalence'],
    index=0
)

try:
    explainer = shap.Explainer(models[analysis_target])
    shap_values = explainer(input_data)
    
    # 瀑布图
    st.subheader("Feature Impact Analysis")
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(shap_values[0], max_display=7, show=False)
    plt.title(f"{analysis_target} - SHAP Value Explanation", fontsize=14)
    st.pyplot(fig)
    
    # 特征依赖图
    st.subheader("Feature Relationship Exploration")
    selected_feature = st.selectbox(
        "Select Feature",
        input_data.columns,
        index=0
    )
    
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.dependence_plot(
        selected_feature,
        shap_values.values,
        input_data,
        interaction_index=None,
        ax=ax
    )
    plt.title(f"{selected_feature} Dependency", fontsize=12)
    st.pyplot(fig)

except Exception as e:
    st.error(f"模型解释失败: {str(e)}")

# -------------------- 数据验证 --------------------
with st.expander("📊 Data Validation"):
    st.write("### 数据样本", df.head(2))
    st.write("### 年龄分布", df['age_code'].value_counts().sort_index())
    st.write("### 性别分布", df['sex_code'].map({0: 'Female', 1: 'Male'}).value_counts())
