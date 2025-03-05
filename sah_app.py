# core.py
import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
from xgboost import XGBRegressor
from sklearnex import patch_sklearn
from datetime import datetime

# ==================== 初始化加速 ====================
patch_sklearn()

# ==================== 全局配置 ====================
MODEL_PATH = "sah_models/"
AGE_GROUPS = ['15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49']
os.makedirs(MODEL_PATH, exist_ok=True)

# ==================== 模型管理 ====================
@st.cache_resource
def load_models():
    """加载预训练模型"""
    try:
        return {
            'DALYs': XGBRegressor().load_model(f"{MODEL_PATH}DALYs.json"),
            'Incidence': XGBRegressor().load_model(f"{MODEL_PATH}Incidence.json"),
            'Prevalence': XGBRegressor().load_model(f"{MODEL_PATH}Prevalence.json")
        }
    except Exception as e:
        st.error(f"模型加载失败: {str(e)}")
        st.stop()

# ==================== 数据处理 ====================
@st.cache_data(ttl=3600, show_spinner=False)
def load_dataset():
    """加载并验证数据集"""
    try:
        df = pd.read_excel("Merged_Data.xlsx", sheet_name="Merged_Data", engine='openpyxl')
        
        # 数据清洗
        df['age_group'] = df['age_name'].str.replace(' years', '').str.strip()
        df['sex'] = df['sex_name'].str.lower().str.strip()
        
        # 数据验证
        if not df['age_group'].isin(AGE_GROUPS).all():
            invalid_age = df.loc[~df['age_group'].isin(AGE_GROUPS), 'age_group'].unique()
            st.error(f"无效年龄组: {invalid_age}")
            st.stop()
            
        if not df['sex'].isin(['female', 'male']).all():
            invalid_sex = df.loc[~df['sex'].isin(['female', 'male']), 'sex'].unique()
            st.error(f"无效性别: {invalid_sex}")
            st.stop()
            
        return df[['age_group', 'sex', 'year', 'log_population']]
    
    except Exception as e:
        st.error(f"数据加载失败: {str(e)}")
        st.stop()

# ==================== 界面配置 ====================
st.set_page_config(
    page_title="SAH Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 主界面 ====================
def main():
    # 标题部分
    st.title("SAH Predictor: Interactive Burden Forecasting with Explainable AI")
    st.markdown("""
    **An XGBoost-based prediction system for subarachnoid hemorrhage outcomes with SHAP interpretation (1990-2050)**  
    *Data Source: GBD Database | Developer: Walkerdii*
    """)
    
    # 侧边栏输入
    with st.sidebar:
        st.header("⚙️ 预测参数")
        
        # 年龄组选择
        age = st.selectbox(
            "年龄组",
            AGE_GROUPS,
            index=3,
            help="选择15-49岁之间的年龄分组"
        )
        
        # 性别选择
        sex = st.radio(
            "性别",
            ['Female', 'Male'],
            index=0,
            horizontal=True
        )
        
        # 年份选择
        current_year = datetime.now().year
        year = st.slider(
            "年份",
            min_value=1990,
            max_value=2050,
            value=current_year,
            step=1,
            help=f"有效范围: 1990-2050 (当前系统年份: {current_year})"
        )
        
        # 人口输入
        population = st.number_input(
            "人口数量 (百万)",
            min_value=0.1,
            max_value=1000.0,
            value=10.0,
            step=0.1,
            format="%.1f",
            help="实际人口 = 输入值 × 1,000,000"
        )
        log_pop = np.log(population * 1_000_000)
    
    # ==================== 模型加载 ====================
    with st.spinner('正在加载预测模型...'):
        models = load_models()
        df = load_dataset()
    
    # ==================== 生成预测 ====================
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
        st.error(f"预测失败: {str(e)}")
        st.stop()
    
    # ==================== 结果展示 ====================
    col1, col2, col3 = st.columns(3)
    col1.metric("伤残调整生命年 (DALYs)", 
              f"{predictions['DALYs']:,.1f}",
              help="疾病总负担的测量指标")
    col2.metric("发病率", 
              f"{predictions['Incidence']:.2f}%",
              delta_color="inverse",
              help="每10万人口新发病例数")
    col3.metric("患病率", 
              f"{predictions['Prevalence']:.2f}%",
              delta_color="inverse",
              help="每10万人口现存病例数")
    
    # ==================== 趋势可视化 ====================
    st.divider()
    st.header("📈 趋势预测")
    
    # 生成时间序列预测
    years_range = range(max(1990, year-10), min(2050, year+10)+1)
    plot_data = []
    
    for y in years_range:
        temp_data = input_data.copy()
        temp_data['year'] = y
        plot_data.append({
            'Year': y,
            'DALYs': models['DALYs'].predict(temp_data)[0],
            'Incidence': models['Incidence'].predict(temp_data)[0],
            'Prevalence': models['Prevalence'].predict(temp_data)[0]
        })
    
    df_plot = pd.DataFrame(plot_data)
    
    # 交互式图表
    selected_outcomes = st.multiselect(
        "选择展示指标",
        ['DALYs', 'Incidence', 'Prevalence'],
        default=['DALYs'],
        key="outcome_selector"
    )
    
    fig = px.line(
        df_plot, 
        x='Year', 
        y=selected_outcomes,
        title="10年趋势预测",
        markers=True,
        labels={'value': '指标值', 'variable': '指标'},
        height=400
    )
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    
    # ==================== SHAP解释 ====================
    st.divider()
    st.header("🧠 模型解释")
    
    # 选择分析目标
    analysis_target = st.selectbox(
        "选择分析指标",
        ['DALYs', 'Incidence', 'Prevalence'],
        index=0,
        key="shap_target"
    )
    
    try:
        explainer = shap.Explainer(models[analysis_target])
        shap_values = explainer(input_data)
        
        # 瀑布图
        st.subheader("特征影响分析")
        fig, ax = plt.subplots(figsize=(10,4))
        shap.plots.waterfall(shap_values[0], max_display=7, show=False)
        plt.title(f"{analysis_target} - SHAP值解释", fontsize=14)
        st.pyplot(fig)
        
        # 特征依赖图
        st.subheader("特征关系探索")
        selected_feature = st.selectbox(
            "选择特征",
            input_data.columns,
            index=0,
            key="feature_selector"
        )
        
        fig, ax = plt.subplots(figsize=(8,5))
        shap.dependence_plot(
            selected_feature,
            shap_values.values,
            input_data,
            interaction_index=None,
            ax=ax
        )
        plt.title(f"{selected_feature} 依赖关系", fontsize=12)
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"模型解释失败: {str(e)}")
    
    # ==================== 调试信息 ====================
    with st.expander("🔍 数据验证信息"):
        st.write("### 数据样本", df.head(2))
        st.write("### 年龄分布", df['age_group'].value_counts())
        st.write("### 性别分布", df['sex'].value_counts())

if __name__ == "__main__":
    main()
