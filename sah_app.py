import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from xgboost import XGBRegressor


# -------------------- 数据准备 --------------------
@st.cache_data
def load_data():
    data = pd.read_excel("Merged_Data.xlsx")  # 确保路径正确

    age_mapping = {'15-19': 0, '20-24': 1, '25-29': 2, '30-34': 3,
                   '35-39': 4, '40-44': 5, '45-49': 6}
    sex_mapping = {'female': 0, 'male': 1}

    data['age_encoded'] = data['age_name'].map(age_mapping)
    data['sex_encoded'] = data['sex_name'].map(sex_mapping)

    features = data[['age_encoded', 'sex_encoded', 'year', 'log_population']]
    models = {
        'DALYs': XGBRegressor().fit(features, data['DALYs']),
        'Incidence': XGBRegressor().fit(features, data['Incidence']),
        'Prevalence': XGBRegressor().fit(features, data['Prevalence'])
    }
    return models, age_mapping, sex_mapping


# -------------------- 网页布局 --------------------
st.set_page_config(page_title="SAH Prediction Model", layout="wide")
st.title("Subarachnoid Hemorrhage Risk Prediction")
st.markdown("""
**Instructions:**  
1. 在左侧输入参数 ➡️  
2. 查看实时预测结果和模型解释 ⬇️
""")

with st.sidebar:
    st.header("Input Parameters")
    age_group = st.selectbox(
        "Age Group",
        options=['15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49']
    )
    sex = st.radio("Sex", options=['female', 'male'])
    year = st.slider("Year", 2000, 2030, 2023)
    log_pop = st.number_input("Log Population", value=15.0)

# -------------------- 模型预测 --------------------
models, age_map, sex_map = load_data()
input_df = pd.DataFrame([[
    age_map[age_group], sex_map[sex], year, log_pop
]], columns=['age_encoded', 'sex_encoded', 'year', 'log_population'])

predictions = {
    'DALYs': models['DALYs'].predict(input_df)[0],
    'Incidence': models['Incidence'].predict(input_df)[0],
    'Prevalence': models['Prevalence'].predict(input_df)[0]
}

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("DALYs", f"{predictions['DALYs']:.2f}")
with col2:
    st.metric("Incidence", f"{predictions['Incidence']:.2f}%")
with col3:
    st.metric("Prevalence", f"{predictions['Prevalence']:.2f}%")

# -------------------- SHAP解释 --------------------
st.header("Model Interpretation")
try:
    explainer = shap.TreeExplainer(models['DALYs'])
    shap_values = explainer.shap_values(input_df)

    fig, ax = plt.subplots()
    shap.force_plot(
        explainer.expected_value,
        shap_values[0],
        input_df.iloc[0],
        feature_names=['Age', 'Sex', 'Year', 'Log Population'],
        matplotlib=True,
        show=False
    )
    st.pyplot(fig)
except Exception as e:
    st.error(f"SHAP可视化失败: {str(e)}")

# -------------------- 删除或生成特征分布图 --------------------
# st.header("Feature Distributions in Training Data")
# st.image("feature_distributions.png")