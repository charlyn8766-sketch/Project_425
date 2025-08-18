
import streamlit as st
import pandas as pd
from optimizer import build_and_solve_shift_model

st.set_page_config(page_title="员工排班优化系统", layout="wide")
st.title("👷 员工排班优化系统")

time_slots = {
    1: "10:00-11:00", 2: "11:00-12:00", 3: "12:00-13:00",
    4: "13:00-14:00", 5: "14:00-15:00", 6: "15:00-16:00",
    7: "16:00-17:00", 8: "17:00-18:00", 9: "18:00-19:00",
    10: "19:00-20:00", 11: "20:00-21:00", 12: "21:00-22:00",
    13: "22:00-23:00", 14: "23:00-00:00", 15: "00:00-01:00"
}

D = list(range(1,8))
T = list(range(1,16))
S = [(s,e) for s in T for e in T if 4 <= e - s <= 8]

st.sidebar.title("🔧 参数设置")
all_workers = ["Ana", "Vanessa_M", "Ines", "Yuliia", "Giulia", "Tomas", 
               "Ana_Bernabe", "Raul_Calero", "Jose_Antonio", "Haely"]
W = st.sidebar.multiselect("选择参与排班的员工", all_workers, default=all_workers[:5])

MinHw = {}
MaxHw = {}
for w in W:
    col1, col2 = st.sidebar.columns(2)
    with col1:
        MinHw[w] = st.number_input(f"{w} 最小小时", 15, 40, 25, key=f"{w}_min")
    with col2:
        MaxHw[w] = st.number_input(f"{w} 最大小时", 15, 45, 32, key=f"{w}_max")

Max_Deviation = st.sidebar.slider("最大允许人数偏差", 0.0, 5.0, value=2.5, step=0.1)
time_limit = st.sidebar.number_input("求解器最大运行时间（秒）", min_value=10, value=120)

uploaded_file = st.file_uploader("📤 上传销售需求 CSV（7行×15列，对应每小时需求）", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if df.shape != (7, 15):
        st.error("❌ CSV 格式应为 7 行 × 15 列！")
        st.stop()
    Demand = {d+1: df.iloc[d].tolist() for d in range(7)}
else:
    st.info("使用内置默认销售需求")
    Demand = {
        1:[0.0,0.89,1.08,1.15,2.51,3.11,2.16,4.06,1.64,1.45,1.31,2.68,2.73,2.14,0.86],
        2:[0.37,1.08,0.90,0.59,2.64,3.40,3.26,3.97,0.86,1.51,1.63,1.77,2.53,2.58,0.07],
        3:[0.12,0.80,1.67,2.64,2.43,2.64,2.87,2.25,2.61,1.62,1.60,0.88,1.90,2.25,0.72],
        4:[0.63,1.00,1.67,2.46,1.56,1.91,2.58,2.04,2.63,2.11,1.04,1.34,2.31,2.12,0.61],
        5:[0.31,0.74,1.39,1.88,2.77,1.75,4.15,3.55,1.85,2.22,1.57,1.34,3.27,3.07,0.76],
        6:[0.66,0.48,0.64,1.05,1.85,3.61,4.63,3.06,1.99,2.04,1.77,1.82,2.87,3.40,0.88],
        7:[0.26,0.52,1.46,2.39,1.43,3.18,3.79,3.23,2.91,1.41,2.06,2.28,2.18,2.03,0.86]
    }

if st.button("🚀 运行优化模型"):
    with st.spinner("正在求解，请稍等..."):
        result = build_and_solve_shift_model(W, D, T, S, MinHw, MaxHw, Demand, Max_Deviation, time_limit)

    st.success("✅ 求解完成！")
    st.write(f"🔧 状态：{result['status']}")
    st.write(f"🎯 目标值：{result['objective']:.2f}")
    st.write(f"⏱️ 求解时间：{result['elapsed_time']:.2f} 秒")

    df_result = pd.DataFrame(result["schedule"], columns=["员工", "星期", "时间段"])
    df_result["时间"] = df_result["时间段"].map(time_slots)
    df_result = df_result[["员工", "星期", "时间"]]
    st.dataframe(df_result)

    csv = df_result.to_csv(index=False).encode("utf-8-sig")
    st.download_button("📥 下载排班结果 CSV", csv, file_name="schedule_result.csv", mime="text/csv")
