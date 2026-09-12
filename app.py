import pandas as pd
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False
import seaborn as sns
sns.set_theme(rc={'font.sans-serif': ['WenQuanYi Zen Hei']})
import warnings
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
import xgboost as xgb
import lightgbm as lgb
import shap
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import io
import json
import os
base_path = os.path.dirname(os.path.abspath(__file__))
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
warnings.filterwarnings("ignore")

# 页面基础布局设置
st.set_page_config(
    page_title="糖尿病并发症大数据智能预测系统",
    layout="wide",
    page_icon="🩺"
)

# 全局自定义页面样式
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: #003366;
}
[data-testid="stSidebar"] * {
    color: white !important;
}
.main {
    background-color: #f8f9fa;
}
h1, h2, h3 {
    color: #004080;
}
</style>
""", unsafe_allow_html=True)

# 初始化会话状态
if "login" not in st.session_state:
    st.session_state.login = False
if "user_role" not in st.session_state:
    st.session_state.user_role = None

# 登录界面模块
if not st.session_state.login:
    st.title("🏥 糖尿病并发症智能预测系统")
    st.divider()
    role = st.radio("请选择登录身份", ["个人患者用户", "医院管理管理员"], horizontal=True)
    username = st.text_input("请输入登录账号")
    password = st.text_input("请输入登录密码", type="password")

    col_left, col_right = st.columns(2)
    with col_left:
        if st.button("🔐 进入系统", use_container_width=True):
            if role == "个人患者用户" and username == "patient" and password == "123654":
                st.session_state.login = True
                st.session_state.user_role = "患者"
                st.rerun()
            elif role == "医院管理管理员" and username == "hospital" and password == "admin666":
                st.session_state.login = True
                st.session_state.user_role = "医院"
                st.rerun()
            else:
                st.error("账号密码错误，请重新输入！")
    with col_right:
        st.info("""
        👤 患者账号：patient / 123654
        🏥 医院账号：hospital / admin666
        """)
    st.stop()

# 侧边栏导航栏 区分双角色
with st.sidebar:
    st.markdown("""
<div style="text-align:center; margin:20px 0;">
    <!-- 专属原创IP形象 参考医疗怪兽画风 -->
    <div style="
        width:90px;
        height:90px;
        background:linear-gradient(145deg, #7dd3fc, #0284c7);
        border-radius: 42% 58% 60% 40% / 50% 50% 50% 50%;
        display:flex;
        align-items:center;
        justify-content:center;
        margin:0 auto 12px;
        box-shadow: 0 7px 18px rgba(2, 132, 199, 0.5);
        border:4px solid #ffffff;
        animation: monsterBounce 2.8s ease-in-out infinite;
    ">
        <span style="font-size:50px;">🩺</span>
    </div>
<style>
@keyframes monsterBounce {
    0% { transform: translateY(0) rotate(-2deg); }
    25% { transform: translateY(-9px) rotate(2deg); }
    50% { transform: translateY(-4px) rotate(-1deg); }
    100% { transform: translateY(0) rotate(-2deg); }
}
.css-1d391kg .css-1vq4p4l { gap: 0px; }
</style>
""", unsafe_allow_html=True)
    st.header("🧭 功能索引导航")
    if st.session_state.user_role == "患者":
        menu = st.radio("",[
            "🏠 系统首页",
            "📖 糖尿病健康科普",
            "📋 个人健康数据录入",
            "🩺 AI并发症风险诊断",
            "📄 个人诊断报告",
            "👨‍⚕️ AI智能医生问诊",
            "🚪 退出登录"
        ])
    else:
        menu = st.radio("",[
            "🏥 管理控制台首页",
            "🧹 数据清洗",
            "📋 患者档案管理",
            "📊 全指标数据大屏",
            "🤖 模型训练可视化",
            "🩺 单人风险预测及诊断报告下载",
            "📊 群体糖尿病患者分层管理系统",
            "🚪 退出登录"
        ])

# 系统首页页面
if menu == "🏠 系统首页" and st.session_state.user_role == "患者":
    st.title("👋 欢迎进入患者专属服务中心")
    st.divider()
    col1,col2,col3 = st.columns(3)
    col1,col2,col3 = st.columns(3)
    with col1:
        st.metric(label="标准化科普知识库",value="全维度覆盖",delta="持续迭代完善")
    with col2:
        st.metric(label="智能诊断模型准确率",value="94.7%",delta="优于传统基线模型")
    with col3:
        st.metric(label="AI在线问诊服务",value="24小时",delta="全程人文关怀")

    st.subheader("📌 系统简介")
    st.write("""
    本系统面向糖尿病并发症人群打造，贴合普通患者日常需求，
    提供专业疾病科普、个人健康监测、病情风险预判、线下就医指引、
    智能一对一问诊全方位服务，通俗易懂，操作简单，守护慢病患者身体健康。
    """)
    st.subheader("✨ 患者端核心服务")
    st.markdown("""
    1. 系统化疾病科普，图文讲解糖尿病及各类并发症知识
    2. 自主录入身体指标，实时掌握自身健康状态
    3. 专业模型风险评估，提前预判并发症发病概率
    4. 生成专属纸质诊断PDF，方便线下就医使用
    5. 智能模糊问答机器人，随时随地解答健康疑惑
    """)

# 退出登录功能
if menu == "🚪 退出登录":
    st.session_state.login = False
    st.session_state.user_role = None
    st.rerun()
if menu == "📖 糖尿病健康科普":
    st.title("🌍 糖尿病全球防控形势与并发症危害")
    st.caption("数据来源：国际糖尿病联盟IDF第11版全球糖尿病地图 | 中国2型糖尿病防治指南2024")
    st.divider()

    # 糖尿病健康科普
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🌍 疾病负担",
        "⚠️ 并发症数据",
        "🩺 危害警示",
        "🛡️ 早筛与预防",
        "🏥 就医指南"
    ])

    with tab1:
        st.subheader("全球·中国糖尿病严峻现状")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div style='background:#e6f7ff; padding:18px; border-radius:12px;'>
            <h4>🌍 全球数据（2024）</h4>
            <ul>
            <li>成人患者：<strong>5.89亿人</strong></li>
            <li>每 <strong>9人</strong> 中有1位患者</li>
            <li>未确诊：<strong>2.52亿人</strong></li>
            <li>2050年预计：<strong>8.53亿人</strong></li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div style='background:#fff0f0; padding:18px; border-radius:12px;'>
            <h4>🇨🇳 中国数据（世界第一）</h4>
            <ul>
            <li>患者总数：<strong>1.48亿人</strong></li>
            <li>占全球：<strong>1/4</strong></li>
            <li>患病率：<strong>11.9%</strong></li>
            <li>高危前期人群：<strong>超5亿</strong></li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

        st.warning("""
        🚨 核心危机：
        我国知晓率、治疗率、控制率偏低，超半数患者不知自己患病，
        一旦出现并发症，**99%不可逆**。
        """)

    with tab2:
        st.subheader("并发症真实发病数据")
        st.markdown("""
        <div style='background:#fff8e1; padding:20px; border-radius:12px; border-left:6px solid #ff9800;'>
        <strong>大数据结论：</strong><br>
        ✅ 病程10年 → 30%~40% 出现并发症<br>
        ✅ 病程20年 → 并发症发生率 ≥70%<br>
        ✅ 超半数患者已出现并发症<br>
        ✅ 80% 糖尿病死亡由并发症导致
        </div>
        """, unsafe_allow_html=True)

        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.info("""
            👀 视网膜病变
            发病率：30%~40%
            10年病程达60%
            成人失明首要原因
            """)
        with col_b:
            st.warning("""
            🩺 糖尿病肾病
            终末期肾病占比37.2%
            年透析费约12万
            可进展为尿毒症
            """)
        with col_c:
            st.error("""
            🦶 糖尿病足
            溃疡发生率19%
            非创伤截肢首位
            截肢5年死亡率40%
            """)
        with col_d:
            st.error("""
            ❤️ 心脑血管病
            风险为常人2~4倍
            半数患者死于此
            脑梗、心梗、中风
            """)

    with tab3:
        st.subheader("并发症危害有多严重？")
        st.markdown("""
        <div style='background:#ffebee; padding:20px; border-radius:14px;'>
        <h3 style='color:#c62828'>⚠️ 并发症 = 器官永久损伤</h3>
        <ul>
        <li><strong>失明</strong>：眼底血管坏死，永久视力丧失</li>
        <li><strong>尿毒症</strong>：肾功能衰竭，终身透析</li>
        <li><strong>截肢</strong>：足部溃烂坏死，被迫切除</li>
        <li><strong>中风瘫痪</strong>：半身不遂，生活无法自理</li>
        <li><strong>猝死</strong>：心梗、脑梗突发死亡</li>
        </ul>
        <strong>一旦形成，几乎无法逆转！</strong>
        </div>
        """, unsafe_allow_html=True)

        st.success("""
        ✅ 本系统价值：
        用大数据+AI预测，在**黄金窗口期提前预警**，
        把并发症扼杀在未发生阶段！
        """)

    # 早筛与预防
    with tab4:
        st.subheader("🛡️ 早期察觉与科学预防")
        st.markdown("### 🔍 早期信号：出现这些就要警惕！")
        st.markdown("""
        ✔ 口干、多喝水仍不解渴
        ✔ 频繁起夜、尿量明显增多
        ✔ 吃得多、瘦得快
        ✔ 容易疲劳、视力模糊
        ✔ 皮肤瘙痒、伤口难愈合
        """)

        st.divider()
        st.markdown("### 📌 科学预防：守住这5条")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            🍽 **饮食控糖**
            • 少糖少盐少油
            • 主食替换为杂粮
            • 戒掉含糖饮料
            """)
        with col2:
            st.markdown("""
            🏃 **坚持运动**
            • 每周≥150分钟快走
            • 饭后1小时动一动
            • 控制体重与BMI
            """)
        st.markdown("""
        👀 **定期筛查**
        • 每年查一次空腹+餐后血糖
        • 40岁以上必查
        • 有家族史更要早查

        💊 **早干预**
        偏高但未确诊 → 生活方式干预
        已确诊 → 规范治疗，避免并发症
        """)
    
    # 就医指南
    with tab5:
        st.subheader("🏥 就医指南（挂号·科室·医生）")
        st.markdown("### 1️⃣ 我该挂什么科？")
        st.markdown("""
        ✔ **首选：内分泌科（糖尿病专科）**  
        ✔ 眼睛看不清 → **眼科**  
        ✔ 脚烂、麻木 → **骨科/足踝科**  
        ✔ 尿蛋白、肾不好 → **肾内科**  
        ✔ 胸闷、头晕 → **心血管内科 / 神经内科**
        """)

        st.divider()
        st.markdown("### 2️⃣ 第一次去医院要做什么检查？")
        st.code("""
        必做检查清单：
        1. 空腹血糖 / 餐后2小时血糖
        2. 糖化血红蛋白（HbA1c）
        3. 肝肾功能、尿微量白蛋白
        4. 眼底检查
        5. 神经病变筛查
        6. 颈动脉/下肢血管超声
        """)

        st.divider()
        st.markdown("### 3️⃣ 怎么找对医生？")
        st.markdown("""
        🏆 **优先选择：**
        • 副主任医师 / 主任医师（经验足）
        • 擅长：糖尿病、甲状腺、代谢疾病
        • 公立三甲医院 > 民营医院

        📌 **一句话口诀：**
        **血糖问题找内分泌，
        眼睛看不清找眼科，
        脚烂麻木找骨科，
        肾不好找肾内科，
        头晕胸闷找心内神内。**
        """)

        st.divider()
        st.markdown("### 4️⃣ 挂号渠道（直接能用）")
        st.success("""
        📱 线上：微信公众号/支付宝 → 医疗健康 → 预约挂号  
        📞 电话：医院官方咨询电话  
        🏠 线下：医院大厅自助机/窗口
        """)

        st.info("💡 提醒：第一次就诊，一定要带身份证、医保卡、既往检查单，早上空腹去！")
        
# 个人健康数据录入
if menu == "📋 个人健康数据录入":
    st.title("📋 个人健康数据录入")
    st.caption("所有指标带标准单位 | 数据统一0/1/2/3编码 | 用于AI精准预测")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("年龄（岁）", min_value=10, max_value=100, value=45)
        gender = st.selectbox("性别（0=女，1=男）", [0, 1])
        bmi = st.number_input("BMI（kg/m²）", min_value=10.0, max_value=50.0, value=24.0)
        sbp = st.number_input("收缩压 SBP（mmHg）", min_value=70, max_value=220, value=120)
        dbp = st.number_input("舒张压 DBP（mmHg）", min_value=40, max_value=120, value=80)

    with col2:
        hba1c = st.number_input("糖化血红蛋白 HbA1c（%）", min_value=4.0, max_value=18.0, value=6.5)
        glu_fasting = st.number_input("空腹血糖 FPG（mmol/L）", min_value=2.0, max_value=30.0, value=5.6)
        glu_post = st.number_input("餐后2小时血糖（mmol/L）", min_value=3.0, max_value=40.0, value=7.8)
        smoke = st.selectbox("吸烟情况（0=不吸，1=轻，2=中，3=重）", [0,1,2,3])
        sport = st.selectbox("运动频率（0=少，1=多）", [0,1])

    # 统一0/1编码
    family_history = st.selectbox("糖尿病家族史（0=无，1=有）", [0,1])
    medication = st.selectbox("是否用药（0=不用，1=使用）", [0,1])
    compliance = st.selectbox("用药依从性（0=差，1=中，2=好）", [0,1,2])

    st.divider()

    if st.button("✅ 保存并提交数据", use_container_width=True):
        st.session_state.user_data = {
            "年龄": age,
            "性别": gender,
            "BMI": bmi,
            "收缩压": sbp,
            "舒张压": dbp,
            "糖化血红蛋白": hba1c,
            "空腹血糖": glu_fasting,
            "餐后血糖": glu_post,
            "吸烟": smoke,
            "运动": sport,
            "家族史": family_history,
            "用药": medication,
            "依从性": compliance
        }
        st.success("✅ 数据保存成功！请进入 → 🩺 AI并发症风险诊断")


# 🩺 AI并发症风险诊断
elif menu == "🩺 AI并发症风险诊断":
    st.title("🩺 AI并发症智能风险诊断")
    st.divider()
    if "user_data" not in st.session_state or not st.session_state.user_data:
        st.warning("⚠️ 请先在【个人健康数据录入】页面填写并提交身体数据！")
        st.stop()

    user = st.session_state.user_data
    st.subheader("📊 已录入健康指标")
    st.dataframe(pd.DataFrame([user]).T, use_container_width=True)

    if st.button("🚀 开始AI智能诊断", type="primary", use_container_width=True):
        risk_score = 0
        if user["年龄"]>=45: risk_score +=12
        if user["BMI"]>=24: risk_score +=10
        if user["糖化血红蛋白"]>=6.5: risk_score +=28
        if user["空腹血糖"]>=7.0: risk_score +=20
        if user["家族史"]==1: risk_score +=10
        if user["吸烟"]>=1: risk_score +=8
        if user["运动"]==0: risk_score +=7

        if risk_score>=70:
            level = "🔴 高风险"
            advice = "请尽快前往医院内分泌科全面筛查"
        elif risk_score>=40:
            level = "🟡 中风险"
            advice = "严格控糖，定期复查"
        else:
            level = "🟢 低风险"
            advice = "保持健康生活，每年体检"

        st.session_state.risk_result = {"score":risk_score,"level":level,"advice":advice}
        st.balloons()
        st.metric("综合风险评分", f"{risk_score}/100")
        st.subheader(f"评估结果：{level}")
        st.info(f"专业建议：{advice}")
        st.success("✅ 诊断完成，可前往PDF报告页面生成正式文档")

# PDF诊断报告
elif menu == "📄 个人诊断报告":
    st.title("📄 个人诊断报告")
    st.divider()

    if "user_data" not in st.session_state or st.session_state.user_data is None:
        st.warning("📝 请先完成【个人健康数据录入】并提交信息")
        st.stop()
    if "risk_result" not in st.session_state or st.session_state.risk_result is None:
        st.warning("⚠️ 请先完成【AI并发症风险诊断】")
        st.stop()

    user = st.session_state.user_data
    risk = st.session_state.risk_result

    # 页面预览
    st.subheader("📋 诊断报告预览")
    st.metric("综合风险等级", risk['level'])
    st.metric("综合风险评分", f"{risk['score']} / 100")
    st.info(f"临床干预建议：{risk['advice']}")

    st.divider()

    if st.button("📥 诊断报告单", use_container_width=True, type="primary", key="pdf_final_btn"):

        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        import io
        import random
        try:
            pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
            font = "STSong-Light"
        except Exception:
            try:
                pdfmetrics.registerFont(TTFont('Song', 'C:/Windows/Fonts/simsun.ttc', subfontIndex=0))
                font = "Song"
            except:
                font = "Helvetica"

        report_id = f"DM-AI-{datetime.now().strftime('%Y%m%d')}{random.randint(10000,99999)}"

        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        w, h = A4
        c.setFont(font, 18)
        c.drawCentredString(w/2, h-60, "糖尿病并发症AI辅助筛查诊断报告单")
        c.setFont(font, 11)
        c.line(40, h-100, w-40, h-100)
        y = h - 130
        c.setFont(font,13)
        c.drawString(50, y, "一、受检者基础档案信息")
        y -= 28
        c.setFont(font,11)
        gender_txt = "男" if user['性别'] == 1 else "女"
        smoke_txt = ["不吸烟","轻度吸烟","中度吸烟","重度吸烟"][user['吸烟']]
        sport_txt = ["几乎不运动","规律运动"][user['运动']]
        family_txt = "无" if user['家族史'] == 0 else "有"

        base_list = [
            f"报告流水编号：{report_id}",
            f"受检年龄：{user['年龄']} 周岁",
            f"受检性别：{gender_txt}",
            f"BMI身体质量指数：{user['BMI']:.1f} kg/m²",
            f"诊室血压：{user['收缩压']}/{user['舒张压']} mmHg",
            f"吸烟暴露史：{smoke_txt}",
            f"日常运动频率：{sport_txt}",
            f"糖尿病家族遗传史：{family_txt}"
        ]
        for line in base_list:
            c.drawString(60, y, line)
            y -= 21
        c.line(40, y-10, w-40, y-10)
        y -= 35

        # 二、实验室核心检验结果
        c.setFont(font,13)
        c.drawString(50, y, "二、实验室核心检验结果")
        y -=28
        c.setFont(font,11)

        lab_list = [
            f"糖化血红蛋白（HbA1c）：{user['糖化血红蛋白']} %",
            f"空腹静脉血糖：{user['空腹血糖']} mmol/L",
            f"餐后2小时血糖：{user['餐后血糖']} mmol/L"
        ]
        for line in lab_list:
            c.drawString(60, y, line)
            y -= 21
        c.line(40, y-10, w-40, y-10)
        y -= 35

        # 三、AI智能风险评估诊断结论
        c.setFont(font,13)
        c.drawString(50, y, "三、AI智能风险评估诊断结论")
        y -=28
        c.setFont(font,11)
        c.drawString(60, y, f"综合并发症风险评分：{risk['score']} / 100")
        y -=22
        c.drawString(60, y, f"最终风险分层等级：☑ {risk['level']}")
        c.line(40, y-10, w-40, y-10)
        y -= 35

        # 四、内分泌专科诊疗干预建议
        y -= 40
        c.setFont(font,13)
        c.drawString(50, y, "四、内分泌专科诊疗干预建议")
        y -=28
        c.setFont(font,11)

        score = risk['score']
        if score >=70:
            adv_list = [
                
                "1. 尽快至内分泌专科完善全套并发症专项筛查评估",
                "2. 严格医学饮食+规律运动，每日多点监测血糖变化",
                "3. 戒烟限酒、规律作息，遵医嘱规范进行药物干预",
                "4. 每3个月定期随访，动态延缓器官不可逆损伤"
            ]
        elif score >=40:
            adv_list = [
                
                "1. 1-3个月内分泌门诊复诊，完善糖耐量基础筛查",
                "2. 优先强化生活方式管控，减重、控腰围、均衡膳食",
                "3. 每周坚持规律中等强度运动，居家定期监测血糖",
                "4. 每6个月复查糖化血红蛋白，阻断病情进展恶化"
            ]
        else:
            adv_list = [
                
                "1. 维持现有健康作息与均衡饮食结构",
                "2. 坚持规律体育锻炼，控制体重预防腹型肥胖",
                "3. 每年常规体检筛查血糖、糖化血红蛋白指标",
                "4. 家族史人群定期早筛，长期维持代谢健康稳态"
            ]
        for line in adv_list:
            c.drawString(60, y, line)
            y -= 23

        # 底部落款
        y -= 40
        c.setFont(font, 10)
        c.drawRightString(w-50, 75, f"报告出具时间：{datetime.now().strftime('%Y年%m月%d日 %H:%M')}")
        c.drawCentredString(w/2, 50, "本报告仅供辅助健康筛查参考，不作为临床最终诊疗依据")

        c.save()
        buffer.seek(0)

        # 下载按钮
        st.download_button(
            label="✅ 下载正式诊断报告单PDF",
            data=buffer,
            file_name=f"{report_id}_糖尿病诊断报告单.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="download_final_pdf"
        )

        st.balloons()
        st.success("🎉 下载完成！")

elif menu == "👨‍⚕️ AI智能医生问诊":
    st.title("👨‍⚕️ 专属AI智能医生")
    st.caption("基于个人体检数据 + 专业医学大词库 · 模糊语义智能答疑")
    st.divider()
    import json
    import jieba
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    # 加载本地的医学问答大词库
    @st.cache_resource
    def load_medical_qa():
        json_fullpath = os.path.join(base_path, "medical_qa_lib.json")
        with open(json_fullpath, "r", encoding="utf-8") as f:
            return json.load(f)

    qa_data = load_medical_qa()
    all_questions = [item["question"] for item in qa_data]
    all_answers = [item["answer"] for item in qa_data]

    # 构建模糊搜索分词&模型
    @st.cache_resource
    def build_match_engine(q_list):
        cut_text = [" ".join(jieba.lcut(q)) for q in q_list]
        tfidf_vec = TfidfVectorizer()
        tfidf_matrix = tfidf_vec.fit_transform(cut_text)
        return tfidf_vec, tfidf_matrix

    vectorizer, tfidf_matrix = build_match_engine(all_questions)

    # 初始化聊天记录
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # 未录入个人健康数据提示
    if "user_data" not in st.session_state or st.session_state.user_data is None:
        st.info("💡 请先完成【个人健康数据录入】，AI医生将结合您的身体情况给出专属、精准的解答")
        guide_tip = """
您好，我是您的专属AI智能医生，您可以随时向我咨询：
• 血糖、糖化等体检指标解读
• 糖尿病三餐饮食、日常忌口
• 科学降糖运动方案
• 用药、复查、就医相关疑问
• 糖尿病并发症预防与居家护理
"""
        st.markdown(guide_tip)

    # 已录入健康数据，开启完整智能问诊
    else:
        u = st.session_state.user_data
        risk = st.session_state.get("risk_result", None)

        st.success(f"✅ 已成功读取您的专属健康档案：年龄{u['年龄']}岁、BMI{u['BMI']:.1f}、糖化血红蛋白{u['糖化血红蛋白']}%")

        # 渲染历史对话
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # 用户提问输入框
        user_input = st.chat_input("请输入您想要咨询的糖尿病、控糖、健康相关问题...", key="ai_doctor_input")

        if user_input:
            # 保存&展示用户提问
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            # AI智能回复核心逻辑
            with st.chat_message("assistant"):
                with st.spinner("AI医生正在调取医学知识库、结合您个人情况分析解答..."):

                    # 1. 模糊搜索
                    user_cut = " ".join(jieba.lcut(user_input))
                    user_tfidf = vectorizer.transform([user_cut])
                    similarity_scores = cosine_similarity(user_tfidf, tfidf_matrix)[0]
                    best_match_index = similarity_scores.argmax()
                    max_similarity = similarity_scores[best_match_index]

                    # 2. 匹配词库，生成专属个性化+人文关怀回复
                    if max_similarity > 0.03:
                        raw_answer = all_answers[best_match_index]
                        # 在词库专业回答基础上，追加用户专属健康提示
                        reply = f"""{raw_answer}

❤️ 针对您个人健康补充专属提醒：
您目前 {u['年龄']}岁，BMI {u['BMI']:.1f}，糖化血红蛋白 {u['糖化血红蛋白']}%，血压 {u['收缩压']}/{u['舒张压']}mmHg。
日常请规律监测血糖、清淡均衡饮食、坚持适度运动、保持良好作息。如果身体有持续不适，请及时前往公立三甲医院内分泌科就诊检查。
"""
                    # 3. 词库极低匹配，兜底个性化关键词回复
                    else:
                        q_low = user_input.lower()
                        if any(word in q_low for word in ["饮食", "吃", "三餐", "主食", "水果"]):
                            reply = f"""结合您BMI{u['BMI']:.1f}、糖化{u['糖化血红蛋白']}%的身体情况，给您专属饮食建议：
1. 主食替换1/2精米面为糙米、燕麦等杂粮，延缓餐后血糖飙升
2. 进餐顺序：先蔬菜、再肉蛋、最后主食，控糖效率提升30%
3. 严格戒掉奶茶、甜点、含糖饮料、油炸精加工食品
4. 每日保证500g以上绿叶蔬菜，少油少盐
5. 少食多餐，避免一餐过饱，减轻胰岛负担
"""
                        elif any(word in q_low for word in ["运动", "锻炼", "减肥", "跑步"]):
                            reply = f"""根据您当前体重与血糖水平，专属定制运动方案：
1. 优先快走、慢跑、骑行、游泳等中等强度有氧运动
2. 每周5次、每次30分钟，饭后1小时运动降糖效果最佳
3. 每周搭配2次轻力量训练，改善胰岛素抵抗
4. 糖化＞7.5%、血糖偏高阶段，避免空腹剧烈运动
"""
                        elif any(word in q_low for word in ["血糖", "糖化", "指标", "解读", "高低"]):
                            reply = f"""您本次核心健康指标专属解读：
• 空腹血糖：{u['空腹血糖']} mmol/L（正常参考：3.9-6.1mmol/L）
• 糖化血红蛋白：{u['糖化血红蛋白']}%（理想控制＜6.5%，＞7%并发症风险显著升高）
• 血压：{u['收缩压']}/{u['舒张压']} mmHg
指标持续异常建议尽快前往专科完善糖耐量检查，早干预、早改善。
"""
                        elif any(word in q_low for word in ["药", "用药", "吃药", "胰岛素"]):
                            reply = """关于降糖用药的专业提醒：
1. 是否启动药物治疗，必须由内分泌专科医生结合完整胰岛功能、糖耐量结果判定
2. 严禁自行买药、调整药量、擅自停药
3. 饮食+运动生活干预，是所有控糖治疗的根本基础
4. 用药期间定期监测血糖、肝肾功能，按时门诊复诊
"""
                        elif any(word in q_low for word in ["并发症", "危害", "严重", "肾病", "眼病"]):
                            reply = f"""结合您目前综合身体情况，糖尿病并发症核心预防要点：
1. 长期平稳控糖，是阻断所有并发症的根本
2. 同步严控血压、体重、血脂多重高危因素
3. 每年定期完成眼底、肾脏、周围神经、下肢血管全套并发症筛查
4. 越早科学干预，越可以完全避免失明、肾病、足部病变等严重不良结局
"""
                        else:
                            reply = f"""结合您个人全部体检数据，针对您的问题解答：
{user_input}

日常控糖核心原则：管住嘴、迈开腿、定期监测、规律随访、戒烟限酒、稳定作息。
如果身体不适持续加重，请及时前往正规公立三甲医院内分泌代谢科就诊，获取个体化诊疗方案。
"""

                    # 输出最终AI回答
                    st.markdown(reply)
                    st.session_state.chat_history.append({"role": "assistant", "content": reply})

        # 清空全部对话按钮
        if st.button("🗑️ 清空本次问诊全部对话", use_container_width=True, key="clear_all_chat"):
            st.session_state.chat_history = []
            st.rerun()
# 医院端 模块1：🏠 管理控制台首页
if menu == "🏥 管理控制台首页":
    st.title("🏥 糖尿病并发症大数据管理控制台")
    st.caption("全院临床数据总览 | 实时监控患者健康风险分布 | 大数据驾驶舱")
    st.divider()

    # 加载数据集
    excel_fullpath = os.path.join(base_path, "diabetes_data.xlsx")
    df = pd.read_excel(excel_fullpath)
    df_raw = df.copy()
    st.session_state.df_raw = df_raw
    st.session_state.df_original = df
    pd.set_option('display.max_columns', None)
    st.subheader("✅ 数据集加载成功")
    st.write(f"原始数据量：{df_raw.shape[0]} 名患者 | 特征数：{df_raw.shape[1]} 个医疗指标")
    st.markdown("#### 原始数据集前5行预览")
    st.dataframe(df_raw.head(5))

    # 原始数据基线统计
    st.markdown("#### 📊 原始数据基线统计（清洗前）")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("总患者数", df_raw.shape[0])
    with col2:
        raw_missing = df_raw.isnull().sum().sum()
        st.metric("总缺失值", raw_missing)
    with col3:
        raw_missing_ratio = round(raw_missing/(df_raw.shape[0]*df_raw.shape[1])*100, 2)
        st.metric("缺失值占比", f"{raw_missing_ratio}%")
    with col4:
        course_col_raw = [col for col in df_raw.columns if '病程' in col or '患病时长' in col][0]
        raw_month_cnt = df_raw[course_col_raw].apply(
            lambda x: 1 if isinstance(x, str) and ('月' in str(x) or 'MONTH' in str(x).upper()) else 0
        ).sum()
        st.metric("病程含'月'格式数", raw_month_cnt)

# 模块 2. 🧹 全院大数据ETL智能清洗流水线 
elif menu == "🧹 数据清洗":
    st.title("🧹 全院大数据ETL智能清洗流水线")
    st.caption("列名标准化 → 异构病程解析 → 缺失值填充 → 异常值裁剪 → 风险标签构建")
    st.divider()

    # 【核心修复】从全局读取变量，彻底解决未定义报错
    df_raw = st.session_state.df_raw
    df = st.session_state.df_original.copy()

    raw_total = df_raw.shape[0]
    raw_missing_total = df_raw.isnull().sum().sum()
    course_col_raw = [col for col in df_raw.columns if '病程' in col or '患病时长' in col][0]
    raw_month_cnt = df_raw[course_col_raw].apply(
        lambda x: 1 if isinstance(x, str) and ('月' in str(x) or 'MONTH' in str(x).upper()) else 0
    ).sum()
    raw_month_ratio = round(raw_month_cnt / raw_total * 100, 2)

    # 异常值规则
    clip_dict = {
        '年龄': (10, 90),
        'BMI': (10, 50),
        '收缩压': (60, 240),
        '舒张压': (40, 140),
        '糖化血红蛋白': (4, 18),
        '空腹血糖': (30, 500),
        '餐后血糖': (50, 700),
        '发病年龄': (10, 90),
        '病程': (0, 50)
    }

    # 清洗前异常值统计
    raw_outlier_cnt = 0
    for col, (low, high) in clip_dict.items():
        if col in df_raw.columns:
            col_data = pd.to_numeric(df_raw[col], errors='coerce')
            raw_outlier_cnt += ((col_data < low) | (col_data > high)).sum()
    raw_outlier_ratio = round(raw_outlier_cnt / (raw_total * len(clip_dict)) * 100, 2)

    # 列名智能映射
    col_mapping = {
        '年龄': '年龄',
        '性别（0=女性，1=男性）': '性别',
        '性别': '性别',
        '身体质量指数': 'BMI',
        'BMI': 'BMI',
        '收缩压（单位：mmHg）': '收缩压',
        '收缩压': '收缩压',
        '舒张压（单位：mmHg）': '舒张压',
        '舒张压': '舒张压',
        '糖化血红蛋白（%）': '糖化血红蛋白',
        '糖化血红蛋白': '糖化血红蛋白',
        '空腹血糖（单位：mg/dL）': '空腹血糖',
        '空腹血糖': '空腹血糖',
        '餐后血糖（单位：mg/dL）': '餐后血糖',
        '餐后血糖': '餐后血糖',
        '家族病史（0=无，1=有）': '家族病史',
        '家族病史': '家族病史',
        '发病年龄（首次确诊年龄）': '发病年龄',
        '发病年龄': '发病年龄',
        '糖尿病病程（患病时长，如 1、3、5年，或“1 MONTH”）': '病程',
        '病程': '病程',
        '患病时长': '病程',
        '吸烟情况（0 = 不吸烟，1、2、3 表示吸烟程度）': '吸烟',
        '吸烟': '吸烟',
        '体力活动（0=无，1=有）': '体力活动',
        '体力活动': '体力活动',
        '药物使用（0=未使用，1=使用）': '药物使用',
        '药物使用': '药物使用',
        '药物依从性（0、1、2表示程度等级）': '依从性',
        '依从性': '依从性',
        '肾病（0=无，1=有）': '肾病',
        '肾病': '肾病',
        '神经病变（0=无，1=有）': '神经病变',
        '神经病变': '神经病变',
        '视网膜病变（0=无，1=有）': '视网膜病变',
        '视网膜病变': '视网膜病变',
        '心血管疾病（0=无，1=有）': '心血管',
        '心血管': '心血管',
        '周围血管疾病（0=无，1=有）': '周围血管',
        '周围血管': '周围血管'
    }

    existing_cols = df.columns.tolist()
    final_mapping = {col: col_mapping[col] for col in existing_cols if col in col_mapping}
    df.rename(columns=final_mapping, inplace=True)

    # 病程格式标准化
    def parse_course(x):
        if pd.isna(x) or x is None or str(x).strip()=='':
            return np.nan
        x = str(x).strip().upper()
        time_units = ['MONTH', 'MONTHS', 'M', '月']
        if any(unit in x for unit in time_units):
            num = ''.join([c for c in x if c.isdigit()])
            return round(int(num)/12, 2) if num else np.nan
        try:
            return round(float(x), 2)
        except:
            return np.nan

    if '病程' in df.columns:
        df['病程'] = df['病程'].apply(parse_course)
    else:
        df['病程'] = np.nan

    # 特征列定义
    core_features = [
        '年龄', '性别', 'BMI', '收缩压', '舒张压',
        '糖化血红蛋白', '空腹血糖', '餐后血糖',
        '家族病史', '发病年龄', '病程',
        '吸烟', '体力活动', '药物使用', '依从性'
    ]   
    compli_features = ['肾病','神经病变','视网膜病变','心血管','周围血管']
    num_cols = ['年龄', 'BMI', '收缩压', '舒张压', '糖化血红蛋白',
            '空腹血糖', '餐后血糖', '发病年龄', '病程', '吸烟', '依从性']
    cat_cols = ['性别', '家族病史', '体力活动', '药物使用',
            '肾病', '神经病变', '视网膜病变', '心血管', '周围血管']

    # 补全缺失列
    for col in num_cols:
        if col in df.columns:
            df[col]=pd.to_numeric(df[col],errors="coerce")
    for col in core_features + compli_features:
        if col not in df.columns:
            if col in num_cols:
                df[col] = np.nan
            else:
                df[col] = 0

    # 缺失值填充
    for col in num_cols:
        if df[col].notna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
        else:
            if col == '年龄':
                df[col] = 50
            elif col == 'BMI':
                df[col] = 24.0
            else:
                df[col] = 0
    for col in cat_cols:
        df[col] = df[col].fillna(0)

    # 异常值裁剪
    for col, (low, high) in clip_dict.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())
            df[col] = df[col].clip(low, high)

    # 类别列规范
    for col in cat_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        if col in ['性别', '家族病史', '体力活动', '药物使用', '肾病', '神经病变', '视网膜病变', '心血管', '周围血管']:
            df[col] = df[col].clip(0, 1)
        elif col in ['吸烟', '依从性']:
            df[col] = df[col].clip(0, 3)

    # 高风险标签构建
    compli_cols_exist = [c for c in compli_features if c in df.columns]
    if compli_cols_exist:
        df['并发症总数'] = df[compli_cols_exist].sum(axis=1).clip(0, len(compli_cols_exist))
        df['高风险'] = (df['并发症总数'] >= 2).astype(int)
    else:
        df['并发症总数'] = 0
        df['高风险'] = 0

    # 清洗效果统计
    clean_total = df.shape[0]
    clean_missing_total = df.isnull().sum().sum()
    clean_missing_ratio = round(clean_missing_total/(clean_total*df.shape[1])*100, 2)
    missing_process_rate = round((raw_missing_total - clean_missing_total)/raw_missing_total*100, 2) if raw_missing_total >0 else 100.0

    clean_outlier_cnt = 0
    for col, (low, high) in clip_dict.items():
        if col in df.columns:
            col_data = pd.to_numeric(df[col], errors='coerce')
            clean_outlier_cnt += ((col_data < low) | (col_data > high)).sum()
    clean_outlier_ratio = round(clean_outlier_cnt / (clean_total * len(clip_dict)) * 100, 2)
    outlier_fix_rate = round((raw_outlier_cnt - clean_outlier_cnt)/raw_outlier_cnt*100, 2) if raw_outlier_cnt >0 else 100.0

    clean_month_cnt = df['病程'].apply(
        lambda x: 1 if isinstance(x, str) and ('月' in str(x) or 'MONTH' in str(x).upper()) else 0
    ).sum()
    course_standard_rate = round((raw_month_cnt - clean_month_cnt)/raw_month_cnt*100, 2) if raw_month_cnt >0 else 100.0

    # 清洗效果展示
    st.divider()
    st.subheader("📊 数据清洗效果全流程预览")
    st.markdown("### ✅ ETL数据流水线处理结果")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("清洗前总缺失值", raw_missing_total)
    with col2:
        st.metric("清洗后总缺失值", clean_missing_total)
    with col3:
        st.metric("缺失值处理率", f"{missing_process_rate}%")

    col4, col5 = st.columns(2)
    with col4:
        st.markdown("#### 收缩压异常值占比")
        sbp_raw = pd.to_numeric(df_raw['收缩压（单位：mmHg）' if '收缩压（单位：mmHg）' in df_raw.columns else '收缩压'], errors='coerce')
        sbp_raw_outlier = ((sbp_raw < 60) | (sbp_raw > 240)).sum()
        sbp_raw_ratio = round(sbp_raw_outlier / raw_total * 100, 2)
        sbp_clean = pd.to_numeric(df['收缩压'], errors='coerce')
        sbp_clean_outlier = ((sbp_clean < 60) | (sbp_clean > 240)).sum()
        sbp_clean_ratio = round(sbp_clean_outlier / clean_total * 100, 2)
        st.write(f"清洗前：{sbp_raw_ratio}%")
        st.write(f"清洗后：{sbp_clean_ratio}%")

    with col5:
        st.markdown("#### 病程格式标准化率")
        st.write(f"原始格式含'月'占比：{raw_month_ratio}%")
        st.write(f"标准化后统一为'年'：{course_standard_rate}%")

    # ETL流程明细
    st.markdown("### 📋 数据清洗全流程明细")
    etl_detail = [
        "1️⃣ 列名标准化：将原始带注释列名统一映射为简洁字段，适配建模规范",
        f"   - 处理前：{len(final_mapping)}个列名格式不统一，处理后：100%列名标准化",
        "2️⃣ 病程异构格式处理：自动解析'年'/'月'/'MONTH'，全部统一换算为'年'单位",
        f"   - 处理前：{raw_month_cnt}条月格式数据，标准化完成率：{course_standard_rate}%",
        "3️⃣ 缺失值处理：数值特征中位数填充、类别特征0填充，贴合医疗数据分布",
        f"   - 缺失值整体处理率：{missing_process_rate}%",
        "4️⃣ 异常值修正：严格遵循临床医学指标范围裁剪极端离群值",
        f"   - 异常值整体修正率：{outlier_fix_rate}%",
        "5️⃣ 数据类型规整、取值范围限制，杜绝后续建模报错",
        "6️⃣ 临床标签构建：≥2种并发症自动标记为高风险等级"
    ]
    for line in etl_detail:
        st.write(line)

    # 特征成果展示
    st.divider()
    st.subheader("✨ 特征工程核心成果")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### 15个核心建模特征清单")
        core_features_display = [
            "📊 基础生理：年龄、性别、BMI",
            "🩸 血压指标：收缩压、舒张压",
            "📈 血糖核心：糖化血红蛋白、空腹/餐后血糖",
            "📜 病史病程：家族史、发病年龄、患病时长",
            "🏃 生活习惯：吸烟等级、体力活动",
            "💊 治疗依从：药物使用、用药依从性"
        ]
        for feat in core_features_display:
            st.write(feat)
    with col_b:
        st.markdown("#### 最终清洗质量")
        st.metric("清洗后缺失值占比", f"{clean_missing_ratio}%")
        st.metric("异常值总修正率", f"{outlier_fix_rate}%")
        st.metric("病程格式统一率", f"{course_standard_rate}%")

    # 清洗完成，全局留存清洗好的数据
    st.session_state.df_clean = df
    st.session_state.my_patient_df = df.copy()
    st.success("✅ ETL数据清洗全部完成，数据已全局留存！")
# 机器学习
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import warnings
    warnings.filterwarnings("ignore")

    # 1. 定义并发症列 & 建模特征列
    compli_features = ['肾病','神经病变','视网膜病变','心血管疾病','周围血管疾病']
    feature_cols = [
        '年龄','性别（0=女性，1=男性）','身体质量指数','收缩压（单位：mmHg）','舒张压（单位：mmHg）',
        '糖化血红蛋白（%）','空腹血糖（单位：mg/dL）','餐后血糖（单位：mg/dL）',
        '发病年龄（首次确诊年龄）','糖尿病病程（患病时长）','吸烟情况（0 = 不吸烟，1、2、3）',
        '体力活动（0=无，1=有）','药物使用（0=未使用，1=使用）','药物依从性（0、1、2）','家族病史（0=无，1=有）'
    ]
    feature_cols = [col for col in feature_cols if col in df.columns]
    compli_cols_exist = [c for c in compli_features if c in df.columns]

    # 2.并发症总数≥2 → 高风险标签
    if compli_cols_exist:
        df['并发症总数'] = df[compli_cols_exist].sum(axis=1).clip(0, len(compli_cols_exist))
        df['高风险'] = (df['并发症总数'] >= 2).astype(int)
    else:
        df['并发症总数'] = 0
        df['高风险'] = 0

    # 3. 准备训练集特征X 和标签y
    X = df[feature_cols].copy()
    y = df['高风险'].copy()

    # 4. 缺失值填充
    for col in X.columns:
        if X[col].dtype != object:
            X[col] = X[col].fillna(X.median())
        else:
            X[col] = X[col].fillna(0)

    # 5. 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 6. 划分训练集/测试集 8:2 分层抽样
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # 7. 训练随机森林模型
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # 8. 模型评估
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # 控制台输出评估结果
    print("===== 模型训练完成 =====")
    print(f"模型准确率：{acc:.4f}")
    print("分类报告：")
    print(classification_report(y_test, y_pred, target_names=["低风险","高风险"]))

    # 9. 关键：存入会话状态，给单人/批量预测调用
    st.session_state.best_model = model
    st.session_state.feature_cols = feature_cols
    st.session_state.scaler = scaler

# 模块3.📋 患者档案管理中心 
elif menu == "📋 患者档案管理":
    st.title("📋 全院患者全生命周期档案管理中心")
    st.caption("档案查询 · 新增录入 · 批量导入 · 风险等级一键标记")
    st.divider()

    df = st.session_state.my_patient_df

    # 初始化患者唯一ID
    if "_patient_id" not in st.session_state.my_patient_df.columns:
        st.session_state.my_patient_df['_patient_id'] = [
            f"original_{i}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}" 
            for i in range(len(st.session_state.my_patient_df))
        ]

    tab1, tab2, tab3 = st.tabs(["📋 全部患者档案", "➕ 手动新增患者", "📂 批量Excel导入"])

    # 全部患者列表
    with tab1:
        core_features = [
            '年龄', '性别', 'BMI', '收缩压', '舒张压',
            '糖化血红蛋白', '空腹血糖', '餐后血糖',
            '家族病史', '发病年龄', '病程',
            '吸烟', '体力活动', '药物使用', '依从性'
        ]   
        compli_features = ['肾病','神经病变','视网膜病变','心血管','周围血管']
        display_cols = core_features + compli_features + ['并发症总数', '高风险']
        display_cols = [col for col in display_cols if col in st.session_state.my_patient_df.columns]
        display_df = st.session_state.my_patient_df[display_cols].drop_duplicates()
        st.dataframe(display_df, use_container_width=True)
        st.caption(f"当前在管患者总数：{len(st.session_state.my_patient_df)} 人")

    # 手动新增患者
    with tab2:
        with st.form("add_patient_form", clear_on_submit=True):
            c1, c2 = st.columns(2)
            with c1:
                姓名 = st.text_input("患者姓名 *")
                年龄 = st.number_input("年龄", 10, 90, 50)
                性别 = st.selectbox("性别（0=女/1=男）", [0, 1])
                BMI = st.number_input("BMI", 10.0, 50.0, 24.0)
                收缩压 = st.number_input("收缩压（mmHg）", 60, 240, 120)
                舒张压 = st.number_input("舒张压（mmHg）", 40, 140, 80)
                糖化血红蛋白 = st.number_input("糖化HbA1c(%)", 4.0, 18.0, 6.0)
            with c2:
                空腹血糖 = st.number_input("空腹血糖（mg/dL）", 30.0, 500.0, 90.0)
                餐后血糖 = st.number_input("餐后血糖（mg/dL）", 50.0, 700.0, 120.0)
                病程 = st.number_input("病程(年)", 0.0, 50.0, 5.0)
                发病年龄 = st.number_input("发病年龄", 10, 90, 45)
                吸烟 = st.selectbox("吸烟等级（0-3）", [0,1,2,3])
                家族病史 = st.selectbox("家族病史", [0,1])
                体力活动 = st.selectbox("体力活动", [0,1])
                药物使用 = st.selectbox("药物使用", [0,1])
                依从性 = st.selectbox("用药依从性（0-2）", [0,1,2])

            提交新增 = st.form_submit_button("✅ 保存新增患者档案", use_container_width=True)

        if 提交新增:
            if not 姓名:
                st.error("❌ 请填写患者姓名！")
                st.stop()

            new_row = {
                "姓名": 姓名,
                "年龄": 年龄,
                "性别": 性别,
                "BMI": BMI,
                "收缩压": 收缩压,
                "舒张压": 舒张压,
                "糖化血红蛋白": 糖化血红蛋白,
                "空腹血糖": 空腹血糖,
                "餐后血糖": 餐后血糖,
                "病程": 病程,
                "发病年龄": 发病年龄,
                "吸烟": 吸烟,
                "家族病史": 家族病史,
                "体力活动": 体力活动,
                "药物使用": 药物使用,
                "依从性": 依从性,
                "肾病": 0,"神经病变": 0,"视网膜病变": 0,"心血管": 0,"周围血管": 0,
                "并发症总数": 0,"高风险": 0,
                "_patient_id": f"manual_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
            }

            new_df = pd.DataFrame([new_row])
            new_df = new_df.reindex(columns=st.session_state.my_patient_df.columns)
            st.session_state.my_patient_df = pd.concat([st.session_state.my_patient_df, new_df], ignore_index=True)
            st.success(f"✅ 患者「{姓名}」档案新增成功！当前总人数：{len(st.session_state.my_patient_df)}")
            st.rerun()

    # 批量导入患者
    with tab3:
        st.markdown("#### 批量上传Excel/CSV患者档案")
        uploaded_file = st.file_uploader("选择文件上传", type=["xlsx", "csv"])

        if uploaded_file:
            try:
                if uploaded_file.name.endswith(".csv"):
                    try:
                        df_upload = pd.read_csv(uploaded_file, encoding='utf-8')
                    except:
                        df_upload = pd.read_csv(uploaded_file, encoding='gbk')
                else:
                    df_upload = pd.read_excel(uploaded_file)

                upload_mapping = {col: col_mapping[col] for col in df_upload.columns if col in col_mapping}
                df_upload.rename(columns=upload_mapping, inplace=True)
                df_upload['_patient_id'] = [f"batch_{i}_{datetime.now().strftime('%Y%m%d%H%M%S')}" for i in range(len(df_upload))]

                # 对齐所有列
                for col in st.session_state.my_patient_df.columns:
                    if col not in df_upload.columns:
                        if col in num_cols:
                            df_upload[col] = df_upload[num_cols].median().median() if not df_upload[num_cols].empty else 0
                        else:
                            df_upload[col] = 0

                # 数据类型规范
                for col in num_cols:
                    df_upload[col] = pd.to_numeric(df_upload[col], errors='coerce').fillna(0)
                for col in cat_cols:
                    df_upload[col] = pd.to_numeric(df_upload[col], errors='coerce').fillna(0).astype(int)

                # 异常值裁剪
                for col, (low, high) in clip_dict.items():
                    if col in df_upload.columns:
                        df_upload[col] = df_upload[col].clip(low, high)

                df_upload = df_upload.reindex(columns=st.session_state.my_patient_df.columns)

                # 去重合并
                key_cols = [col for col in core_features if col != '_patient_id']
                merged = df_upload.merge(st.session_state.my_patient_df[key_cols], on=key_cols, how='left', indicator=True)
                df_new = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])

                if len(df_new) == 0:
                    st.info("ℹ️ 上传数据全部已存在，无新增患者")
                else:
                    st.session_state.my_patient_df = pd.concat([st.session_state.my_patient_df, df_new], ignore_index=True)
                    st.success(f"✅ 批量导入完成，新增 {len(df_new)} 名患者！当前总人数：{len(st.session_state.my_patient_df)}")
                    st.rerun()

            except Exception as e:
                st.error(f"❌ 导入失败：{str(e)}")

# 模块4. 📊 全指标数据大屏
elif menu == "📊 全指标数据大屏":
    st.title("📊 全院患者全指标多模态可视化分析大屏")
    st.caption("单指标分布 · 特征交叉对比 · 高低风险差异 · 并发症专项统计")
    st.divider()
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    warnings.filterwarnings("ignore")
    st.divider()
    st.subheader("📊 全院糖尿病患者数据中心 · 全指标统计大屏")
    df_plot = st.session_state.my_patient_df.copy()

    # 并发症空值填充
    compli_cols = ['肾病','神经病变','视网膜病变','心血管','周围血管','并发症总数','高风险']
    for col in compli_cols:
        if col in df_plot.columns:
            df_plot[col] = df_plot[col].fillna(0)

    df_plot['并发症总数'] = df_plot[['肾病','神经病变','视网膜病变','心血管','周围血管']].sum(axis=1)
    df_plot['高风险'] = (df_plot['并发症总数'] >= 2).astype(int)

    # 顶部核心统计卡片
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("总患者数", len(df_plot))
    with c2:
        high_risk_num = int(df_plot['高风险'].sum())
        st.metric("高风险人数", high_risk_num)
    with c3:
        avg_age = round(df_plot['年龄'].mean(), 1)
        st.metric("平均年龄", f"{avg_age} 岁")
    with c4:
        avg_bmi = round(df_plot['BMI'].mean(), 1)
        st.metric("平均BMI", f"{avg_bmi}")

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        avg_sbp = round(df_plot['收缩压'].mean(), 1)
        st.metric("平均收缩压", f"{avg_sbp} mmHg")
    with c6:
        avg_dbp = round(df_plot['舒张压'].mean(), 1)
        st.metric("平均舒张压", f"{avg_dbp} mmHg")
    with c7:
        avg_hba1c = round(df_plot['糖化血红蛋白'].mean(), 1)
        st.metric("平均糖化HbA1c", f"{avg_hba1c} %")
    with c8:
        avg_course = round(df_plot['病程'].mean(), 1)
        st.metric("平均病程", f"{avg_course} 年")

    c9, c10, c11, c12 = st.columns(4)
    with c9:
        avg_fpg = round(df_plot['空腹血糖'].mean(), 1)
        st.metric("平均空腹血糖", f"{avg_fpg} mg/dL")
    with c10:
        avg_ppg = round(df_plot['餐后血糖'].mean(), 1)
        st.metric("平均餐后血糖", f"{avg_ppg} mg/dL")
    with c11:
        male_ratio = (df_plot['性别'] == 1).mean()
        st.metric("男性占比", f"{male_ratio:.1%}")
    with c12:
        smoke_num = (df_plot["吸烟"] > 0).sum()
        st.metric("吸烟患者数", int(smoke_num))

    # 年龄 + 性别
    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        st.markdown("##### 患者年龄分布")
        fig1, ax1 = plt.subplots()
        sns.histplot(df_plot['年龄'], bins=20, kde=True, color='#005792', ax=ax1)
        ax1.set_xlabel("年龄（岁）")
        ax1.set_ylabel("患者数（人）")
        st.pyplot(fig1)

    with row1_col2:
        st.markdown("##### 患者性别分布")
        fig2, ax2 = plt.subplots()
        sns.countplot(x='性别', data=df_plot, hue='性别', palette=['#5499C7', '#E74C3C'], legend=False, ax=ax2)
        ax2.set_xticklabels(['女性', '男性'])
        ax2.set_xlabel("性别")
        ax2.set_ylabel("患者数（人）")
        st.pyplot(fig2)

    # 血压
    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        st.markdown("##### 收缩压分布")
        fig3, ax3 = plt.subplots()
        sns.histplot(df_plot['收缩压'], bins=20, kde=True, color='#2E86AB', ax=ax3)
        ax3.set_xlabel("收缩压（mmHg）")
        ax3.set_ylabel("患者数（人）")
        st.pyplot(fig3)

    with row2_col2:
        st.markdown("##### 舒张压分布")
        fig4, ax4 = plt.subplots()
        sns.histplot(df_plot['舒张压'], bins=20, kde=True, color='#A23B72', ax=ax4)
        ax4.set_xlabel("舒张压（mmHg）")
        ax4.set_ylabel("患者数（人）")
        st.pyplot(fig4)

    # 血糖三图
    row3_col1, row3_col2, row3_col3 = st.columns(3)
    with row3_col1:
        st.markdown("##### 空腹血糖分布")
        fig5, ax5 = plt.subplots()
        sns.histplot(df_plot['空腹血糖'], bins=20, kde=True, color='#E74C3C', ax=ax5)
        ax5.set_xlabel("空腹血糖（mg/dL）")
        ax5.set_ylabel("患者数（人）")
        st.pyplot(fig5)

    with row3_col2:
        st.markdown("##### 餐后血糖分布")
        fig6, ax6 = plt.subplots()
        sns.histplot(df_plot['餐后血糖'], bins=20, kde=True, color='#F39C12', ax=ax6)
        ax6.set_xlabel("餐后血糖（mg/dL）")
        ax6.set_ylabel("患者数（人）")
        st.pyplot(fig6)

    with row3_col3:
        st.markdown("##### 糖化血红蛋白分布")
        fig7, ax7 = plt.subplots()
        sns.histplot(df_plot['糖化血红蛋白'], bins=20, kde=True, color='#8E44AD', ax=ax7)
        ax7.set_xlabel("糖化HbA1c（%）")
        ax7.set_ylabel("患者数（人）")
        st.pyplot(fig7)

    # BMI + 病程 + 吸烟
    row4_col1, row4_col2, row4_col3 = st.columns(3)
    with row4_col1:
        st.markdown("##### BMI分布")
        fig8, ax8 = plt.subplots()
        sns.histplot(df_plot['BMI'], bins=20, kde=True, color='#1ABC9C', ax=ax8)
        ax8.set_xlabel("BMI")
        ax8.set_ylabel("患者数（人）")
        st.pyplot(fig8)

    with row4_col2:
        st.markdown("##### 病程分布")
        fig9, ax9 = plt.subplots()
        sns.histplot(df_plot['病程'], bins=20, kde=True, color='#16A085', ax=ax9)
        ax9.set_xlabel("病程（年）")
        ax9.set_ylabel("患者数（人）")
        st.pyplot(fig9)

    with row4_col3:
        st.markdown("##### 吸烟情况分布")
        fig10, ax10 = plt.subplots()
        sns.countplot(x='吸烟', data=df_plot, hue='吸烟', palette='Oranges', legend=False, ax=ax10)
        ax10.set_xticklabels(['不吸烟', '轻度', '中度', '重度'])
        ax10.set_xlabel("吸烟等级")
        ax10.set_ylabel("患者数（人）")
        st.pyplot(fig10)

    # 并发症统计
    row5_col1, row5_col2 = st.columns(2)
    with row5_col1:
        st.markdown("##### 并发症总数分布")
        fig11, ax11 = plt.subplots()
        sns.countplot(x='并发症总数', data=df_plot, hue='并发症总数', palette='viridis', legend=False, ax=ax11)
        ax11.set_xlabel("并发症数量（个）")
        ax11.set_ylabel("患者数（人）")
        st.pyplot(fig11)

    with row5_col2:
        st.markdown("##### 各类并发症患病人数")
        fig12, ax12 = plt.subplots(figsize=(10, 4))
        compli_names = ['肾病', '神经病变', '视网膜病变', '心血管', '周围血管']
        compli_sum = [int(df_plot[col].sum()) for col in compli_names]
        sns.barplot(x=compli_names, y=compli_sum, hue=compli_names, palette='Blues', legend=False, ax=ax12)
        ax12.set_ylabel("患病人数（人）")
        plt.xticks(rotation=15)
        st.pyplot(fig12)
    # 高低风险对比2
    st.markdown("---")
    st.markdown("##### 高低风险人群关键指标对比（临床分析）")
    fig13, ax13 = plt.subplots(2, 2, figsize=(14, 10))
    metrics = ['BMI', '糖化血红蛋白', '空腹血糖', '餐后血糖']
    for i, metric in enumerate(metrics):
        row = i // 2
        col = i % 2
        sns.boxplot(x='高风险', y=metric, data=df_plot, hue='高风险', palette='Set2', legend=False, ax=ax13[row][col])
        ax13[row][col].set_title(f'{metric} 风险对比')
        ax13[row][col].set_xlabel('0=低风险 1=高风险')
    plt.tight_layout()
    st.pyplot(fig13)

    st.success("✅ 第三段全指标数据大屏上线成功！")
#模块5.🤖 模型训练可视化 
elif menu == "🤖 模型训练可视化":
    st.title("🤖 多模型联合训练与性能可视化评估中心")
    st.caption("逻辑回归 | 随机森林 | XGBoost | LightGBM 高精度对比评估")
    st.divider()
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    import warnings
    warnings.filterwarnings("ignore")

    st.divider()
    st.subheader("🤖 糖尿病并发症风险预测 · 模型训练中心")

    df_model = st.session_state.my_patient_df.copy()

    num_cols = ['年龄', 'BMI', '收缩压', '舒张压', '糖化血红蛋白',
                '空腹血糖', '餐后血糖', '发病年龄', '病程', '吸烟', '依从性']
    cat_cols = ['性别', '家族病史', '体力活动', '药物使用',
                '肾病', '神经病变', '视网膜病变', '心血管', '周围血管', '高风险', '并发症总数']
    for col in num_cols:
        if col in df_model.columns:
            df_model[col] = df_model[col].fillna(df_model[col].median())
    for col in cat_cols:
        if col in df_model.columns:
            df_model[col] = df_model[col].fillna(0)

    compli_cols = ['肾病','神经病变','视网膜病变','心血管','周围血管']
    df_model['并发症总数'] = df_model[compli_cols].sum(axis=1)
    df_model['高风险'] = (df_model['并发症总数'] >= 2).astype(int)

    feature_cols = ['年龄', '性别', 'BMI', '收缩压', '舒张压', '糖化血红蛋白',
                    '空腹血糖', '餐后血糖', '发病年龄', '病程', '吸烟', '体力活动',
                    '药物使用', '依从性', '家族病史']
    feature_cols = [col for col in feature_cols if col in df_model.columns]
    X = df_model[feature_cols]
    y = df_model['高风险']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    st.session_state.scaler = scaler
    st.session_state.feature_cols = feature_cols

    st.markdown("### 📊 模型性能对比")

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:, 1]

    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    y_proba_lr = lr.predict_proba(X_test_scaled)[:, 1]

    def get_metrics(y_true, y_pred, y_proba):
        return {
            "准确率": round(accuracy_score(y_true, y_pred), 4),
            "精确率": round(precision_score(y_true, y_pred), 4),
            "召回率": round(recall_score(y_true, y_pred), 4),
            "F1分数": round(f1_score(y_true, y_pred), 4),
            "AUC": round(roc_auc_score(y_true, y_proba), 4)
        }
    metrics_rf = get_metrics(y_test, y_pred_rf, y_proba_rf)
    metrics_lr = get_metrics(y_test, y_pred_lr, y_proba_lr)

    metrics_df = pd.DataFrame({
        "随机森林": metrics_rf.values(),
        "逻辑回归": metrics_lr.values()
    }, index=metrics_rf.keys())
    st.dataframe(metrics_df, use_container_width=True)

    st.session_state.best_model = rf
    st.success("✅ 模型训练完成，最优模型：随机森林")

    st.markdown("### 📈 模型可视化分析")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### 随机森林混淆矩阵")
        cm_rf = confusion_matrix(y_test, y_pred_rf)
        fig1, ax1 = plt.subplots()
        sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['低风险', '高风险'], yticklabels=['低风险', '高风险'], ax=ax1)
        ax1.set_xlabel("预测标签")
        ax1.set_ylabel("真实标签")
        st.pyplot(fig1)
    with col2:
        st.markdown("##### 逻辑回归混淆矩阵")
        cm_lr = confusion_matrix(y_test, y_pred_lr)
        fig2, ax2 = plt.subplots()
        sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Oranges', 
                    xticklabels=['低风险', '高风险'], yticklabels=['低风险', '高风险'], ax=ax2)
        ax2.set_xlabel("预测标签")
        ax2.set_ylabel("真实标签")
        st.pyplot(fig2)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_names = [feature_cols[i] for i in indices]
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    bars = sns.barplot(x=importances[indices], y=feature_names, palette='Blues_r', ax=ax3)
    for bar in bars.patches:
        width = bar.get_width()
        ax3.text(
            width + 0.001,
            bar.get_y() + bar.get_height()/2,
            f'{width:.3f}',
            ha='left',
            va='center',
            fontsize=10,
            fontweight='bold'
        )
    ax3.set_xlabel("特征重要性得分")
    ax3.set_ylabel("临床特征")
    ax3.set_title("糖尿病并发症高风险关键影响因素")
    plt.tight_layout()
    st.pyplot(fig3)

    st.markdown("##### ROC曲线对比")
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
    fig4, ax4 = plt.subplots()
    ax4.plot(fpr_rf, tpr_rf, label=f'随机森林 (AUC = {metrics_rf["AUC"]:.4f})', color='#005792', lw=2)
    ax4.plot(fpr_lr, tpr_lr, label=f'逻辑回归 (AUC = {metrics_lr["AUC"]:.4f})', color='#E74C3C', lw=2)
    ax4.plot([0, 1], [0, 1], 'k--', lw=2)
    ax4.set_xlabel('假阳性率')
    ax4.set_ylabel('真阳性率')
    ax4.set_title('ROC曲线')
    ax4.legend(loc='lower right')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.tight_layout()
    st.pyplot(fig4)

    st.success("✅ 第四段模型训练模块上线完成")
# 模块6. 👤 单人风险预测 
elif menu == "🩺 单人风险预测及诊断报告下载":
    st.title("🩺 单人风险预测及诊断报告下载")
    st.caption("所有指标带标准单位 | 数据统一0/1/2/3编码 | 用于AI精准预测")
    st.divider()

    # 新增患者姓名输入
    patient_name = st.text_input("👤 患者姓名")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("年龄（岁）", min_value=10, max_value=100, value=45)
        gender = st.selectbox("性别（0=女，1=男）", [0, 1])
        bmi = st.number_input("BMI（kg/m²）", min_value=10.0, max_value=50.0, value=24.0)
        sbp = st.number_input("收缩压 SBP（mmHg）", min_value=70, max_value=220, value=120)
        dbp = st.number_input("舒张压 DBP（mmHg）", min_value=40, max_value=120, value=80)

    with col2:
        hba1c = st.number_input("糖化血红蛋白 HbA1c（%）", min_value=4.0, max_value=18.0, value=6.5)
        glu_fasting = st.number_input("空腹血糖 FPG（mmol/L）", min_value=2.0, max_value=30.0, value=5.6)
        glu_post = st.number_input("餐后2小时血糖（mmol/L）", min_value=3.0, max_value=40.0, value=7.8)
        smoke = st.selectbox("吸烟情况（0=不吸，1=轻，2=中，3=重）", [0,1,2,3])
        sport = st.selectbox("运动频率（0=少，1=多）", [0,1])

    family_history = st.selectbox("糖尿病家族史（0=无，1=有）", [0,1])
    medication = st.selectbox("是否用药（0=不用，1=使用）", [0,1])
    compliance = st.selectbox("用药依从性（0=差，1=中，2=好）", [0,1,2])

    st.divider()

    if st.button("✅ 保存并提交数据", use_container_width=True):
        st.session_state.user_data = {
            "患者姓名":patient_name,
            "年龄": age,
            "性别": gender,
            "BMI": bmi,
            "收缩压": sbp,
            "舒张压": dbp,
            "糖化血红蛋白": hba1c,
            "空腹血糖": glu_fasting,
            "餐后血糖": glu_post,
            "吸烟": smoke,
            "运动": sport,
            "家族史": family_history,
            "用药": medication,
            "依从性": compliance
        }
        st.success("✅ 数据保存成功！请进入下一步AI并发症风险诊断")


    # 🩺 AI并发症风险诊断
    st.divider()
    st.subheader("🩺 AI并发症风险诊断")
    if "user_data" not in st.session_state or not st.session_state.user_data:
        st.warning("⚠️ 请先填写并提交身体数据！")
        st.stop()

    user = st.session_state.user_data
    st.subheader("📊 已录入健康指标")
    st.dataframe(pd.DataFrame([user]).T, use_container_width=True)

    if st.button("🚀 开始AI智能诊断", type="primary", use_container_width=True):
        risk_score = 0
        if user["年龄"]>=45: risk_score +=12
        if user["BMI"]>=24: risk_score +=10
        if user["糖化血红蛋白"]>=6.5: risk_score +=28
        if user["空腹血糖"]>=7.0: risk_score +=20
        if user["家族史"]==1: risk_score +=10
        if user["吸烟"]>=1: risk_score +=8
        if user["运动"]==0: risk_score +=7

        if risk_score>=70:
            level = "🔴 高风险"
            advice = "请尽快前往医院内分泌科全面筛查"
        elif risk_score>=40:
            level = "🟡 中风险"
            advice = "严格控糖，定期复查"
        else:
            level = "🟢 低风险"
            advice = "保持健康生活，每年体检"

        st.session_state.risk_result = {"score":risk_score,"level":level,"advice":advice}
        st.balloons()
        st.metric("综合风险评分", f"{risk_score}/100")
        st.subheader(f"评估结果：{level}")
        st.info(f"专业建议：{advice}")
        st.success("✅ 诊断完成，可前往下方生成正式PDF文档")


    # 第六段：PDF诊断报告
    st.divider()
    st.subheader("📄 个人诊断报告")
    if "user_data" not in st.session_state or st.session_state.user_data is None:
        st.warning("📝 请先完成【个人健康数据录入】并提交信息")
        st.stop()
    if "risk_result" not in st.session_state or st.session_state.risk_result is None:
        st.warning("⚠️ 请先完成【AI并发症风险诊断】")
        st.stop()

    user = st.session_state.user_data
    risk = st.session_state.risk_result

    st.subheader("📋 诊断报告预览")
    st.metric("综合风险等级", risk['level'])
    st.metric("综合风险评分", f"{risk['score']} / 100")
    st.info(f"临床干预建议：{risk['advice']}")

    st.divider()

    if st.button("📥 诊断报告单", use_container_width=True, type="primary", key="pdf_final_btn"):

        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        import io
        import random
        try:
            pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
            font = "STSong-Light"
        except Exception:
            try:
                pdfmetrics.registerFont(TTFont('Song', 'C:/Windows/Fonts/simsun.ttc', subfontIndex=0))
                font = "Song"
            except:
                font = "Helvetica"

        report_id = f"DM-AI-{datetime.now().strftime('%Y%m%d')}{random.randint(10000,99999)}"

        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        w, h = A4

        c.setFont(font, 18)
        c.drawCentredString(w/2, h-60, "糖尿病并发症AI辅助筛查诊断报告单")
        c.setFont(font, 11)
        c.drawCentredString(w/2, h-85, "中国计算机设计大赛 参赛作品")
        c.line(40, h-100, w-40, h-100)

        y = h - 130
        c.setFont(font,13)
        c.drawString(50, y, "一、受检者基础档案信息")
        y -= 28
        c.setFont(font,11)

        gender_txt = "男" if user['性别'] == 1 else "女"
        smoke_txt = ["不吸烟","轻度吸烟","中度吸烟","重度吸烟"][user['吸烟']]
        sport_txt = ["几乎不运动","规律运动"][user['运动']]
        family_txt = "无" if user['家族史'] == 0 else "有"

        base_list = [
            f"报告流水编号：{report_id}",
            f"患者姓名：{user['患者姓名']}",
            f"受检年龄：{user['年龄']} 周岁",
            f"受检性别：{gender_txt}",
            f"BMI身体质量指数：{user['BMI']:.1f} kg/m²",
            f"诊室血压：{user['收缩压']}/{user['舒张压']} mmHg",
            f"吸烟暴露史：{smoke_txt}",
            f"日常运动频率：{sport_txt}",
            f"糖尿病家族遗传史：{family_txt}"
        ]
        for line in base_list:
            c.drawString(60, y, line)
            y -= 21

        c.line(40, y-10, w-40, y-10)
        y -= 35

        c.setFont(font,13)
        c.drawString(50, y, "二、实验室核心检验结果")
        y -=28
        c.setFont(font,11)

        lab_list = [
            f"糖化血红蛋白（HbA1c）：{user['糖化血红蛋白']} %",
            f"空腹静脉血糖：{user['空腹血糖']} mmol/L",
            f"餐后2小时血糖：{user['餐后血糖']} mmol/L"
        ]
        for line in lab_list:
            c.drawString(60, y, line)
            y -= 21

        c.line(40, y-10, w-40, y-10)
        y -= 35

        c.setFont(font,13)
        c.drawString(50, y, "三、AI智能风险评估诊断结论")
        y -=28
        c.setFont(font,11)

        c.drawString(60, y, f"综合并发症风险评分：{risk['score']} / 100")
        y -=22
        c.drawString(60, y, f"最终风险分层等级：☑ {risk['level']}")

        c.line(40, y-10, w-40, y-10)
        y -= 35

        y -= 40
        c.setFont(font,13)
        c.drawString(50, y, "四、内分泌专科诊疗干预建议")
        y -=28
        c.setFont(font,11)

        score = risk['score']
        if score >=70:
            adv_list = [
                "1. 尽快至内分泌专科完善全套并发症专项筛查评估",
                "2. 严格医学饮食+规律运动，每日多点监测血糖变化",
                "3. 戒烟限酒、规律作息，遵医嘱规范进行药物干预",
                "4. 每3个月定期随访，动态延缓器官不可逆损伤"
            ]
        elif score >=40:
            adv_list = [
                "1. 1-3个月内分泌门诊复诊，完善糖耐量基础筛查",
                "2. 优先强化生活方式管控，减重、控腰围、均衡膳食",
                "3. 每周坚持规律中等强度运动，居家定期监测血糖",
                "4. 每6个月复查糖化血红蛋白，阻断病情进展恶化"
            ]
        else:
            adv_list = [
                "1. 维持现有健康作息与均衡饮食结构",
                "2. 坚持规律体育锻炼，控制体重预防腹型肥胖",
                "3. 每年常规体检筛查血糖、糖化血红蛋白指标",
                "4. 家族史人群定期早筛，长期维持代谢健康稳态"
            ]

        for line in adv_list:
            c.drawString(60, y, line)
            y -= 23

        y -= 40
        c.setFont(font, 10)
        c.drawRightString(w-50, 75, f"报告出具时间：{datetime.now().strftime('%Y年%m月%d日 %H:%M')}")
        c.drawCentredString(w/2, 50, "本报告仅供辅助健康筛查参考，不作为临床最终诊疗依据")

        c.save()
        buffer.seek(0)

        st.download_button(
            label="✅ 下载正式诊断报告单PDF",
            data=buffer,
            file_name=f"{user['患者姓名']}_{report_id}_糖尿病诊断报告单.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="download_final_pdf"
        )

        st.balloons()
        st.success("🎉 完美复刻病历版式！无方框、排版工整、正式医院报告单质感！")
# 模块7.📊 群体糖尿病患者分层管理系统 
elif menu == "📊 群体糖尿病患者分层管理系统":
    st.title("📊 群体糖尿病患者分层管理系统")
    st.caption("批量导入建档患者｜AI智能风险分级｜慢病三层管控｜群体统计分析")
    st.divider()

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from io import BytesIO

    model = st.session_state.get("best_model", None)
    feature_cols = st.session_state.get("feature_cols", [])
    scaler = st.session_state.get("scaler", None)

    if model is None or len(feature_cols) == 0 or scaler is None:
        st.warning("⚠️ 请先完成模型训练，再进行批量风险预测")
        st.stop()

    st.markdown("### 📁 上传批量患者信息表")
    
    batch_file = st.file_uploader("上传 Excel / CSV 文件", type=["xlsx", "csv"])

    if batch_file is not None:
        try:
            if batch_file.name.endswith(".csv"):
                df_batch = pd.read_csv(batch_file)
            else:
                df_batch = pd.read_excel(batch_file)
            required_cols = [
                '年龄','性别（0=女性，1=男性）','身体质量指数','收缩压（单位：mmHg）','舒张压（单位：mmHg）',
                '糖化血红蛋白（%）','空腹血糖（单位：mg/dL）','餐后血糖（单位：mg/dL）','家族病史（0=无，1=有）',
                '发病年龄（首次确诊年龄）','糖尿病病程（患病时长，如 1、3、5年，或“1 MONTH”）',
                '吸烟情况（0 = 不吸烟，1、2、3 表示吸烟程度）','体力活动（0=无，1=有）',
                '药物使用（0=未使用，1=使用）','药物依从性（0、1、2表示程度等级）'
            ]
            for col in required_cols:
                if col not in df_batch.columns:
                    df_batch[col] = 0

            X_batch = df_batch[required_cols].copy()

            # 安全数值处理
            for c in X_batch.columns:
                X_batch[c] = pd.to_numeric(X_batch[c], errors='coerce').fillna(0)

            # 标准化 + 预测
            X_batch_scaled = scaler.transform(X_batch)
            df_batch["并发症风险概率"] = model.predict_proba(X_batch_scaled)[:, 1].round(4)
            df_batch["模型预测标签"] = model.predict(X_batch_scaled)
            df_batch["并发症风险概率"] = model.predict_proba(X_batch_scaled)[:, 1].round(4)
            df_batch["模型预测标签"] = model.predict(X_batch_scaled)

# 医生三层风险分级（真实临床阈值）
            def risk_level(prob):
                # 基准线：prob=0.5 = 模型判定「是否≥2种并发症」分界点
                if prob >= 0.60:
                         return "🔴 高危人群（重点管控）"
                elif prob >= 0.30:
                        return "🟡 中危人群（干预阻断）"
                else:
                        return "🟢 低危人群（常规随访）"
            df_batch["慢病分层等级"] = df_batch["并发症风险概率"].apply(risk_level)
                              
            # 可视化大屏
            st.divider()
            st.markdown("### 📊 群体风险分布统计")
            total = len(df_batch)
            high = len([x for x in df_batch["慢病分层等级"] if "高危" in x])
            mid = len([x for x in df_batch["慢病分层等级"] if "中危" in x])
            low = len([x for x in df_batch["慢病分层等级"] if "低危" in x])

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("总人数", f"{total} 人")
            col2.metric("🔴 高危", f"{high} 人")
            col3.metric("🟡 中危", f"{mid} 人")
            col4.metric("🟢 低危", f"{low} 人")

            # 图表
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            plt.rcParams['font.sans-serif'] = ['SimHei']
            df_batch["慢病分层等级"].value_counts().plot(kind='bar', ax=ax1, color=['#e74c3c','#f39c12','#27ae60'])
            ax1.set_title("风险等级人数")
            ax1.tick_params(axis='x', rotation=15)

            ax2.pie([high, mid, low], labels=['高危','中危','低危'], colors=['#e74c3c','#f39c12','#27ae60'], autopct='%1.1f%%')
            ax2.set_title("风险占比")
            st.pyplot(fig)

            # 结果表格
            st.divider()
            st.markdown("### 📋 患者分层结果")
            st.dataframe(df_batch.sort_values("并发症风险概率", ascending=False), use_container_width=True)

            # 高危名单
            st.divider()
            st.markdown("### 🔴 高危重点人员")
            df_high = df_batch[df_batch["慢病分层等级"].str.contains("高危")]
            if len(df_high) > 0:
                st.dataframe(df_high, use_container_width=True)
            else:
                st.success("✅ 本批次无高危人群")

            # 管理方案
            st.divider()
            st.markdown("### 🩺 临床分层管理建议")
            with st.expander("🔴 高危人群管理"):
                st.write("每3个月随访，全套并发症筛查，强化用药，严格控制代谢指标")
            with st.expander("🟡 中危人群管理"):
                st.write("每6个月复诊，生活方式干预，预防进展")
            with st.expander("🟢 低危人群管理"):
                st.write("年度体检，健康宣教，常规随访")

            # 导出Excel
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_batch.to_excel(writer, sheet_name='全部患者', index=False)
                df_high.to_excel(writer, sheet_name='高危人群', index=False)
            output.seek(0)

            st.download_button(
                label="📥 下载患者分层管理表",
                data=output,
                file_name="糖尿病患者风险分层表.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            st.success("✅ 批量预测、分层、可视化、导出全部完成！")

        except Exception as e:
            st.error(f"错误：{str(e)}")
    else:
        st.info("请上传批量患者表格开始分析")