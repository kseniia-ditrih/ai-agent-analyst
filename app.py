import streamlit as st
import pandas as pd
import os
import tempfile
from agent_module import create_agent_executor, find_outliers, correlation_analysis, plot_trend

# === Настройка страницы ===
st.set_page_config(
    page_title="💜 Агент-аналитик данных",
    page_icon="💜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === Тёмная фиолетовая тема ===
st.markdown("""
<style>
    /* Общий фон */
    .main { background-color: #1E1E1E; color: white; }
    
    /* Боковая панель */
    .stSidebar { background-color: #2D2D2D; color: white; }
    
    /* Заголовки */
    h1, h2, h3 { color: #B19CD9; }
    
    /* Кнопки */
    .stButton>button {
        background: linear-gradient(135deg, #B19CD9, #9B59B6) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.2rem !important;
        font-weight: bold !important;
    }
    
    /* Метрики */
    .stMetric {
        background-color: #3A3A3A !important;
        color: white !important;
        border-radius: 10px !important;
        padding: 1rem !important;
    }
    
    /* Загрузчик файлов */
    .stFileUploader>label {
        color: #B19CD9 !important;
        font-weight: bold !important;
    }
    .stFileUploader>div>div>button {
        background: linear-gradient(135deg, #B19CD9, #9B59B6) !important;
        color: white !important;
        border-radius: 8px !important;
        font-weight: bold !important;
    }
    
    /* Текст */
    .stMarkdown, .stText {
        color: white !important;
    }
    
    /* Таблицы */
    [data-testid="stDataFrame"] {
        background-color: #2D2D2D !important;
        color: white !important;
    }
    [data-testid="stDataFrame"] th {
        background-color: #3A3A3A !important;
        color: #B19CD9 !important;
    }
</style>
""", unsafe_allow_html=True)

# === Заголовок ===
st.title("💜 Агент-аналитик данных")
st.markdown("Загрузи CSV-файл и получи автоматический анализ продаж")

# === Инициализация агента ===
if "agent_executor" not in st.session_state:
    with st.spinner("🧠 Инициализация агента (первый запуск может занять 20-30 секунд)..."):
        st.session_state.agent_executor = create_agent_executor()

# === Боковая панель (только одна!) ===
with st.sidebar:
    st.header("📁 Загрузка данных")
    uploaded_file = st.file_uploader("Выбери CSV-файл", type=["csv"])
    
    if uploaded_file is not None:
        st.success(f"✅ Файл загружен: {uploaded_file.name}")
    else:
        st.info("📎 Подсказка: используй датасет с Kaggle")

# === Основное окно ===
if uploaded_file is not None:
    try:
        # Сохраняем файл во временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Загружаем данные
        df = pd.read_csv(tmp_file_path)
        
        # Метрики
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Строки", len(df))
        with col2:
            st.metric("📋 Столбцы", len(df.columns))
        with col3:
            missing = df.isnull().sum().sum()
            st.metric("🔍 Пропуски", missing)
        
        # Превью данных
        st.subheader("👀 Превью данных")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Кнопки анализа
        st.markdown("### 🧠 Автоматический анализ")
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        
        with col_btn1:
            btn_outliers = st.button("🔍 Найти аномалии", use_container_width=True)
        
        with col_btn2:
            btn_correlation = st.button("🔗 Корреляции", use_container_width=True)
        
        with col_btn3:
            btn_trend = st.button("📈 График тренда", use_container_width=True)
        
        # Обработка кнопок
        if btn_outliers:
            with st.spinner("🔍 Ищу аномалии методом IQR..."):
                result = find_outliers.invoke(tmp_file_path)
                st.subheader("🚨 Результаты поиска аномалий")
                st.info(result)
        
        if btn_correlation:
            with st.spinner("🔗 Строю матрицу корреляций..."):
                result = correlation_analysis.invoke(tmp_file_path)
                st.subheader("📊 Корреляционный анализ")
                st.info(result)
        
        if btn_trend:
            with st.spinner("📈 Строю график продаж по годам..."):
                result = plot_trend.invoke(tmp_file_path)
                st.subheader("📉 График продаж")
                st.success(result)
                
                if os.path.exists("sales_trend.png"):
                    st.image("sales_trend.png", caption="Продажи по годам основания магазинов", use_container_width=True)
        
        # Удаляем временный файл
        os.unlink(tmp_file_path)
        
    except Exception as e:
        st.error(f"❌ Ошибка при анализе: {e}")
else:
    st.info("👈 Загрузи CSV-файл через боковую панель, чтобы начать анализ")
    
    st.subheader("💡 Возможности агента")
    st.markdown("""
    После загрузки файла доступны кнопки:
    - 🔍 **Найти аномалии** — автоматическое выявление выбросов (метод IQR)
    - 🔗 **Корреляции** — анализ связей между переменными
    - 📈 **График тренда** — визуализация продаж по годам
    
    Все результаты включают **бизнес-интерпретацию**!
    """)