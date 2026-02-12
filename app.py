import streamlit as st
import pandas as pd
import os
import tempfile
from agent_module import create_agent_executor, find_outliers, correlation_analysis, plot_trend

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# === Настройка страницы ===
st.set_page_config(
    page_title="💜 Агент-аналитик данных",
    page_icon="💜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === Кастомный стиль (динамическая фиолетовая тема) ===
if st.session_state.dark_mode:
    # Тёмная тема
    st.markdown("""
    <style>
        /* Общий фон */
        [data-testid="stAppViewContainer"] {
            background-color: #1E1E1E !important;
        }
        /* Боковая панель */
        [data-testid="stSidebar"] {
            background-color: #2D2D2D !important;
        }
        /* Заголовок */
        h1 {
            color: #FFFFFF !important;
            font-family: 'Arial', sans-serif;
            font-weight: bold;
        }
        /* Кнопки */
        .stButton>button {
            background: linear-gradient(135deg, #B19CD9, #9B59B6) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.6rem 1.2rem !important;
            font-weight: bold !important;
            box-shadow: 0 4px 6px rgba(155, 89, 182, 0.3) !important;
        }
        /* Загрузчик файлов */
        .stFileUploader>label {
            color: #FFFFFF !important;
            font-weight: bold !important;
        }
        /* Текст */
        .stMarkdown, .stText {
            color: #FFFFFF !important;
        }
        /* Метрики */
        [data-testid="stMetric"] {
            background-color: #3A3A3A !important;
            border-radius: 10px !important;
            padding: 1rem !important;
        }
        /* Таблицы */
        .stDataFrame {
            background-color: #2D2D2D !important;
        }
    </style>
    """, unsafe_allow_html=True)
    theme_emoji = "🌙"
    theme_name = "Тёмная тема"
else:
    # Светлая тема
    st.markdown("""
    <style>
        /* Общий фон */
        [data-testid="stAppViewContainer"] {
            background-color: #F8F9FA !important;
        }
        /* Боковая панель */
        [data-testid="stSidebar"] {
            background-color: #E8DAEF !important;
        }
        /* Заголовок */
        h1 {
            color: #6C3483 !important;
            font-family: 'Arial', sans-serif;
            font-weight: bold;
        }
        /* Кнопки */
        .stButton>button {
            background: linear-gradient(135deg, #ca79ea, #9B59B6) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.6rem 1.2rem !important;
            font-weight: bold !important;
            box-shadow: 0 4px 6px rgba(155, 89, 182, 0.2) !important;
        }
        /* Загрузчик файлов */
        .stFileUploader>label {
            color: #6C3483 !important;
            font-weight: bold !important;
        }
        /* Текст */
        .stMarkdown, .stText {
            color: #2C3E50 !important;
        }
        /* Метрики */
        [data-testid="stMetric"] {
            background-color: #FFFFFF !important;
            border-radius: 10px !important;
            padding: 1rem !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
        }
        /* Таблицы */
        .stDataFrame {
            background-color: #FFFFFF !important;
        }
    </style>
    """, unsafe_allow_html=True)
    theme_emoji = "☀️"
    theme_name = "Светлая тема"

# Обновляем заголовок с эмодзи темы
st.title(f"{theme_emoji} Агент-аналитик данных")

# === Заголовок ===
st.title("💜 Агент-аналитик данных")
st.markdown("Загрузи CSV-файл и получи автоматический анализ продаж")

# === Инициализация состояния ===
if "agent_executor" not in st.session_state:
    with st.spinner("🧠 Инициализация агента (первый запуск может занять 20-30 секунд)..."):
        st.session_state.agent_executor = create_agent_executor()

# === Боковая панель: загрузка файла ===
with st.sidebar:
    st.header("📁 Загрузка данных")
    uploaded_file = st.file_uploader("Выбери CSV-файл", type=["csv"])
    
    if uploaded_file is not None:
        st.success(f"✅ Файл загружен: {uploaded_file.name}")
    else:
        st.info("📎 Подсказка: используй датасет с Kaggle")

with st.sidebar:
    if st.button(f"🌙 / ☀️ Переключить ({theme_name})"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

# === Основное окно ===
if uploaded_file is not None:
    try:
        # Сохраняем загруженный файл во временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Загружаем данные для превью
        df = pd.read_csv(tmp_file_path)
        
        # Метрики
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Строки", len(df))
        with col2:
            st.metric("📋 Столбцы", len(df.columns))
        with col3:
            missing = df.isnull().sum().sum()
            st.metric("🔍 Пропуски", missing, delta="-целевые для очистки" if missing > 0 else "чисто", delta_color="off")
        
        # Превью данных
        st.subheader("👀 Превью данных")
        st.dataframe(df.head(10), use_container_width=True)
        
        # === Кнопки анализа ===
        st.markdown("### 🧠 Автоматический анализ")
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        
        with col_btn1:
            btn_outliers = st.button("🔍 Найти аномалии", use_container_width=True)
        
        with col_btn2:
            btn_correlation = st.button("🔗 Корреляции", use_container_width=True)
        
        with col_btn3:
            btn_trend = st.button("📈 График тренда", use_container_width=True)
        
        # === Обработка нажатий кнопок ===
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
                
                # Отображаем график прямо в интерфейсе
                if os.path.exists("sales_trend.png"):
                    st.image("sales_trend.png", caption="Продажи по годам основания магазинов", use_container_width=True)
        
        # Удаляем временный файл
        os.unlink(tmp_file_path)
        
    except Exception as e:
        st.error(f"❌ Ошибка при анализе: {e}")
        st.exception(e)  # Показывает полный трейсбек для отладки
else:
    # Приветственное сообщение
    st.info("👈 Загрузи CSV-файл через боковую панель, чтобы начать анализ")
    
    # Пример возможностей
    st.subheader("💡 Возможности агента")
    st.markdown("""
    После загрузки файла доступны кнопки:
    - 🔍 **Найти аномалии** — автоматическое выявление выбросов (метод IQR)
    - 🔗 **Корреляции** — анализ связей между переменными + категориальные признаки
    - 📈 **График тренда** — визуализация продаж по годам основания магазинов
    
    Все результаты включают **бизнес-интерпретацию** и рекомендации!
    """)
    
    # Скриншот демо (опционально)
    st.image("https://via.placeholder.com/800x400/E8DAEF/6C3483?text=💜+Демо+агента+в+действии", 
             caption="Пример работы агента с данными продаж", use_container_width=True)