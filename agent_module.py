# agent_module.py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage
import os
import tempfile

# Инициализация модели
llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0,
    num_predict=250  
)

# Инструмент: загрузка CSV
@tool
def load_csv(query: str) -> str:
    """
    Загружает CSV-файл и возвращает краткую информацию о данных.
    
    Аргументы:
        query: путь к файлу (например, "sales_data.csv")
    
    Возвращает:
        Строку с количеством строк/столбцов и именами столбцов
        или сообщение об ошибке.
    """
    try:
        if not query or not isinstance(query, str):
            return "Ошибка: пустой или некорректный путь к файлу"
        
        df = pd.read_csv(query)
        if df.empty:
            return f"Ошибка: файл '{query}' пустой."
        
        num_rows = len(df)
        num_cols = len(df.columns)
        columns_str = ", ".join(df.columns.tolist())
        return f"Загружено {num_rows} строк, {num_cols} столбцов. Столбцы: {columns_str}"
    
    except FileNotFoundError:
        return f"Ошибка: файл '{query}' не найден. Проверь путь к файлу."
    except pd.errors.EmptyDataError:
        return f"Ошибка: файл '{query}' пустой или содержит только заголовки."
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(query, encoding='latin1')
            num_rows = len(df)
            num_cols = len(df.columns)
            columns_str = ", ".join(df.columns.tolist())
            return f"Загружено {num_rows} строк, {num_cols} столбцов (кодировка latin1). Столбцы: {columns_str}"
        except Exception as e2:
            return f"Ошибка кодировки: {e2}"
    except Exception as e:
        return f"Неизвестная ошибка при загрузке файла '{query}': {type(e).__name__} - {e}"

# Инструмент: статистика
@tool
def describe_data(query: str) -> str:
    """
    Возвращает статистику по числовым столбцам CSV-файла.
    
    Аргументы:
        query: путь к файлу (например, "sales_data.csv")
    
    Возвращает:
        Текстовую таблицу с мин/макс/средним или сообщение об ошибке.
    """
    try:
        if not query or not isinstance(query, str):
            return "Ошибка: пустой или некорректный путь к файлу"
        
        df = pd.read_csv(query)
        if df.empty:
            return f"Ошибка: файл '{query}' пустой."
        
        stats = df.describe()
        if stats.empty:
            return f"Ошибка: в файле '{query}' нет числовых столбцов для анализа."
        
        return stats.to_string()
    
    except FileNotFoundError:
        return f"Ошибка: файл '{query}' не найден. Проверь путь к файлу."
    except Exception as e:
        return f"Неизвестная ошибка при получении статистики: {type(e).__name__} - {e}"

# Инструмент: график по годам основания магазинов
@tool
def plot_trend(query: str) -> str:
    """
    Строит график распределения продаж по годам основания магазинов.
    
    Аргументы:
        query: путь к CSV-файлу
    
    Возвращает:
        Сообщение об успешном сохранении графика или ошибку.
    """
    try:
        if not query or not isinstance(query, str):
            return "Ошибка: пустой или некорректный путь к файлу"
        
        try:
            df = pd.read_csv(query)
        except UnicodeDecodeError:
            df = pd.read_csv(query, encoding='latin1')
        
        if df.empty:
            return f"Ошибка: файл '{query}' пустой."
        
        # Гибкий поиск столбцов
        year_col = None
        sales_col = None
        for col in df.columns:
            col_lower = col.lower()
            if 'year' in col_lower and 'establish' in col_lower:
                year_col = col
            if 'sales' in col_lower or 'outlet_sales' in col_lower:
                sales_col = col
        
        if not year_col:
            return f"Ошибка: не найден столбец с годом основания. Доступные столбцы: {', '.join(df.columns)}"
        if not sales_col:
            return f"Ошибка: не найден столбец с продажами. Доступные столбцы: {', '.join(df.columns)}"
        
        sales_by_year = df.groupby(year_col)[sales_col].sum().sort_index()
        if sales_by_year.empty:
            return f"Ошибка: нет данных для построения графика после группировки."
        
        plt.figure(figsize=(10, 6))
        plt.bar(sales_by_year.index.astype(str), sales_by_year.values, color='#2E86AB')
        plt.title('Суммарные продажи по годам основания магазинов', fontsize=14, fontweight='bold')
        plt.xlabel('Год основания магазина')
        plt.ylabel('Суммарные продажи')
        plt.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('sales_trend.png', dpi=150)
        plt.close()
        
        return (f"✅ График сохранён как 'sales_trend.png'.\n"
                f"Диапазон лет: {sales_by_year.index.min()}–{sales_by_year.index.max()}\n"
                f"Всего записей: {len(df)}")
    
    except FileNotFoundError:
        return f"Ошибка: файл '{query}' не найден. Проверь путь к файлу."
    except Exception as e:
        return f"Ошибка при построении графика: {type(e).__name__} - {e}"

# Инструмент: поиск аномалий методом IQR
@tool
def find_outliers(query: str) -> str:
    """
    Находит аномальные значения продаж (выбросы) методом IQR.
    
    Аргументы:
        query: путь к CSV-файлу
    
    Возвращает:
        Строку с количеством аномалий, примерами и бизнес-интерпретацией
        или сообщение об ошибке.
    """
    try:
        if not query or not isinstance(query, str):
            return "Ошибка: пустой или некорректный путь к файлу"
        
        try:
            df = pd.read_csv(query)
        except UnicodeDecodeError:
            df = pd.read_csv(query, encoding='latin1')
        
        if df.empty:
            return f"Ошибка: файл '{query}' пустой."
        
        # Гибкий поиск столбца с продажами
        sales_col = None
        for col in df.columns:
            col_lower = col.lower()
            if 'sales' in col_lower or 'outlet_sales' in col_lower:
                sales_col = col
                break
        
        if not sales_col:
            return f"Ошибка: не найден столбец с продажами. Доступные столбцы: {', '.join(df.columns)}"
        
        # Удаляем пропуски в столбце продаж
        df_clean = df.dropna(subset=[sales_col])
        if df_clean.empty:
            return "Ошибка: все значения в столбце продаж отсутствуют (NaN)."
        
        # Расчёт границ IQR
        sales_values = df_clean[sales_col].values
        q1 = np.percentile(sales_values, 25)
        q3 = np.percentile(sales_values, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Фильтрация аномалий
        outliers = df_clean[
            (df_clean[sales_col] < lower_bound) | 
            (df_clean[sales_col] > upper_bound)
        ]
        
        # Формирование ответа
        if len(outliers) == 0:
            return f"Аномалии не обнаружены. Все продажи находятся в диапазоне [{lower_bound:.2f}, {upper_bound:.2f}]."
        
        # Примеры аномалий (первые 3 записи)
        examples = []
        for idx, row in outliers.head(3).iterrows():
            examples.append(f"- Запись #{idx}: {row[sales_col]:.2f}")
        
        examples_text = "\n".join(examples)
        
        return (
            f"Найдено {len(outliers)} аномальных продаж вне диапазона [{lower_bound:.2f}, {upper_bound:.2f}].\n\n"
            f"Примеры аномалий (первые 3 записи):\n{examples_text}\n\n"
            f"Бизнес-интерпретация:\n"
            f"- Высокие аномалии (> {upper_bound:.2f}): вероятно оптовые заказы или ошибки ввода (лишний ноль)\n"
            f"- Низкие аномалии (< {lower_bound:.2f}): возможны возвраты товара или технические ошибки\n"
            f"Рекомендация: проверить {len(outliers)} записей вручную перед удалением из датасета."
        )
    
    except FileNotFoundError:
        return f"Ошибка: файл '{query}' не найден. Проверь путь к файлу."
    except Exception as e:
        return f"Ошибка при поиске аномалий: {type(e).__name__} - {e}"

# Инструмент: Корреляционный анализ
@tool
def correlation_analysis(query: str) -> str:
    """
    Анализирует корреляции между числовыми переменными и связи категориальных признаков с продажами.
    
    Аргументы:
        query: путь к CSV-файлу
    
    Возвращает:
        Топ корреляций + анализ категориальных переменных + бизнес-интерпретацию.
    """
    try:
        if not query or not isinstance(query, str):
            return "Ошибка: пустой или некорректный путь к файлу"
        
        try:
            df = pd.read_csv(query)
        except UnicodeDecodeError:
            df = pd.read_csv(query, encoding='latin1')
        
        if df.empty:
            return f"Ошибка: файл '{query}' пустой."
        
        # Поиск столбца с продажами
        sales_col = None
        for col in df.columns:
            col_lower = col.lower()
            if 'sales' in col_lower or 'outlet_sales' in col_lower:
                sales_col = col
                break
        
        if not sales_col:
            return f"Ошибка: не найден столбец с продажами. Доступные столбцы: {', '.join(df.columns)}"
        
        # Анализ числовых переменных
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != sales_col]
        
        if not numeric_cols:
            numeric_report = "Числовых переменных для корреляции не найдено."
        else:
            corr_matrix = df[numeric_cols + [sales_col]].corr()
            correlations = []
            for col in numeric_cols:
                corr_value = corr_matrix.loc[col, sales_col]
                if abs(corr_value) > 0.3:
                    correlations.append((col, corr_value))
            
            correlations.sort(key=lambda x: abs(x[1]), reverse=True)
            
            if correlations:
                top3 = correlations[:3]
                numeric_report_lines = ["ТОП-3 корреляций с продажами:"]
                for i, (col, corr) in enumerate(top3, 1):
                    direction = "положительная" if corr > 0 else "отрицательная"
                    numeric_report_lines.append(f"{i}. {col} ↔ {sales_col}: {corr:.2f} ({direction})")
                numeric_report = "\n".join(numeric_report_lines)
            else:
                numeric_report = "Сильных корреляций (|коэф| > 0.3) не обнаружено."
        
        # Анализ категориальных переменных
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        categorical_cols = [col for col in categorical_cols if df[col].nunique() <= 50]
        
        if not categorical_cols:
            categorical_report = "Категориальных переменных для анализа не найдено."
        else:
            cat_analysis = []
            for col in categorical_cols[:2]:
                grouped = df.groupby(col)[sales_col].mean().sort_values(ascending=False)
                top_categories = grouped.head(3).to_dict()
                
                cat_analysis.append(f"\nСредние продажи по '{col}':")
                for cat, avg_sales in top_categories.items():
                    cat_analysis.append(f"  • {cat}: {avg_sales:.2f} руб")
            
            categorical_report = "Анализ категориальных переменных:" + "".join(cat_analysis)
        
        # Бизнес-интерпретация
        interpretation = """
Бизнес-интерпретация:
- Положительная корреляция означает: рост переменной → рост продаж
- Отрицательная корреляция означает: рост переменной → падение продаж
- Категориальные различия показывают: какие сегменты генерируют больше выручки

Рекомендация: сконцентрировать маркетинговые усилия на сегментах с высокими средними продажами.
"""
        
        return f"{numeric_report}\n\n{categorical_report}\n\n{interpretation}"
    
    except FileNotFoundError:
        return f"Ошибка: файл '{query}' не найден. Проверь путь к файлу."
    except Exception as e:
        return f"Ошибка при корреляционном анализе: {type(e).__name__} - {e}"

# Функция инициализации агента
def create_agent_executor():
    llm = ChatOllama(model="llama3.1:8b", temperature=0, num_predict=250)
    tools = [load_csv, describe_data, plot_trend, find_outliers, correlation_analysis]
    tool_node = ToolNode(tools)
    
    def should_continue(state: MessagesState):
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and len(last.tool_calls) > 0:
            return "tools"
        return "__end__"
    
    builder = StateGraph(MessagesState)
    builder.add_node("agent", lambda state: {"messages": [llm.bind_tools(tools).invoke(state["messages"])]})
    builder.add_node("tools", tool_node)
    builder.add_edge("agent", "tools")
    builder.add_conditional_edges("tools", should_continue)
    builder.set_entry_point("agent")
    return builder.compile()

# Экспортируем функции для использования в Streamlit
__all__ = ["create_agent_executor", "find_outliers", "correlation_analysis", "plot_trend"]