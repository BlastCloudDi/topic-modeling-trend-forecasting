# app.py – Дашборд прогнозирования смысловых трендов
# SARIMA vs CatBoost vs LSTM vs Seasonal Naive

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from statsmodels.tsa.statespace.sarimax import SARIMAX
from catboost import CatBoostRegressor
import torch
from torch import nn


# 1. Архитектура LSTM

class LSTMForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, horizon, output_dim, dropout=0.3):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_dim1, hidden_dim2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim2, horizon * output_dim)
        self.horizon = horizon
        self.output_dim = output_dim

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = out[:, -1, :]
        out = self.dropout2(out)
        out = self.fc(out)
        return out.view(-1, self.horizon, self.output_dim)


# 2. Настройка страницы

st.set_page_config(page_title="Прогнозирование трендов", page_icon="", layout="wide")
st.title(" Прогнозирование смысловых трендов в стратегическом дискурсе")
st.markdown("*Модуль прогнозирования (SARIMA, CatBoost, LSTM, Naive) тематических трендов*")
st.markdown("---")


# 3. Загрузка моделей и данных

@st.cache_resource
def load_all():
    models = {'sarima': {}, 'catboost': {}, 'lstm': None, 'scaler': None}
    data = {'trends': None, 'metadata': {}, 'metrics': {}}
    
    # Актуальный список тем
    topic_ids = [
        '0_дрон_беспилотный_бас', '1_стартап_бизнес_предприниматель',
        '3_креативный_индустрия_экономика', '5_туризм_туристический_маршрут',
        '6_бренд_победитель_одежда'
    ]
    
    # Загрузка SARIMA
    for topic in topic_ids:
        path = f"sarima_{topic}.pkl"
        if os.path.exists(path):
            models['sarima'][topic] = joblib.load(path)
            
    # Загрузка CatBoost
    for topic in topic_ids:
        path = f"catboost_{topic}.cbm"
        if os.path.exists(path):
            cb = CatBoostRegressor()
            cb.load_model(path)
            models['catboost'][topic] = cb
            
    # Загрузка LSTM
    if os.path.exists("lstm_model.pt"):
        checkpoint = torch.load("lstm_model.pt", map_location='cpu', weights_only=False)
        model_lstm = LSTMForecaster(
            checkpoint['input_dim'], checkpoint['hidden_dim1'], checkpoint['hidden_dim2'],
            checkpoint['horizon'], checkpoint['output_dim'], checkpoint['dropout']
        )
        model_lstm.load_state_dict(checkpoint['model_state_dict'])
        model_lstm.eval()
        models['lstm'] = model_lstm
        models['lstm_config'] = checkpoint
        
    # Загрузка Scaler
    if os.path.exists("scaler.pkl"):
        models['scaler'] = joblib.load("scaler.pkl")
        
    # Загрузка данных
    if os.path.exists("topic_trends_top5.csv"):
        df = pd.read_csv("topic_trends_top5.csv", index_col=0, parse_dates=True)
        # Оставляем только активные темы, чтобы индексы совпадали с моделями
        data['trends'] = df[topic_ids].sort_index()
        
    if os.path.exists("topic_metadata.json"):
        with open("topic_metadata.json", "r", encoding="utf-8") as f:
            data['metadata'] = json.load(f)
            
    if os.path.exists("metrics_summary.json"):
        with open("metrics_summary.json", "r") as f:
            data['metrics'] = json.load(f)
            
    return models, data

models, data = load_all()
trends = data['trends']
metadata = data.get('metadata', {})
metrics_info = data.get('metrics', {})

LOOK_BACK = 52
HORIZON = 12

# Словарь отображаемых названий
topic_names = {
    '0_дрон_беспилотный_бас': '🚁 Беспилотные системы',
    '1_стартап_бизнес_предприниматель': '💡 Предпринимательство',
    '3_креативный_индустрия_экономика': '🎨 Креативные индустрии',
    '5_туризм_туристический_маршрут': '🏔️ Туризм',
    '6_бренд_победитель_одежда': '👗 Бренд одежды'
}


# 4. Функции прогнозирования

def forecast_sarima(topic_name, steps):
    model = models['sarima'].get(topic_name)
    if model is not None:
        try:
            forecast = model.forecast(steps=steps)
            return np.clip(forecast.values, 0, None)
        except:
            return np.zeros(steps)
    return np.zeros(steps)

def forecast_catboost(topic_name, steps):
    model = models['catboost'].get(topic_name)
    if model is None or trends is None: 
        return np.zeros(steps)
    
    topic_idx = list(trends.columns).index(topic_name)
    history = trends.iloc[-LOOK_BACK:].copy().astype(float)
    forecasts = []
    
    for _ in range(steps):
        features = history.values.flatten().reshape(1, -1)
        pred = max(0.0, model.predict(features)[0])
        forecasts.append(pred)
        
        # Корректный сдвиг окна
        new_row = history.iloc[-1].copy()
        new_row[topic_name] = pred
        history = pd.concat([history.iloc[1:], new_row.to_frame().T], ignore_index=True)
        
    return np.array(forecasts)

def forecast_lstm(topic_name, steps):
    model_lstm = models.get('lstm')
    scaler = models.get('scaler')
    
    if model_lstm is None or scaler is None or trends is None: return np.zeros(steps)
    
    topic_idx = list(trends.columns).index(topic_name)
    last_data = trends.iloc[-LOOK_BACK:].values
    scaled = scaler.transform(last_data)
    X_input = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        pred_scaled = model_lstm(X_input).cpu().numpy().reshape(HORIZON, -1)
    pred = scaler.inverse_transform(pred_scaled)
    return np.clip(pred[:steps, topic_idx], 0, None)

def forecast_naive(topic_name, steps, seasonal_period=52):
    if trends is None: return np.zeros(steps)
    series = trends[topic_name]
    last_season = series.values[-seasonal_period:]
    forecast = np.tile(last_season, (steps // seasonal_period) + 1)[:steps]
    return np.clip(forecast, 0, None)


# 5. Интерфейс

with st.sidebar:
    st.header("⚙️ Настройки")
    
    selected_topic = st.selectbox("Выберите тему", list(topic_names.keys()),
                                  format_func=lambda x: topic_names.get(x, x))
    
    horizon = st.slider("Горизонт прогноза (недель)", min_value=4, max_value=24, value=12, step=4)
    
    model_type = st.radio("Модель прогнозирования",
                          ["LSTM", "SARIMA", "CatBoost", "Seasonal Naive", "Сравнение всех"],
                          index=0)
    
    st.markdown("---")
    st.header("ℹ️ О данных")
    st.markdown(f"**Период:** {trends.index[0].date()} – {trends.index[-1].date()}")
    st.markdown(f"**Наблюдений:** {len(trends)} недель")
    st.markdown(f"**Источники:** АСИ, НТИ, Сильные идеи")
    
    st.markdown("---")
    st.header("🔑 Ключевые слова")
    keywords = metadata.get(selected_topic, {}).get('keywords', [])
    for word in keywords[:8]:
        st.markdown(f"- {word}")


# 6. Основная область

col1, col2 = st.columns([3, 1])

with col1:
    st.subheader(f"Динамика темы: {topic_names.get(selected_topic, selected_topic)}")
    
    series = trends[selected_topic]
    last_date = series.index[-1]
    
    # Генерация дат прогноза
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1),
                                   periods=horizon, freq='W')
    
    fig = go.Figure()
    
    # История
    history = series.iloc[-60:]
    fig.add_trace(go.Scatter(x=history.index, y=history.values,
                             mode='lines', name='История',
                             line=dict(color='#1f77b4', width=2)))
    
    # Модели
    show_models = [model_type] if model_type != "Сравнение всех" else ["SARIMA", "CatBoost", "LSTM", "Seasonal Naive"]
    
    if "SARIMA" in show_models:
        sarima_pred = forecast_sarima(selected_topic, horizon)
        fig.add_trace(go.Scatter(x=forecast_dates, y=sarima_pred,
                                 mode='lines', name='SARIMA',
                                 line=dict(color='#ff7f0e', width=2, dash='dash'),
                                 marker=dict(size=4)))
                                 
    if "CatBoost" in show_models:
        catboost_pred = forecast_catboost(selected_topic, horizon)
        fig.add_trace(go.Scatter(x=forecast_dates, y=catboost_pred,
                                 mode='lines', name='CatBoost',
                                 line=dict(color='#9467bd', width=2, dash='dashdot'),
                                 marker=dict(size=4)))
                                 
    if "LSTM" in show_models:
        lstm_pred = forecast_lstm(selected_topic, horizon)
        fig.add_trace(go.Scatter(x=forecast_dates, y=lstm_pred,
                                 mode='lines', name='LSTM',
                                 line=dict(color='#2ca02c', width=2, dash='dot'),
                                 marker=dict(size=4)))
                                 
    if "Seasonal Naive" in show_models:
        naive_pred = forecast_naive(selected_topic, horizon)
        fig.add_trace(go.Scatter(x=forecast_dates, y=naive_pred,
                                 mode='lines', name='Seasonal Naive',
                                 line=dict(color='#808080', width=2, dash='dash'),
                                 marker=dict(size=4, symbol='square')))
    
    # Разделитель train/test
    fig.add_vline(x=last_date, line_width=1.5, line_dash="solid", line_color="gray", opacity=0.6)
    fig.add_annotation(x=last_date, y=history.max() * 0.9,
                       text="← История | Прогноз →", showarrow=False,
                       font=dict(size=11, color="gray", family="Arial Black"))
    
    fig.update_layout(height=480, xaxis_title="Дата", yaxis_title="Сообщений в неделю",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      hovermode='x unified', template='plotly_white')
    
    st.plotly_chart(fig, width="stretch")
    
    # Метрики темы
    st.subheader("📈 Аналитика темы")
    m1, m2, m3, m4 = st.columns(4)
    
    with m1: st.metric("Среднее (вся история)", f"{series.mean():.1f}")
    
    with m2:
        # Сглаженное значение за последние 4 недели против предыдущих 4 недель
        recent_mean = series.tail(4).mean()
        prev_mean = series.iloc[-8:-4].mean()
        delta = recent_mean - prev_mean
        st.metric("Среднее (4 нед.)", f"{recent_mean:.1f}", delta=f"{delta:+.1f}")
        
    with m3:
        # Краткосрочный тренд (сравнение последнего месяца с последним кварталом)
        is_growing = series.tail(4).mean() > series.tail(12).mean()
        trend = "↗️ Рост" if is_growing else "↘️ Спад"
        st.metric("Краткосрочный тренд", trend)
    
    with m4:
        # Вычисляем среднее значение прогноза для выбранной модели
        if model_type == "CatBoost":
            pred_mean = forecast_catboost(selected_topic, horizon).mean()
        elif model_type == "LSTM":
            pred_mean = forecast_lstm(selected_topic, horizon).mean()
        elif model_type == "SARIMA":
            pred_mean = forecast_sarima(selected_topic, horizon).mean()
        elif model_type == "Seasonal Naive":
            pred_mean = forecast_naive(selected_topic, horizon).mean()
        else:  # "Сравнение всех" - показываем среднее по LSTM как основной
            pred_mean = forecast_lstm(selected_topic, horizon).mean()
            
        st.metric("Прогноз (выбранная модель)", f"{pred_mean:.1f}")

with col2:
    st.subheader("📊 Распределение тем")
    topic_sums = trends.sum().sort_values(ascending=False)
    topic_labels = [topic_names.get(t, t) for t in topic_sums.index]
    
    fig_pie = go.Figure(data=[go.Pie(labels=topic_labels, values=topic_sums.values, hole=0.45,
                                     marker=dict(colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']))])
    fig_pie.update_layout(height=320, showlegend=True, 
                          legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
                          margin=dict(b=40))
    st.plotly_chart(fig_pie, width="stretch")
    
    st.markdown("---")
    st.subheader("📋 Сводка по темам")
    for t in trends.columns:
        total = trends[t].sum()
        last = trends[t].iloc[-1]
        st.markdown(f"**{topic_names.get(t, t)}**")
        st.markdown(f"Всего: `{int(total)}` | Тек. нед.: `{int(last)}`")
    
    st.markdown("---")
    st.subheader("🎯 Сравнение моделей (RMSE)")
    if metrics_info:
        detailed = metrics_info.get('detailed', {})
        topic_metrics = detailed.get(selected_topic, {})
        
        rows = []
        for model, key in [("SARIMA", 'SARIMA_RMSE'), 
                            ("CatBoost", 'CatBoost_RMSE'), 
                            ("LSTM", 'LSTM_RMSE')]:
            val = topic_metrics.get(key)
            if val is not None:
                rows.append({"Модель": model, "RMSE": float(val)})
        
        if rows:
            metrics_df = pd.DataFrame(rows)
            st.dataframe(metrics_df, hide_index=True, width="stretch")
        else:
            st.info("Метрики по выбранной теме отсутствуют в файле metrics_summary.json")
    else:
        st.info("Файл metrics_summary.json не найден. Метрики не отображаются.")


# 7. Футер

st.markdown("---")
st.caption("© 2026 ВКР | Пайплайн: BERTopic → Анализ → Прогноз (SARIMA/CatBoost/LSTM/Naive) → Streamlit")