import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json

def load_predictions():
    """Cargar predicciones desde el archivo JSON"""
    with open('predictions/predictions_task_3.json', 'r') as f:
        return json.load(f)

def create_dataframe(predictions):
    """Convertir predicciones a DataFrame con información adicional"""
    # Mapeo de contaminantes a nombres y unidades
    pollutant_info = {
        "206": {"name": "SO2", "unit": "ppm", "color": "#FF6B6B"},
        "211": {"name": "NO2", "unit": "ppm", "color": "#4ECDC4"},
        "217": {"name": "O3", "unit": "ppm", "color": "#45B7D1"},
        "219": {"name": "CO", "unit": "ppm", "color": "#96CEB4"},
        "225": {"name": "PM10", "unit": "µg/m³", "color": "#FFEEAD"},
        "228": {"name": "PM2.5", "unit": "µg/m³", "color": "#D4A5A5"}
    }
    
    # Crear DataFrame
    dfs = []
    for station_code, data in predictions["target"].items():
        df = pd.DataFrame.from_dict(data, orient='index')
        df.index = pd.to_datetime(df.index)
        df.columns = ['value']
        df['station'] = station_code
        df['pollutant'] = pollutant_info[station_code]["name"]
        df['unit'] = pollutant_info[station_code]["unit"]
        df['color'] = pollutant_info[station_code]["color"]
        dfs.append(df)
    
    return pd.concat(dfs)

def plot_time_series(df):
    """Crear gráfico de series temporales para cada contaminante"""
    fig = go.Figure()
    
    for pollutant in df['pollutant'].unique():
        pollutant_data = df[df['pollutant'] == pollutant]
        station_code = pollutant_data['station'].iloc[0]
        unit = pollutant_data['unit'].iloc[0]
        color = pollutant_data['color'].iloc[0]
        
        fig.add_trace(go.Scatter(
            x=pollutant_data.index,
            y=pollutant_data['value'],
            name=f"{pollutant} ({unit})",
            line=dict(color=color),
            mode='lines'
        ))
        
        # Añadir anotaciones para picos significativos
        peaks = pollutant_data[pollutant_data['value'] > pollutant_data['value'].mean() + pollutant_data['value'].std()]
        for idx, row in peaks.iterrows():
            fig.add_annotation(
                x=idx,
                y=row['value'],
                text=f"{row['value']:.2f}",
                showarrow=True,
                arrowhead=1
            )
    
    fig.update_layout(
        title="Evolución temporal de contaminantes",
        xaxis_title="Fecha",
        yaxis_title="Concentración",
        template="plotly_white",
        height=600
    )
    
    return fig

def plot_daily_patterns(df):
    """Crear gráfico de patrones diarios"""
    fig = go.Figure()
    
    for pollutant in df['pollutant'].unique():
        pollutant_data = df[df['pollutant'] == pollutant]
        station_code = pollutant_data['station'].iloc[0]
        unit = pollutant_data['unit'].iloc[0]
        color = pollutant_data['color'].iloc[0]
        
        # Calcular promedio por hora
        hourly_avg = pollutant_data.groupby(pollutant_data.index.hour)['value'].mean()
        hourly_std = pollutant_data.groupby(pollutant_data.index.hour)['value'].std()
        
        fig.add_trace(go.Scatter(
            x=hourly_avg.index,
            y=hourly_avg,
            name=f"{pollutant} ({unit})",
            line=dict(color=color),
            mode='lines+markers'
        ))
        
        # Añadir banda de confianza
        fig.add_trace(go.Scatter(
            x=list(hourly_avg.index) + list(hourly_avg.index)[::-1],
            y=list(hourly_avg + hourly_std) + list(hourly_avg - hourly_std)[::-1],
            fill='toself',
            fillcolor=color,
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False
        ))
        
        # Resaltar horas punta
        fig.add_vrect(x0=7, x1=9, fillcolor="gray", opacity=0.2, line_width=0)
        fig.add_vrect(x0=17, x1=19, fillcolor="gray", opacity=0.2, line_width=0)
    
    fig.update_layout(
        title="Patrones diarios de contaminantes",
        xaxis_title="Hora del día",
        yaxis_title="Concentración promedio",
        template="plotly_white",
        height=500
    )
    
    return fig

def plot_weekly_patterns(df):
    """Crear gráfico de patrones semanales"""
    fig = go.Figure()
    
    for pollutant in df['pollutant'].unique():
        pollutant_data = df[df['pollutant'] == pollutant]
        station_code = pollutant_data['station'].iloc[0]
        unit = pollutant_data['unit'].iloc[0]
        color = pollutant_data['color'].iloc[0]
        
        # Calcular promedio por día de la semana
        weekly_avg = pollutant_data.groupby(pollutant_data.index.weekday)['value'].mean()
        
        fig.add_trace(go.Bar(
            x=['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo'],
            y=weekly_avg,
            name=f"{pollutant} ({unit})",
            marker_color=color
        ))
    
    fig.update_layout(
        title="Patrones semanales de contaminantes",
        xaxis_title="Día de la semana",
        yaxis_title="Concentración promedio",
        template="plotly_white",
        height=500
    )
    
    return fig

def plot_pollutant_correlation(df):
    """Crear mapa de calor de correlaciones entre contaminantes"""
    # Crear tabla pivote para correlación
    pivot_df = df.pivot_table(
        values='value',
        index=df.index,
        columns='pollutant',
        aggfunc='first'
    )
    
    # Calcular matriz de correlación
    corr_matrix = pivot_df.corr()
    
    # Crear mapa de calor
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title="Correlación entre contaminantes",
        template="plotly_white",
        height=600
    )
    
    return fig

def main():
    """Función principal para generar todas las visualizaciones"""
    print("Generando visualizaciones para Task 3...")
    
    # Cargar predicciones
    predictions = load_predictions()
    
    # Crear DataFrame
    df = create_dataframe(predictions)
    
    # Crear directorio de salida
    output_dir = Path('reports/figures/predicciones task 3')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generar y guardar visualizaciones
    print("Generando series temporales...")
    fig_time = plot_time_series(df)
    fig_time.write_html(output_dir / 'series_temporales.html')
    
    print("Generando patrones diarios...")
    fig_daily = plot_daily_patterns(df)
    fig_daily.write_html(output_dir / 'patrones_diarios.html')
    
    print("Generando patrones semanales...")
    fig_weekly = plot_weekly_patterns(df)
    fig_weekly.write_html(output_dir / 'patrones_semanales.html')
    
    print("Generando mapa de correlaciones...")
    fig_corr = plot_pollutant_correlation(df)
    fig_corr.write_html(output_dir / 'correlacion_contaminantes.html')
    
    print("Visualizaciones generadas exitosamente en:", output_dir)

if __name__ == "__main__":
    main() 