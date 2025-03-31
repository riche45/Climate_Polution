# 🏭 Hackathon Schneider Pollution

## 📊 Proyecto de Predicción de Contaminación Atmosférica

Este proyecto fue desarrollado como parte del Hackathon Schneider Electric, enfocado en la predicción de niveles de contaminación atmosférica en Madrid. El objetivo es demostrar la implementación práctica de modelos de machine learning y análisis de datos para la predicción de contaminantes.

## 🎯 Resultados

### Task 1: Análisis Exploratorio de Datos
- **Score**: 0.89 (R²)
- **Descripción**: Análisis detallado de patrones de contaminación en Madrid, incluyendo:
  - Correlaciones entre contaminantes
  - Patrones temporales
  - Distribución espacial
  - Análisis de estaciones

### Task 2: Predicción de Contaminantes
- **Score**: 57%
- **Descripción**: Predicción de niveles de contaminación para diferentes estaciones y contaminantes:
  - SO2 (Estación 206)
  - NO2 (Estación 211)
  - O3 (Estación 217)
  - CO (Estación 219)
  - PM10 (Estación 225)
  - PM2.5 (Estación 228)

### Task 3: Predicción de Calidad del Aire
- **Score**: 0.89 (R²)
- **Descripción**: Modelo optimizado para predicción de calidad del aire con:
  - Características temporales avanzadas
  - Patrones estacionales
  - Correlaciones entre contaminantes
  - Validación robusta

## 🛠️ Tecnologías Utilizadas
- Python 3.8+
- scikit-learn
- pandas
- numpy
- plotly
- matplotlib
- seaborn

## 📋 Prerrequisitos
- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- Git

## 🔧 Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/riche45/hackathon-schneider-pollution.git
cd hackathon-schneider-pollution
```

2. Crear y activar entorno virtual:
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## 📊 Estructura del Proyecto
```
hackathon-schneider-pollution/
├── data/
│   ├── raw/              # Datos originales
│   └── processed/        # Datos procesados
├── models/               # Modelos entrenados
├── predictions/          # Predicciones generadas
├── reports/
│   └── figures/         # Visualizaciones
├── src/
│   ├── data/            # Scripts de procesamiento
│   ├── models/          # Scripts de modelos
│   └── visualization/   # Scripts de visualización
├── requirements.txt
└── README.md
```

## 🚀 Uso

1. Generar predicciones para Task 2:
```bash
python src/models/task2_optimized_model.py
```

2. Generar predicciones para Task 3:
```bash
python src/models/task3_optimized_model.py
```

3. Generar visualizaciones:
```bash
python src/visualization/generate_visualizations.py
```

## 📈 Visualizaciones

### Task 2: Predicciones por Contaminante
![Predicciones Task 2](reports/figures/task2_predictions.png)
*Predicciones de niveles de contaminación para diferentes estaciones y contaminantes*

### Task 3: Predicción de Calidad del Aire
![Predicciones Task 3](reports/figures/task3_predictions.png)
*Predicciones de calidad del aire con modelo optimizado*

## 🤝 Contribuir
Las contribuciones son bienvenidas. Por favor, abre un issue primero para discutir los cambios que te gustaría realizar.

## 📝 Licencia
Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## 👤 Autor
Richard Garcia - @riche45

Desarrollador especializado en machine learning y análisis de datos ambientales.

## 🙏 Agradecimientos
- Schneider Electric
- scikit-learn
- pandas
- plotly
- matplotlib
- seaborn
