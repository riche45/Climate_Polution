# Predicción de Contaminantes Atmosféricos

Este proyecto se centra en la predicción de niveles de contaminantes atmosféricos utilizando técnicas de machine learning. El objetivo es desarrollar modelos que puedan predecir con precisión las concentraciones de diferentes contaminantes en el aire, lo que puede ser crucial para la toma de decisiones en materia de calidad del aire y salud pública.

## Estructura del Proyecto

```
hackathon-schneider-pollution/
├── data/
│   ├── raw/           # Datos originales
│   └── processed/     # Datos procesados
├── models/            # Modelos entrenados
├── predictions/       # Predicciones generadas
├── reports/
│   └── figures/      # Visualizaciones generadas
└── src/
    ├── data/         # Scripts de procesamiento de datos
    ├── models/       # Scripts de entrenamiento de modelos
    └── visualization/# Scripts de visualización
```

## Contaminantes Monitoreados

- SO2 (Dióxido de Azufre)
- NO2 (Dióxido de Nitrógeno)
- O3 (Ozono)
- CO (Monóxido de Carbono)
- PM10 (Partículas en suspensión ≤ 10 µm)
- PM2.5 (Partículas en suspensión ≤ 2.5 µm)

## Resultados

### Task 2: Predicción de Contaminantes

El modelo desarrollado para la Task 2 logró un coeficiente de determinación (R²) de 0.57, demostrando una capacidad moderada para predecir los niveles de contaminantes. Las predicciones se basan en patrones temporales y características ambientales.

#### Visualizaciones Task 2

1. **Series Temporales**
   - Muestra la evolución temporal de cada contaminante
   - Identifica picos significativos y tendencias

2. **Patrones Diarios**
   - Visualiza la variación de contaminantes a lo largo del día
   - Destaca las horas punta y patrones de actividad

3. **Patrones Semanales**
   - Muestra la variación de contaminantes por día de la semana
   - Identifica diferencias entre días laborables y fines de semana

4. **Correlación entre Contaminantes**
   - Mapa de calor que muestra las relaciones entre diferentes contaminantes
   - Ayuda a identificar fuentes comunes de contaminación

### Task 3: Predicción Optimizada

El modelo optimizado para la Task 3 alcanzó un coeficiente de determinación (R²) de 0.89, representando una mejora significativa en la precisión de las predicciones. Este resultado se logró mediante:

- Optimización de hiperparámetros
- Ingeniería de características avanzada
- Selección de características más relevantes
- Técnicas de regularización

#### Visualizaciones Task 3

1. **Series Temporales Optimizadas**
   - Muestra las predicciones mejoradas
   - Compara con los patrones observados

2. **Patrones Diarios Optimizados**
   - Visualiza los patrones diarios con mayor precisión
   - Incluye bandas de confianza

3. **Patrones Semanales Optimizados**
   - Muestra la variación semanal con mayor detalle
   - Identifica patrones estacionales

4. **Correlación entre Contaminantes Optimizada**
   - Mapa de calor mejorado
   - Muestra relaciones más precisas entre contaminantes

## Instalación y Uso

1. Clonar el repositorio:
```bash
git clone https://github.com/tu-usuario/hackathon-schneider-pollution.git
cd hackathon-schneider-pollution
```

2. Crear y activar el entorno virtual:
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

4. Generar predicciones:
```bash
python src/models/task2_optimized_model.py
python src/models/task3_optimized_model.py
```

5. Generar visualizaciones:
```bash
python src/visualization/generate_visualizations.py
```

## Requisitos

- Python 3.8+
- pandas
- numpy
- scikit-learn
- plotly
- joblib

## Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue primero para discutir los cambios que te gustaría hacer.

## Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.
