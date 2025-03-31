import os
import sys
from pathlib import Path

# Añadir el directorio raíz al path de Python
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

from src.visualization.task2_visualizer import main as task2_main
from src.visualization.task3_visualizer import main as task3_main

def main():
    """Función principal para generar todas las visualizaciones"""
    print("Generando visualizaciones para todas las tareas...")
    
    # Crear directorio de predicciones si no existe
    predictions_dir = root_dir / 'predictions'
    predictions_dir.mkdir(exist_ok=True)
    
    # Generar visualizaciones para Task 2
    print("\nGenerando visualizaciones para Task 2...")
    task2_main()
    
    # Generar visualizaciones para Task 3
    print("\nGenerando visualizaciones para Task 3...")
    task3_main()
    
    print("\nTodas las visualizaciones han sido generadas exitosamente.")

if __name__ == "__main__":
    main() 