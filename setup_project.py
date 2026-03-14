"""
Script de configuración inicial del proyecto
Crea toda la estructura de carpetas necesaria
"""

import os
from pathlib import Path

# Definir estructura de carpetas
FOLDERS = [
    "data/raw",
    "data/processed",
    "data/test",
    "notebooks",
    "src/preprocessing",
    "src/models",
    "src/dashboard",
    "src/utils",
    "tests",
    "docs",
    "models",
    "logs",
    "outputs"
]

def create_project_structure():
    """Crea toda la estructura de carpetas del proyecto"""
    
    print("🚀 Creando estructura del proyecto...")
    print("-" * 50)
    
    for folder in FOLDERS:
        path = Path(folder)
        path.mkdir(parents=True, exist_ok=True)
        
        # Crear archivo __init__.py en carpetas de código Python
        if folder.startswith("src/") or folder == "tests":
            init_file = path / "__init__.py"
            init_file.touch()
            print(f"✅ {folder}/__init__.py")
        else:
            print(f"✅ {folder}/")
    
    print("-" * 50)
    print("✨ Estructura creada exitosamente!")
    print("\n📁 Carpetas principales:")
    print("   data/raw/       → Textos originales (fotos, PDFs)")
    print("   data/processed/ → Textos etiquetados")
    print("   notebooks/      → Jupyter notebooks")
    print("   src/            → Código fuente")
    print("   tests/          → Tests automatizados")
    print("   models/         → Modelos entrenados")
    
    print("\n🎯 Próximos pasos:")
    print("   1. Activar entorno virtual: venv\\Scripts\\activate")
    print("   2. Instalar dependencias: pip install -r requirements.txt")
    print("   3. Descargar modelo spaCy: python -m spacy download es_core_news_lg")
    print("   4. Abrir notebook: jupyter notebook notebooks/01_exploracion_inicial.ipynb")

if __name__ == "__main__":
    create_project_structure()
