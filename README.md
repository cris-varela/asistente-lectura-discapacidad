# Asistente Inteligente de Lectura para Discapacidad Cognitiva

**Proyecto Final - Curso Especialización IA y Big Data**  
**Alumna:** Cristina  
**Plazo:** 12 marzo - 18 mayo 2026 (10 semanas)

---

## 📋 Descripción

Sistema inteligente que analiza automáticamente la dificultad de textos, agrupa usuarios según sus capacidades de lectura, y ofrece a educadores un panel visual de seguimiento de su progreso.

**Tecnología con propósito social** aplicada a personas con discapacidad cognitiva.

---

## 🎯 Objetivos del Proyecto

### Objetivo General
Facilitar la autonomía comunicativa de personas con discapacidad cognitiva mediante adaptación automática de contenidos textuales.

### Objetivos Específicos (SMART)
- **Clasificador de nivel de lectura** basado en métricas lingüísticas
- **Agrupamiento de usuarios** mediante algoritmos de clustering
- **Panel interactivo** mostrando progreso y recomendaciones
- **Validación técnica** con 1000+ textos y 15 casos de uso documentados

---

## 🛠️ Stack Tecnológico (100% Gratuito)

| Categoría | Herramientas |
|-----------|-------------|
| **Lenguaje** | Python 3.13+ |
| **PLN** | spaCy, Hugging Face Transformers |
| **ML** | scikit-learn (Random Forest, K-means, DBSCAN) |
| **Datos** | pandas, numpy |
| **Visualización** | Streamlit, Plotly, matplotlib |
| **Base de datos** | PostgreSQL |
| **Testing** | pytest |
| **Control versiones** | Git, GitHub |

---

## 📁 Estructura del Proyecto

```
asistente-lectura-discapacidad/
├── data/
│   ├── raw/              # Textos originales (fotos, PDFs)
│   ├── processed/        # Textos etiquetados por nivel
│   └── test/             # Conjunto de prueba
├── notebooks/            # Jupyter notebooks exploración
├── src/
│   ├── preprocessing/    # Limpieza y procesamiento textos
│   ├── models/           # Modelos ML (clasificador, clustering)
│   ├── dashboard/        # Aplicación Streamlit
│   └── utils/            # Funciones auxiliares
├── tests/                # Tests automatizados
├── docs/                 # Documentación
├── models/               # Modelos entrenados (.pkl)
├── requirements.txt      # Dependencias Python
├── .gitignore
└── README.md
```

---

## 🚀 Instalación y Setup

### 1. Clonar el repositorio
```bash
git clone https://github.com/tu-usuario/asistente-lectura-discapacidad.git
cd asistente-lectura-discapacidad
```

### 2. Crear entorno virtual
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
python -m spacy download es_core_news_lg
```

### 4. Configurar estructura
```bash
python setup_project.py
```

---

## 📊 Pipeline del Proyecto

### Semanas 1-2: Recopilación de Datos
- [ ] Obtener 1000+ textos de diferentes niveles
- [ ] Etiquetar por dificultad (1=muy fácil → 5=muy difícil)
- [ ] Organizar en `data/raw/` y `data/processed/`

### Semanas 3-4: Modelo de Clasificación
- [ ] Calcular métricas lingüísticas con spaCy
- [ ] Entrenar Random Forest
- [ ] Validar con conjunto de prueba (precisión ≥85%)

### Semanas 5-6: Clustering de Usuarios
- [ ] Crear 10 perfiles de usuario simulados
- [ ] Aplicar K-means/DBSCAN
- [ ] Validar calidad agrupamiento (silueta ≥0.6)

### Semana 7: Dashboard
- [ ] Desarrollar interfaz Streamlit
- [ ] Visualizaciones: progreso, clustering, métricas

### Semanas 8-9: Casos de Uso y Testing
- [ ] Documentar 15 casos de uso
- [ ] Tests automatizados (cobertura ≥80%)
- [ ] Refinamiento del sistema

### Semana 10: Documentación Final
- [ ] Memoria escrita
- [ ] Presentación
- [ ] Entrega

---

## 📖 Uso Rápido

### Explorar datos iniciales
```bash
jupyter notebook notebooks/01_exploracion_inicial.ipynb
```

### Procesar un texto
```python
from src.preprocessing.text_analyzer import TextAnalyzer

analyzer = TextAnalyzer()
text = "Este es un texto de prueba."
metrics = analyzer.calculate_metrics(text)
print(metrics)
```

### Lanzar dashboard
```bash
streamlit run src/dashboard/app.py
```

---

## 🧪 Testing

```bash
# Ejecutar todos los tests
pytest

# Con cobertura
pytest --cov=src tests/
```

---

## 📚 Recursos y Referencias

- [spaCy Documentation](https://spacy.io/usage)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Índice Flesch-Kincaid en español](https://legible.es/blog/flesch-szigriszt/)

---

## 🤝 Contacto y Contribuciones

**Autora:** Cristina  
**Curso:** Especialización IA y Big Data  
**Año:** 2026

Este proyecto se desarrolla con propósito académico y social.

---

## 📄 Licencia

MIT License - Proyecto de código abierto para uso educativo y social.
