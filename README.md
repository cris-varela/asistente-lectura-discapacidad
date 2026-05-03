# 📚 Asistente Inteligente de Lectura para Personas con Discapacidad Cognitiva

**TFM — Curso de Especialización en Inteligencia Artificial y Big Data**  
**ILERNA | Cristina Varela | 2025-2026**

---

## 📋 Descripción

Sistema inteligente que clasifica textos en español por nivel de dificultad lectora (1-5) y personaliza la experiencia de lectura para personas con discapacidad cognitiva. Integra un clasificador XGBoost con embeddings semánticos, un módulo de clustering de usuarios K-means++ y un dashboard interactivo con Streamlit.

**Resultados técnicos:**
- ✅ Accuracy clasificador: **90,54%** (objetivo ≥ 85%)
- ✅ Coeficiente de silueta clustering: **0,7126** (objetivo ≥ 0,6)
- ✅ Dataset: **2.695 fragmentos** en 5 niveles de dificultad

---

## 🗂️ Estructura del repositorio

```
asistente-lectura-discapacidad/
├── data/
│   └── raw/                          # Dataset etiquetado en 5 niveles
│       ├── nivel_1_lectura_facil/
│       ├── nivel_2_cuentos_simple/
│       ├── nivel_3_cuentos_complejo/
│       ├── nivel_4_narrativo_adulto/
│       └── nivel_5_tecnico_especializado/
├── models/
│   └── clasificador_embeddings.pkl   # Modelo XGBoost entrenado (90,54%)
├── notebooks/
│   ├── 01_clasificador_basico.py     # Pipeline de entrenamiento del clasificador
│   ├── 02_procesamiento_embeddings.ipynb
│   └── 03_clustering_usuarios.py     # Módulo de clustering K-means++
├── outputs/
│   ├── usuarios_clusterizados.csv    # Perfiles de usuario generados
│   ├── perfil_clusters.csv           # Estadísticas por cluster
│   ├── clustering_pca.png            # Visualización PCA
│   └── elbow_silhouette.png          # Gráfico codo y silueta
└── dashboard/
    └── app.py                        # Dashboard Streamlit
```

---

## 🚀 Instalación y ejecución

### Requisitos
- Python 3.10+
- pip

### Pasos

**1. Clonar el repositorio:**
```bash
git clone https://github.com/cris-varela/asistente-lectura-discapacidad.git
cd asistente-lectura-discapacidad
```

**2. Instalar dependencias:**
```bash
pip install streamlit plotly scikit-learn xgboost sentence-transformers pandas numpy
```

**3. Lanzar el dashboard:**
```bash
streamlit run dashboard/app.py
```

El dashboard se abrirá automáticamente en `http://localhost:8501`

> **Nota:** Si la página aparece en blanco, abrir el navegador en modo incógnito y navegar a `http://localhost:8501`

---

## 🔧 Módulos del sistema

### Clasificador de nivel de lectura
```bash
python notebooks/01_clasificador_basico.py
```
Entrena el modelo XGBoost con embeddings semánticos (384 dims) + features lingüísticas (9 dims). Genera `models/clasificador_embeddings.pkl`.

### Módulo de clustering
```bash
python notebooks/03_clustering_usuarios.py
```
Ejecuta K-means++ (k=10) sobre 300 usuarios simulados. Genera los archivos CSV y PNG en `outputs/`.

---

## 📊 Dashboard interactivo

El dashboard tiene tres vistas:

| Vista | Descripción | Audiencia |
|-------|-------------|-----------|
| 🏫 Panel del Educador | Clustering PCA, distribución por niveles, heatmap de actividad, importancia de características | Educador / Profesional de apoyo |
| 👤 Mi progreso | Progreso individual, perfil asignado, textos recomendados | Usuario con DI |
| 🔍 Clasificador de textos | Análisis en tiempo real de cualquier texto | Educador |

---

## 🧪 Tecnologías

| Categoría | Tecnología |
|-----------|-----------|
| Lenguaje | Python 3.10+ |
| Embeddings | Sentence Transformers (paraphrase-multilingual-MiniLM-L12-v2) |
| Clasificador | XGBoost |
| Clustering | scikit-learn K-means++ |
| Dashboard | Streamlit + Plotly |
| Control de versiones | Git / GitHub |

---

## 📈 Resultados del clasificador

| Nivel | Precisión | Recall | F1-score |
|-------|-----------|--------|----------|
| N1 — Lectura Fácil | 0,91 | 0,89 | 0,90 |
| N2 — Básico | 0,89 | 0,91 | 0,90 |
| N3 — Intermedio | 0,88 | 0,90 | 0,89 |
| N4 — Avanzado | 0,87 | 0,85 | 0,86 |
| N5 — Especializado | 0,97 | 0,98 | 0,97 |
| **Media ponderada** | **0,90** | **0,91** | **0,90** |

**Accuracy global: 90,54% | Validación cruzada: 82,82% ± 7,85%**

---

## 👤 Autora

**Cristina Varela**  
Curso de Especialización en IA y Big Data — ILERNA 2025-2026  
Técnico Superior DAM | Diplomada en Educación Social

---

## 📄 Licencia

Proyecto académico de uso educativo. Dataset construido a partir de fuentes públicas de acceso libre.
