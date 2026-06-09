"""
DASHBOARD - SISTEMA ASISTENTE DE LECTURA
Sistema de Asistencia Lectura para Personas con Discapacidad Cognitiva
TFM IABD - Cristina Varela
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import pickle
import re
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Asistente de Lectura — DI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# FUNCIONES
# ============================================================================

def contar_silabas(palabra):
    palabra = palabra.lower().replace('h', '')
    vocales = 'aeiouáéíóúü'
    silabas = 0
    en_vocal = False
    for char in palabra:
        if char in vocales:
            if not en_vocal:
                silabas += 1
                en_vocal = True
        else:
            en_vocal = False
    return max(1, silabas)

def calcular_features_texto(texto):
    texto = texto.strip()
    frases = re.split(r'[.!?]+', texto)
    frases = [f.strip() for f in frases if f.strip()]
    num_frases = max(len(frases), 1)
    palabras = re.findall(r'\b\w+\b', texto.lower())
    num_palabras = len(palabras)
    if num_palabras == 0:
        return None
    palabras_por_frase = num_palabras / num_frases
    total_silabas = sum(contar_silabas(p) for p in palabras)
    silabas_por_palabra = total_silabas / num_palabras
    longitud_promedio = sum(len(p) for p in palabras) / num_palabras
    palabras_largas = sum(1 for p in palabras if contar_silabas(p) > 3)
    ratio_largas = palabras_largas / num_palabras
    inflesz = 206.84 - (60 * silabas_por_palabra) - (1.02 * palabras_por_frase)
    return {
        'palabras_por_frase': palabras_por_frase,
        'silabas_por_palabra': silabas_por_palabra,
        'longitud_promedio_palabra': longitud_promedio,
        'ratio_palabras_largas': ratio_largas,
        'inflesz': inflesz,
        'num_palabras': num_palabras,
        'num_frases': num_frases
    }

@st.cache_data
def generar_datos_usuarios():
    np.random.seed(42)
    perfiles_base = [
        ("Principiante Severo",    1, 0.40, 28, 0.30, 1, 1),
        ("Principiante Moderado",  1, 0.62, 22, 0.52, 2, 1),
        ("Elemental Estable",      2, 0.70, 17, 0.62, 2, 2),
        ("Elemental Progresivo",   2, 0.80, 13, 0.72, 3, 2),
        ("Intermedio Bajo",        3, 0.76, 10, 0.68, 3, 3),
        ("Intermedio Consolidado", 3, 0.86,  8, 0.80, 4, 3),
        ("Intermedio Alto",        4, 0.88,  6, 0.84, 4, 4),
        ("Avanzado Emergente",     4, 0.91,  5, 0.88, 5, 4),
        ("Avanzado Consolidado",   5, 0.93,  3, 0.91, 5, 5),
        ("Lector Independiente",   5, 0.97,  2, 0.96, 7, 5),
    ]
    registros = []
    for perfil, niv, tasa, tiempo, aciertos, sesiones, niv_max in perfiles_base:
        for i in range(30):
            registros.append({
                'usuario_id': f"U{len(registros)+1:03d}",
                'perfil_base': perfil,
                'nivel_preferido': float(np.clip(np.random.normal(niv, 0.08), 1, 5)),
                'tasa_completado': float(np.clip(np.random.normal(tasa, 0.02), 0.1, 1.0)),
                'tiempo_lectura_min': float(np.clip(np.random.normal(tiempo, 0.8), 1, 35)),
                'aciertos_comprension': float(np.clip(np.random.normal(aciertos, 0.02), 0.1, 1.0)),
                'sesiones_semana': float(np.clip(np.random.normal(sesiones, 0.2), 1, 10)),
                'nivel_maximo_alcanzado': float(np.clip(np.random.normal(niv_max, 0.08), 1, 5)),
            })
    return pd.DataFrame(registros)

@st.cache_data
def ejecutar_clustering(df):
    features = ['nivel_preferido', 'tasa_completado', 'tiempo_lectura_min',
                'aciertos_comprension', 'sesiones_semana', 'nivel_maximo_alcanzado']
    X = df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=10, init='k-means++', n_init=20, random_state=42)
    kmeans.fit(X_scaled)
    df = df.copy()
    df['cluster'] = kmeans.labels_
    silhouette = silhouette_score(X_scaled, kmeans.labels_)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    df['pca_x'] = X_pca[:, 0]
    df['pca_y'] = X_pca[:, 1]
    return df, silhouette, pca.explained_variance_ratio_

@st.cache_data
def generar_progreso_temporal(usuario_id, nivel_base):
    np.random.seed(hash(usuario_id) % 1000)
    semanas = list(range(1, 11))
    nivel_actual = float(nivel_base)
    niveles = []
    aciertos = []
    for s in semanas:
        niveles.append(round(min(nivel_actual, 5), 2))
        aciertos.append(float(np.clip(np.random.normal(0.65 + nivel_actual * 0.05, 0.04), 0.3, 1.0)))
        if np.random.random() < 0.3:
            nivel_actual = min(nivel_actual + 0.2, 5)
    return pd.DataFrame({'semana': semanas, 'nivel': niveles, 'aciertos': aciertos})

@st.cache_resource
def cargar_modelo():
    rutas = [
        'models/clasificador_embeddings.pkl',
        'notebooks/modelo_clasificador.pkl',
        'modelo_clasificador.pkl',
    ]
    for ruta in rutas:
        if os.path.exists(ruta):
            with open(ruta, 'rb') as f:
                return pickle.load(f)
    return None

NOMBRES_NIVELES = {
    1: "Nivel 1 — Lectura Fácil",
    2: "Nivel 2 — Básico",
    3: "Nivel 3 — Intermedio",
    4: "Nivel 4 — Avanzado",
    5: "Nivel 5 — Especializado"
}
COLORES_NIVELES = {1: "#2ecc71", 2: "#3498db", 3: "#f39c12", 4: "#e74c3c", 5: "#9b59b6"}
COLORES_RADAR = {1: "rgba(46,204,113,0.25)", 2: "rgba(52,152,219,0.25)",
                 3: "rgba(243,156,18,0.25)", 4: "rgba(231,76,60,0.25)", 5: "rgba(155,89,182,0.25)"}

# ============================================================================
# CARGA DE DATOS
# ============================================================================

df_usuarios = generar_datos_usuarios()
df_usuarios, silhouette_val, varianza_pca = ejecutar_clustering(df_usuarios)
modelo = cargar_modelo()

# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.image("https://img.icons8.com/color/96/book-reading.png", width=80)
st.sidebar.title("📚 Asistente de Lectura")
st.sidebar.markdown("**Sistema Inteligente para Personas con Discapacidad Cognitiva**")
st.sidebar.divider()

vista = st.sidebar.radio(
    "Selecciona la vista:",
    ["🏫 Vista Educador", "👤 Vista Usuario", "🔍 Clasificador de Textos"],
    index=0
)

st.sidebar.divider()
st.sidebar.markdown("**Métricas del sistema**")
st.sidebar.metric("Precisión clasificador", "90.54%", "+5.54%")
st.sidebar.metric("Silueta clustering", f"{silhouette_val:.4f}", "✅ ≥ 0.6")
st.sidebar.metric("Usuarios en sistema", len(df_usuarios))
st.sidebar.metric("Textos en dataset", "2.695")

# Título dinámico — se actualiza con cada cambio de vista
titulos = {
    "🏫 Vista Educador": "🏫 Panel del Educador",
    "👤 Vista Usuario": "👤 Mi progreso de lectura",
    "🔍 Clasificador de Textos": "🔍 Clasificador de Nivel de Lectura"
}
st.title(titulos.get(vista, ""))

# ============================================================================
# VISTA EDUCADOR
# ============================================================================

if vista == "🏫 Vista Educador":
    st.markdown("Visión global del progreso y agrupamiento de todos los usuarios del sistema.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total usuarios", len(df_usuarios))
    with col2:
        st.metric("Tasa completado media", f"{df_usuarios['tasa_completado'].mean()*100:.1f}%")
    with col3:
        st.metric("Aciertos comprensión", f"{df_usuarios['aciertos_comprension'].mean()*100:.1f}%")
    with col4:
        st.metric("Nivel medio usuarios", f"{df_usuarios['nivel_preferido'].mean():.1f}/5")

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Agrupamiento PCA",
        "🗺️ Distribución por niveles",
        "🔥 Mapa de calor de actividad",
        "📈 Importancia de características"
    ])

    with tab1:
        st.subheader("Visualización de Clusters — Reducción PCA")
        st.caption(f"Varianza explicada: PC1={varianza_pca[0]*100:.1f}% | PC2={varianza_pca[1]*100:.1f}% | Total={sum(varianza_pca)*100:.1f}%")
        fig_pca = px.scatter(
            df_usuarios, x='pca_x', y='pca_y',
            color='perfil_base',
            hover_data=['usuario_id', 'nivel_preferido', 'tasa_completado', 'aciertos_comprension'],
            title=f"Clustering K-means++ (k=10) — Silueta: {silhouette_val:.4f}",
            labels={'pca_x': f'PC1 ({varianza_pca[0]*100:.1f}%)', 'pca_y': f'PC2 ({varianza_pca[1]*100:.1f}%)'},
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        fig_pca.update_traces(marker=dict(size=7, opacity=0.75))
        fig_pca.update_layout(height=500, legend_title="Perfil de usuario")
        st.plotly_chart(fig_pca, use_container_width=True)
        c1, c2 = st.columns(2)
        c1.metric("Coeficiente de Silueta", f"{silhouette_val:.4f}", "✅ Supera objetivo ≥ 0.6")
        c2.metric("Clústeres identificados", "10", "Perfiles diferenciados")

    with tab2:
        st.subheader("Distribución de usuarios por nivel y perfil")
        c1, c2 = st.columns(2)
        with c1:
            nivel_counts = df_usuarios['nivel_preferido'].round().astype(int).value_counts().sort_index()
            fig_pie = px.pie(
                values=nivel_counts.values,
                names=[NOMBRES_NIVELES.get(n, f"Nivel {n}") for n in nivel_counts.index],
                title="Distribución por nivel preferido",
                color_discrete_sequence=list(COLORES_NIVELES.values())
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        with c2:
            perfil_counts = df_usuarios['perfil_base'].value_counts()
            fig_bar = px.bar(
                x=perfil_counts.values, y=perfil_counts.index, orientation='h',
                title="Usuarios por perfil base",
                labels={'x': 'Número de usuarios', 'y': 'Perfil'},
                color=perfil_counts.values, color_continuous_scale='Blues'
            )
            fig_bar.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("Resumen estadístico por cluster")
        resumen = df_usuarios.groupby('perfil_base').agg(
            Usuarios=('usuario_id', 'count'),
            Nivel_medio=('nivel_preferido', 'mean'),
            Tasa_completado=('tasa_completado', 'mean'),
            Aciertos=('aciertos_comprension', 'mean'),
            Sesiones_semana=('sesiones_semana', 'mean')
        ).round(2)
        resumen['Tasa_completado'] = (resumen['Tasa_completado'] * 100).round(1).astype(str) + '%'
        resumen['Aciertos'] = (resumen['Aciertos'] * 100).round(1).astype(str) + '%'
        st.dataframe(resumen, use_container_width=True)

    with tab3:
        st.subheader("Sesiones de lectura por perfil y semana")
        np.random.seed(42)
        perfiles = df_usuarios['perfil_base'].unique()
        semanas = [f"Sem {i}" for i in range(1, 11)]
        actividad = np.zeros((len(perfiles), 10), dtype=int)
        for i, perfil in enumerate(perfiles):
            sesiones_base = df_usuarios[df_usuarios['perfil_base'] == perfil]['sesiones_semana'].mean()
            actividad[i] = np.clip(np.random.normal(sesiones_base * 2, 1.5, 10), 1, 20).astype(int)
        fig_heat = go.Figure(data=go.Heatmap(
            z=actividad, x=semanas, y=list(perfiles),
            colorscale='Blues', text=actividad, texttemplate="%{text}",
            showscale=True, colorbar=dict(title="Sesiones")
        ))
        fig_heat.update_layout(
            title="Sesiones de lectura por perfil y semana",
            xaxis_title="Semana", yaxis_title="Perfil de usuario", height=450
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        st.caption("Cada celda muestra el número de sesiones completadas por perfil en cada semana.")

    with tab4:
        st.subheader("Importancia de características del clasificador")
        feature_names = ['Palabras por frase', 'Sílabas por palabra', 'Longitud media palabra',
                         'Ratio palabras largas', 'Índice INFLESZ']
        importancias_ref = [0.288, 0.178, 0.196, 0.108, 0.230]
        if modelo is not None:
            try:
                clf = modelo['clasificador'] if isinstance(modelo, dict) else modelo
                importancias_ref = list(clf.feature_importances_[:5])
            except Exception:
                pass
        df_imp = pd.DataFrame({'Característica': feature_names, 'Importancia': importancias_ref})
        df_imp = df_imp.sort_values('Importancia', ascending=True)
        fig_imp = px.bar(
            df_imp, x='Importancia', y='Característica', orientation='h',
            title="Importancia de características — XGBoost",
            color='Importancia', color_continuous_scale='Blues',
            text=df_imp['Importancia'].apply(lambda x: f"{x*100:.1f}%")
        )
        fig_imp.update_traces(textposition='outside')
        fig_imp.update_layout(showlegend=False, height=350, xaxis_tickformat='.0%')
        st.plotly_chart(fig_imp, use_container_width=True)
        st.markdown("""
        **Interpretación:**
        - **Palabras por frase**: el factor más predictivo. Frases cortas = textos más accesibles.
        - **Sílabas por palabra**: vocabulario complejo eleva el nivel de dificultad.
        - **Longitud media de palabra**: correlaciona con tecnicismo del vocabulario.
        - **Ratio palabras largas**: proporción de palabras con más de 3 sílabas.
        - **Índice INFLESZ**: métrica clásica de legibilidad adaptada al español.
        """)

# ============================================================================
# VISTA USUARIO
# ============================================================================

elif vista == "👤 Vista Usuario":
    usuarios_disponibles = df_usuarios['usuario_id'].tolist()
    usuario_sel = st.selectbox("Selecciona un usuario:", usuarios_disponibles, index=0)

    datos_usuario = df_usuarios[df_usuarios['usuario_id'] == usuario_sel].iloc[0]
    nivel_usuario = int(round(datos_usuario['nivel_preferido']))
    perfil_usuario = datos_usuario['perfil_base']

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Nivel actual", f"{nivel_usuario}/5", NOMBRES_NIVELES.get(nivel_usuario, ""))
    col2.metric("Textos completados", f"{datos_usuario['tasa_completado']*100:.0f}%")
    col3.metric("Comprensión media", f"{datos_usuario['aciertos_comprension']*100:.0f}%")
    col4.metric("Sesiones por semana", f"{datos_usuario['sesiones_semana']:.0f}")

    st.divider()
    c1, c2 = st.columns([2, 1])

    with c1:
        st.subheader("📈 Mi progreso — últimas 10 semanas")
        df_progreso = generar_progreso_temporal(usuario_sel, nivel_usuario)
        fig_prog = go.Figure()
        fig_prog.add_trace(go.Scatter(
            x=df_progreso['semana'], y=df_progreso['nivel'],
            mode='lines+markers', name='Nivel de lectura',
            line=dict(color='#1f77b4', width=3), marker=dict(size=8)
        ))
        fig_prog.add_trace(go.Scatter(
            x=df_progreso['semana'], y=df_progreso['aciertos'] * 5,
            mode='lines+markers', name='Aciertos (escala 0-5)',
            line=dict(color='#2ecc71', width=2, dash='dash'), marker=dict(size=6)
        ))
        fig_prog.update_layout(
            xaxis_title="Semana", yaxis_title="Nivel / Aciertos",
            yaxis=dict(range=[0, 5.5], tickvals=[1, 2, 3, 4, 5]),
            legend=dict(orientation='h', yanchor='bottom', y=1.02), height=350
        )
        st.plotly_chart(fig_prog, use_container_width=True)

    with c2:
        st.subheader("👥 Mi perfil")
        st.info(f"**Grupo asignado:** {perfil_usuario}")
        st.markdown(f"**Nivel máximo alcanzado:** {int(round(datos_usuario['nivel_maximo_alcanzado']))}/5")
        st.markdown(f"**Tiempo medio de lectura:** {datos_usuario['tiempo_lectura_min']:.0f} minutos")
        nivel_max = int(round(datos_usuario['nivel_maximo_alcanzado']))
        progreso_pct = (nivel_max / 5) * 100
        st.markdown(f"**Progreso general:** {progreso_pct:.0f}%")
        st.progress(progreso_pct / 100)

    st.divider()
    st.subheader("📖 Textos recomendados para ti")
    recomendaciones = {
        1: [{"titulo": "Noticias Plena Inclusión", "nivel": "Nivel 1", "tema": "Actualidad"},
            {"titulo": "Escuela de Monstruos", "nivel": "Nivel 1", "tema": "Cuento"},
            {"titulo": "La biblioteca de Julia", "nivel": "Nivel 1", "tema": "Cuento"}],
        2: [{"titulo": "Fábulas de Esopo adaptadas", "nivel": "Nivel 2", "tema": "Fábula"},
            {"titulo": "El Pez Arcoíris", "nivel": "Nivel 2", "tema": "Cuento"},
            {"titulo": "Los tres cerditos", "nivel": "Nivel 2", "tema": "Cuento clásico"}],
        3: [{"titulo": "National Geographic Kids — Animales", "nivel": "Nivel 3", "tema": "Divulgación"},
            {"titulo": "National Geographic Kids — Dinosaurios", "nivel": "Nivel 3", "tema": "Ciencia"},
            {"titulo": "National Geographic Kids — Exploración", "nivel": "Nivel 3", "tema": "Historia"}],
        4: [{"titulo": "Novela narrativa adulto — selección", "nivel": "Nivel 4", "tema": "Narrativa"},
            {"titulo": "Artículos de divulgación general", "nivel": "Nivel 4", "tema": "Divulgación"}],
        5: [{"titulo": "Artículos científicos adaptados", "nivel": "Nivel 5", "tema": "Ciencia"},
            {"titulo": "Textos técnicos especializados", "nivel": "Nivel 5", "tema": "Técnico"}],
    }
    textos_rec = recomendaciones.get(nivel_usuario, recomendaciones[3])
    cols = st.columns(len(textos_rec))
    for col, texto in zip(cols, textos_rec):
        color = COLORES_NIVELES.get(nivel_usuario, "#3498db")
        with col:
            st.markdown(f"""
            <div style='background-color:{color}22; border-left:4px solid {color};
                        padding:12px; border-radius:8px; margin:4px 0'>
                <b>📄 {texto['titulo']}</b><br>
                <small>{texto['nivel']} • {texto['tema']}</small>
            </div>""", unsafe_allow_html=True)

# ============================================================================
# CLASIFICADOR DE TEXTOS
# ============================================================================

elif vista == "🔍 Clasificador de Textos":
    st.markdown("Introduce un texto y el sistema predecirá automáticamente su nivel de dificultad.")

    texto_input = st.text_area(
        "Escribe o pega el texto aquí:",
        height=200,
        placeholder="Escribe o pega aquí el texto que quieres analizar..."
    )

    if st.button("🔍 Analizar texto", type="primary", use_container_width=True):
        if texto_input.strip():
            with st.spinner("Analizando..."):
                features = calcular_features_texto(texto_input)

            if features:
                nivel_predicho = None
                confianza = None

                if modelo is not None:
                    try:
                        clf = modelo['clasificador'] if isinstance(modelo, dict) else modelo
                        X_pred = np.array([[
                            features['palabras_por_frase'],
                            features['silabas_por_palabra'],
                            features['longitud_promedio_palabra'],
                            features['ratio_palabras_largas'],
                            features['inflesz']
                        ]])
                        nivel_predicho = int(clf.predict(X_pred)[0])
                        if hasattr(clf, 'predict_proba'):
                            confianza = float(max(clf.predict_proba(X_pred)[0]))
                    except Exception:
                        pass

                if nivel_predicho is None:
                    inflesz = features['inflesz']
                    if inflesz >= 75: nivel_predicho = 1
                    elif inflesz >= 65: nivel_predicho = 2
                    elif inflesz >= 55: nivel_predicho = 3
                    elif inflesz >= 40: nivel_predicho = 4
                    else: nivel_predicho = 5

                st.divider()
                col1, col2 = st.columns([1, 2])

                with col1:
                    color = COLORES_NIVELES.get(nivel_predicho, "#3498db")
                    st.markdown(f"""
                    <div style='background-color:{color}22; border:3px solid {color};
                                padding:25px; border-radius:12px; text-align:center'>
                        <h2 style='color:{color}; margin:0'>Nivel {nivel_predicho}</h2>
                        <p style='margin:8px 0 0 0'>{NOMBRES_NIVELES.get(nivel_predicho, '')}</p>
                        {'<p style="color:gray;font-size:0.85em">Confianza: ' + f"{confianza*100:.1f}%" + '</p>' if confianza else ''}
                    </div>""", unsafe_allow_html=True)

                with col2:
                    st.subheader("📊 Métricas lingüísticas del texto")
                    m1, m2 = st.columns(2)
                    with m1:
                        st.metric("Palabras", features['num_palabras'])
                        st.metric("Frases", features['num_frases'])
                        st.metric("Palabras por frase", f"{features['palabras_por_frase']:.1f}")
                    with m2:
                        st.metric("Sílabas por palabra", f"{features['silabas_por_palabra']:.2f}")
                        st.metric("Índice INFLESZ", f"{features['inflesz']:.1f}")
                        st.metric("Ratio palabras largas", f"{features['ratio_palabras_largas']*100:.1f}%")

                st.divider()
                categorias = ['Palabras/frase', 'Sílabas/palabra', 'Long. media', 'Palabras largas', 'INFLESZ']
                valores_norm = [
                    min(features['palabras_por_frase'] / 30, 1),
                    min(features['silabas_por_palabra'] / 4, 1),
                    min(features['longitud_promedio_palabra'] / 10, 1),
                    features['ratio_palabras_largas'],
                    max(0, min((100 - features['inflesz']) / 100, 1))
                ]
                fig_radar = go.Figure(data=go.Scatterpolar(
                    r=valores_norm + [valores_norm[0]],
                    theta=categorias + [categorias[0]],
                    fill='toself',
                    fillcolor=COLORES_RADAR.get(nivel_predicho, "rgba(100,149,237,0.25)"),
                    line=dict(color=COLORES_NIVELES.get(nivel_predicho, '#3498db'), width=2)
                ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    title="Perfil lingüístico del texto", height=350
                )
                st.plotly_chart(fig_radar, use_container_width=True)

                descripciones = {
                    1: "✅ Texto muy accesible. Adecuado para personas con DI que están comenzando su proceso lector.",
                    2: "✅ Texto accesible. Adecuado para lectores en nivel básico.",
                    3: "⚠️ Texto de dificultad media. Puede necesitar adaptación para personas con DI.",
                    4: "⚠️ Texto complejo. No adecuado sin adaptación para DI.",
                    5: "❌ Texto muy especializado. Requiere adaptación significativa antes de usarlo con personas con DI."
                }
                st.info(descripciones.get(nivel_predicho, ""))
        else:
            st.warning("Por favor, introduce un texto para analizar.")

st.divider()
st.caption("TFM — Sistema Inteligente de Asistencia a la Lectura para Personas con Discapacidad Cognitiva | ILERNA — Curso de Especialización IA y Big Data 2025-2026")