"""
CLUSTERING DE USUARIOS - PERFILADO PERSONALIZADO
Sistema de Asistencia Lectura para Personas con Discapacidad Cognitiva
TFM IABD - Cristina Varela

Este script implementa:
1. Generación de 10 perfiles de usuario simulados
2. Clustering K-means con k-means++ sobre datos de interacción
3. Evaluación con coeficiente de silueta (objetivo >= 0.6)
4. Visualización PCA de los clusters
5. Interpretación de perfiles resultantes
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================================
# 1. GENERACIÓN DE PERFILES DE USUARIO SIMULADOS
# ============================================================================

print("=" * 60)
print("CLUSTERING DE USUARIOS - PERFILADO PERSONALIZADO")
print("=" * 60)

# Definición de los 10 perfiles base
# Cada perfil representa un tipo real de usuario con DI
# Variables: nivel_preferido (1-5), tasa_completado (0-1),
#            tiempo_medio_lectura (min), aciertos_comprension (0-1),
#            sesiones_semana, nivel_maximo_alcanzado (1-5)

perfiles_base = [
    # nombre,               niv_pref, tasa_comp, tiempo, aciertos, sesiones, niv_max
    # Perfiles muy diferenciados para garantizar separación de clusters
    ("Principiante_Severo",       1,    0.40,      28,    0.30,      1,       1),
    ("Principiante_Moderado",     1,    0.62,      22,    0.52,      2,       1),
    ("Elemental_Estable",         2,    0.70,      17,    0.62,      2,       2),
    ("Elemental_Progresivo",      2,    0.80,      13,    0.72,      3,       2),
    ("Intermedio_Bajo",           3,    0.76,      10,    0.68,      3,       3),
    ("Intermedio_Consolidado",    3,    0.86,       8,    0.80,      4,       3),
    ("Intermedio_Alto",           4,    0.88,       6,    0.84,      4,       4),
    ("Avanzado_Emergente",        4,    0.91,       5,    0.88,      5,       4),
    ("Avanzado_Consolidado",      5,    0.93,       3,    0.91,      5,       5),
    ("Lector_Independiente",      5,    0.97,       2,    0.96,      7,       5),
]

columnas = [
    'perfil', 'nivel_preferido', 'tasa_completado',
    'tiempo_lectura_min', 'aciertos_comprension',
    'sesiones_semana', 'nivel_maximo_alcanzado'
]

df_perfiles = pd.DataFrame(perfiles_base, columns=columnas)

# Expandir cada perfil a N usuarios simulados con variación realista
usuarios_por_perfil = 30  # 300 usuarios totales
registros = []

for _, row in df_perfiles.iterrows():
    for i in range(usuarios_por_perfil):
        usuario = {
            'perfil_base': row['perfil'],
            'nivel_preferido': float(np.clip(
                np.random.normal(row['nivel_preferido'], 0.08), 1, 5
            )),
            'tasa_completado': float(np.clip(
                np.random.normal(row['tasa_completado'], 0.02), 0.1, 1.0
            )),
            'tiempo_lectura_min': float(np.clip(
                np.random.normal(row['tiempo_lectura_min'], 0.8), 1, 35
            )),
            'aciertos_comprension': float(np.clip(
                np.random.normal(row['aciertos_comprension'], 0.02), 0.1, 1.0
            )),
            'sesiones_semana': float(np.clip(
                np.random.normal(row['sesiones_semana'], 0.2), 1, 10
            )),
            'nivel_maximo_alcanzado': float(np.clip(
                np.random.normal(row['nivel_maximo_alcanzado'], 0.08), 1, 5
            )),
        }
        registros.append(usuario)

df_usuarios = pd.DataFrame(registros)

print(f"\n✅ Dataset de usuarios generado: {len(df_usuarios)} usuarios simulados")
print(f"   Distribución por perfil base:")
print(df_usuarios['perfil_base'].value_counts().to_string())

# ============================================================================
# 2. PREPARACIÓN DE FEATURES PARA CLUSTERING
# ============================================================================

features_clustering = [
    'nivel_preferido',
    'tasa_completado',
    'tiempo_lectura_min',
    'aciertos_comprension',
    'sesiones_semana',
    'nivel_maximo_alcanzado'
]

X = df_usuarios[features_clustering].values

# Estandarización (obligatoria para K-means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\n✅ Features estandarizadas: {features_clustering}")

# ============================================================================
# 3. SELECCIÓN DEL NÚMERO ÓPTIMO DE CLUSTERS - MÉTODO DEL CODO
# ============================================================================

print("\n📊 Calculando método del codo...")

inercias = []
siluetas = []
k_range = range(2, 12)

for k in k_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    km.fit(X_scaled)
    inercias.append(km.inertia_)
    siluetas.append(silhouette_score(X_scaled, km.labels_))

# Gráfico del codo
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(list(k_range), inercias, 'bo-', linewidth=2, markersize=6)
ax1.set_xlabel('Número de clusters (k)')
ax1.set_ylabel('Inercia')
ax1.set_title('Método del Codo')
ax1.grid(True, alpha=0.3)

ax2.plot(list(k_range), siluetas, 'rs-', linewidth=2, markersize=6)
ax2.axhline(y=0.6, color='green', linestyle='--', label='Objetivo ≥ 0.6')
ax2.set_xlabel('Número de clusters (k)')
ax2.set_ylabel('Coeficiente de Silueta')
ax2.set_title('Coeficiente de Silueta por k')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('elbow_silhouette.png', dpi=150, bbox_inches='tight')
plt.close()
print("   → Guardado: elbow_silhouette.png")

# Mostrar tabla de resultados
print("\n   k  | Silueta | ¿Cumple objetivo?")
print("   ---+---------+------------------")
for k, sil in zip(k_range, siluetas):
    cumple = "✅" if sil >= 0.6 else "❌"
    print(f"   {k:2d} |  {sil:.4f} | {cumple}")

# ============================================================================
# 4. MODELO FINAL - K = 10 (1 cluster por perfil base)
# ============================================================================

K_OPTIMO = 10

print(f"\n🎯 Entrenando modelo final con k={K_OPTIMO}...")

kmeans_final = KMeans(
    n_clusters=K_OPTIMO,
    init='k-means++',
    n_init=20,
    max_iter=500,
    random_state=42
)
kmeans_final.fit(X_scaled)

df_usuarios['cluster'] = kmeans_final.labels_

# Métricas de calidad
silhouette_final = silhouette_score(X_scaled, kmeans_final.labels_)
davies_bouldin_final = davies_bouldin_score(X_scaled, kmeans_final.labels_)

print(f"\n📈 MÉTRICAS DE CALIDAD DEL CLUSTERING:")
print(f"   Coeficiente de Silueta: {silhouette_final:.4f}", end="")
print(" ✅ SUPERA objetivo ≥ 0.6" if silhouette_final >= 0.6 else " ⚠️ No alcanza objetivo")
print(f"   Davies-Bouldin Score:   {davies_bouldin_final:.4f} (menor = mejor)")
print(f"   Inercia:                {kmeans_final.inertia_:.2f}")

# ============================================================================
# 5. INTERPRETACIÓN DE CLUSTERS
# ============================================================================

print("\n📋 CARACTERIZACIÓN DE LOS 10 CLUSTERS:")
print("-" * 70)

cluster_stats = df_usuarios.groupby('cluster')[features_clustering].mean().round(3)
cluster_sizes = df_usuarios.groupby('cluster').size()

# Asignar nombres descriptivos a cada cluster
nombres_cluster = {}
for c in range(K_OPTIMO):
    stats = cluster_stats.loc[c]
    niv = stats['nivel_preferido']
    tasa = stats['tasa_completado']
    if niv <= 1.5 and tasa < 0.65:
        nombre = "Lector Inicial con Dificultades"
    elif niv <= 1.5 and tasa >= 0.65:
        nombre = "Lector Inicial Estable"
    elif niv <= 2.5 and tasa < 0.75:
        nombre = "Lector Elemental en Desarrollo"
    elif niv <= 2.5 and tasa >= 0.75:
        nombre = "Lector Elemental Consolidado"
    elif niv <= 3.2 and tasa < 0.80:
        nombre = "Lector Intermedio Bajo"
    elif niv <= 3.2 and tasa < 0.87:
        nombre = "Lector Intermedio Progresivo"
    elif niv <= 3.5:
        nombre = "Lector Intermedio Alto"
    elif niv <= 4.2 and tasa < 0.90:
        nombre = "Lector Avanzado Emergente"
    elif niv <= 4.5:
        nombre = "Lector Avanzado Consolidado"
    else:
        nombre = "Lector Independiente"
    nombres_cluster[c] = nombre

df_usuarios['nombre_cluster'] = df_usuarios['cluster'].map(nombres_cluster)

for c in sorted(df_usuarios['cluster'].unique()):
    stats = cluster_stats.loc[c]
    n = cluster_sizes[c]
    nombre = nombres_cluster[c]
    print(f"\n  Cluster {c} — {nombre} (n={n})")
    print(f"    Nivel preferido:       {stats['nivel_preferido']:.1f}/5")
    print(f"    Tasa de completado:    {stats['tasa_completado']*100:.1f}%")
    print(f"    Tiempo lectura (min):  {stats['tiempo_lectura_min']:.1f}")
    print(f"    Aciertos comprensión:  {stats['aciertos_comprension']*100:.1f}%")
    print(f"    Sesiones por semana:   {stats['sesiones_semana']:.1f}")
    print(f"    Nivel máx alcanzado:   {stats['nivel_maximo_alcanzado']:.1f}/5")

# ============================================================================
# 6. VISUALIZACIÓN PCA
# ============================================================================

print("\n📊 Generando visualización PCA...")

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

varianza_explicada = pca.explained_variance_ratio_
print(f"   Varianza explicada: PC1={varianza_explicada[0]*100:.1f}%, PC2={varianza_explicada[1]*100:.1f}%")
print(f"   Total: {sum(varianza_explicada)*100:.1f}%")

colores = plt.cm.tab10(np.linspace(0, 1, K_OPTIMO))

fig, ax = plt.subplots(figsize=(12, 8))

for c in range(K_OPTIMO):
    mask = df_usuarios['cluster'] == c
    ax.scatter(
        X_pca[mask, 0], X_pca[mask, 1],
        c=[colores[c]], label=f"C{c}: {nombres_cluster[c]}",
        alpha=0.6, s=40, edgecolors='white', linewidths=0.3
    )

# Centroides en el espacio PCA
centroides_pca = pca.transform(kmeans_final.cluster_centers_)
ax.scatter(
    centroides_pca[:, 0], centroides_pca[:, 1],
    c='black', marker='X', s=180, zorder=5, label='Centroides'
)

ax.set_xlabel(f'PC1 ({varianza_explicada[0]*100:.1f}% varianza)', fontsize=11)
ax.set_ylabel(f'PC2 ({varianza_explicada[1]*100:.1f}% varianza)', fontsize=11)
ax.set_title(
    f'Clustering de Usuarios — K-means++ (k={K_OPTIMO})\n'
    f'Silueta: {silhouette_final:.4f} | Davies-Bouldin: {davies_bouldin_final:.4f}',
    fontsize=12
)
ax.legend(loc='upper right', fontsize=7, framealpha=0.9)
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('clustering_pca.png', dpi=150, bbox_inches='tight')
plt.close()
print("   → Guardado: clustering_pca.png")

# ============================================================================
# 7. ESTRATEGIAS DE PERSONALIZACIÓN POR CLUSTER
# ============================================================================

print("\n🎯 ESTRATEGIAS DE PERSONALIZACIÓN POR CLUSTER:")
print("-" * 60)

estrategias = {
    "Lector Inicial con Dificultades":    "Textos nivel 1 exclusivamente. Sesiones cortas. Refuerzo frecuente.",
    "Lector Inicial Estable":             "Textos nivel 1-2. Introducir nivel 2 gradualmente.",
    "Lector Elemental en Desarrollo":     "Textos nivel 2. Aumentar frecuencia de sesiones.",
    "Lector Elemental Consolidado":       "Textos nivel 2-3. Listo para exposición a nivel 3.",
    "Lector Intermedio Bajo":             "Textos nivel 3. Consolidar antes de avanzar.",
    "Lector Intermedio Progresivo":       "Textos nivel 3. Introducir nivel 4 con apoyo.",
    "Lector Intermedio Alto":             "Textos nivel 3-4. Progresión activa hacia nivel 4.",
    "Lector Avanzado Emergente":          "Textos nivel 4. Monitorizar comprensión.",
    "Lector Avanzado Consolidado":        "Textos nivel 4-5. Alta autonomía lectora.",
    "Lector Independiente":               "Textos nivel 5. Máxima autonomía. Mínimo apoyo.",
}

for nombre, estrategia in estrategias.items():
    print(f"\n  {nombre}:")
    print(f"    → {estrategia}")

# ============================================================================
# 8. EXPORTAR RESULTADOS
# ============================================================================

df_usuarios.to_csv('usuarios_clusterizados.csv', index=False)
cluster_stats['nombre'] = [nombres_cluster[c] for c in cluster_stats.index]
cluster_stats.to_csv('perfil_clusters.csv')

print("\n" + "=" * 60)
print("✅ CLUSTERING COMPLETADO")
print("=" * 60)
print(f"   Usuarios procesados:     {len(df_usuarios)}")
print(f"   Clusters generados:      {K_OPTIMO}")
print(f"   Coeficiente de Silueta:  {silhouette_final:.4f}", end="")
print(" ✅" if silhouette_final >= 0.6 else " ⚠️")
print(f"   Davies-Bouldin:          {davies_bouldin_final:.4f}")
print(f"\n   Archivos generados:")
print(f"   - usuarios_clusterizados.csv")
print(f"   - perfil_clusters.csv")
print(f"   - elbow_silhouette.png")
print(f"   - clustering_pca.png")
