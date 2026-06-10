"""
CLASIFICADOR BÁSICO DE DIFICULTAD LECTORA
Sistema de Asistencia Lectura para Personas con Discapacidad Cognitiva
TFM IABD - Cristina Varela

Este script implementa un clasificador básico que:
1. Carga textos etiquetados con niveles de dificultad (1-5)
2. Calcula métricas de legibilidad (INFLESZ, features lingüísticas)
3. Entrena un clasificador RandomForest (baseline antes de embeddings)
4. Evalúa el rendimiento con métricas estándar
5. Predice el nivel de nuevos textos

NOTA: Este es el modelo baseline (Iteración 0, 29,93% accuracy).
El modelo final usa embeddings semánticos (02_procesamiento_embeddings.ipynb).
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import re

# ============================================================================
# FUNCIONES DE EXTRACCIÓN DE FEATURES
# ============================================================================

def contar_silabas(palabra):
    """
    Cuenta sílabas en español de forma aproximada.
    
    Método: cada grupo de vocales contiguas cuenta como una sílaba.
    La 'h' se elimina antes del conteo porque es muda en español.
    
    Args:
        palabra (str): palabra en español
        
    Returns:
        int: número de sílabas (mínimo 1)
        
    Ejemplo:
        contar_silabas("casa") → 2
        contar_silabas("inteligencia") → 5
    """
    palabra = palabra.lower()
    # Eliminar h muda (no forma sílaba en español)
    palabra = palabra.replace('h', '')
    # Conjunto de vocales españolas, incluyendo acentuadas y diéresis
    vocales = 'aeiouáéíóúü'
    silabas = 0
    en_vocal = False  # Bandera para detectar grupos de vocales
    
    for char in palabra:
        if char in vocales:
            if not en_vocal:
                # Primera vocal del grupo: nueva sílaba
                silabas += 1
                en_vocal = True
        else:
            # Consonante: reset del grupo
            en_vocal = False
    
    return max(1, silabas)  # Toda palabra tiene al menos 1 sílaba


def calcular_metricas_texto(texto):
    """
    Calcula métricas de legibilidad del texto en español.
    
    Estas métricas superficiales forman el baseline del clasificador.
    En el modelo final se combinan con embeddings semánticos de 384 dims.
    
    Args:
        texto (str): texto en español a analizar
        
    Returns:
        dict con las siguientes métricas:
            - num_palabras (int): número total de palabras
            - num_frases (int): número de frases (separadas por . ! ?)
            - palabras_por_frase (float): promedio de palabras por frase
            - silabas_por_palabra (float): promedio de sílabas por palabra
            - longitud_promedio_palabra (float): promedio de caracteres por palabra
            - ratio_palabras_largas (float): proporción de palabras con >3 sílabas
            - inflesz (float): índice INFLESZ de legibilidad para español
    """
    texto = texto.strip()
    
    # Segmentar en frases por puntuación final
    frases = re.split(r'[.!?]+', texto)
    frases = [f.strip() for f in frases if f.strip()]
    num_frases = max(len(frases), 1)  # Evitar división por cero
    
    # Extraer palabras (secuencias alfanuméricas)
    palabras = re.findall(r'\b\w+\b', texto.lower())
    num_palabras = len(palabras)
    
    if num_palabras == 0:
        # Texto vacío o sin palabras válidas: devolver ceros
        return {
            'num_palabras': 0,
            'num_frases': 0,
            'palabras_por_frase': 0,
            'silabas_por_palabra': 0,
            'longitud_promedio_palabra': 0,
            'ratio_palabras_largas': 0,
            'inflesz': 0
        }
    
    # Promedio de palabras por frase (indicador de complejidad sintáctica)
    palabras_por_frase = num_palabras / num_frases
    
    # Promedio de sílabas por palabra (indicador de complejidad léxica)
    total_silabas = sum(contar_silabas(palabra) for palabra in palabras)
    silabas_por_palabra = total_silabas / num_palabras
    
    # Longitud promedio de palabra en caracteres
    longitud_promedio_palabra = sum(len(p) for p in palabras) / num_palabras
    
    # Proporción de palabras largas (>3 sílabas, indicador de vocabulario técnico)
    palabras_largas = sum(1 for p in palabras if contar_silabas(p) > 3)
    ratio_palabras_largas = palabras_largas / num_palabras
    
    # Índice INFLESZ: adaptación española del índice de Flesch
    # Fórmula: 206.84 - (60 × sílabas/palabra) - (1.02 × palabras/frase)
    # Valores: >80 muy fácil, 60-80 normal, 40-60 difícil, <40 muy difícil
    inflesz = 206.84 - (60 * silabas_por_palabra) - (1.02 * palabras_por_frase)
    
    return {
        'num_palabras': num_palabras,
        'num_frases': num_frases,
        'palabras_por_frase': palabras_por_frase,
        'silabas_por_palabra': silabas_por_palabra,
        'longitud_promedio_palabra': longitud_promedio_palabra,
        'ratio_palabras_largas': ratio_palabras_largas,
        'inflesz': inflesz
    }


def procesar_dataset(df):
    """
    Procesa el dataset calculando el vector de features para cada texto.
    
    Solo se usan las features relativas (ratios y promedios), no los
    conteos absolutos como num_palabras, ya que varían con la longitud
    del texto y añadirían ruido al clasificador.
    
    Args:
        df (pd.DataFrame): DataFrame con columnas 'texto' y 'nivel'
        
    Returns:
        tuple:
            X (np.ndarray): matriz de features (n_textos × 5)
            y (np.ndarray): etiquetas de nivel (1-5)
            feature_names (list): nombres de las 5 features
    """
    features_list = []
    
    for texto in df['texto']:
        metricas = calcular_metricas_texto(texto)
        # Vector de 5 features lingüísticas (las relativas, no conteos absolutos)
        features = [
            metricas['palabras_por_frase'],
            metricas['silabas_por_palabra'],
            metricas['longitud_promedio_palabra'],
            metricas['ratio_palabras_largas'],
            metricas['inflesz']
        ]
        features_list.append(features)
    
    feature_names = [
        'palabras_por_frase',
        'silabas_por_palabra',
        'longitud_promedio_palabra',
        'ratio_palabras_largas',
        'inflesz'
    ]
    
    X = np.array(features_list)
    y = df['nivel'].values
    
    return X, y, feature_names


# ============================================================================
# MAIN: ENTRENAMIENTO Y EVALUACIÓN
# ============================================================================

def main():
    print("=" * 70)
    print("CLASIFICADOR DE DIFICULTAD LECTORA — BASELINE")
    print("TFM IABD - Cristina Varela")
    print("=" * 70)
    print()
    
    # ------------------------------------------------------------------
    # 1. CARGAR DATOS
    # Dataset CSV con columnas: texto, nivel (1-5)
    # ------------------------------------------------------------------
    print("1. Cargando datos...")
    try:
        df = pd.read_csv('../data/processed/textos_etiquetados.csv')
        print(f"   ✓ Cargados {len(df)} textos")
        print(f"   ✓ Niveles: {sorted(df['nivel'].unique())}")
        print()
    except FileNotFoundError:
        print("   ✗ ERROR: No se encuentra 'data/textos_etiquetados.csv'")
        print("   Crea el archivo CSV con columnas: texto,nivel")
        return
    
    # ------------------------------------------------------------------
    # 2. PROCESAR FEATURES
    # Calcular las 5 métricas lingüísticas para cada texto
    # ------------------------------------------------------------------
    print("2. Calculando features de legibilidad...")
    X, y, feature_names = procesar_dataset(df)
    print(f"   ✓ Calculados {X.shape[1]} features por texto")
    print(f"   ✓ Features: {', '.join(feature_names)}")
    print()
    
    # ------------------------------------------------------------------
    # 3. DIVIDIR TRAIN/TEST
    # Estratificación para mantener proporciones de clase en ambos splits
    # ------------------------------------------------------------------
    print("3. Dividiendo dataset (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   ✓ Train: {len(X_train)} textos")
    print(f"   ✓ Test: {len(X_test)} textos")
    print()
    
    # ------------------------------------------------------------------
    # 4. ENTRENAR CLASIFICADOR
    # RandomForest como baseline; en el modelo final se usa XGBoost
    # ------------------------------------------------------------------
    print("4. Entrenando clasificador RandomForest...")
    clf = RandomForestClassifier(
        n_estimators=100,   # 100 árboles de decisión
        max_depth=10,       # Profundidad máxima para evitar sobreajuste
        random_state=42     # Semilla para reproducibilidad
    )
    clf.fit(X_train, y_train)
    print("   ✓ Modelo entrenado")
    print()
    
    # ------------------------------------------------------------------
    # 5. EVALUAR
    # Accuracy esperado ~29,93% (demostración de que las métricas
    # superficiales son insuficientes → justifica el uso de embeddings)
    # ------------------------------------------------------------------
    print("5. Evaluando en test set...")
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"   ✓ Accuracy: {accuracy:.2%}")
    print()
    print("   Reporte de clasificación:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    print("   Matriz de confusión:")
    print(confusion_matrix(y_test, y_pred))
    print()
    
    # ------------------------------------------------------------------
    # 6. IMPORTANCIA DE FEATURES
    # Muestra qué métricas aportan más al modelo baseline
    # ------------------------------------------------------------------
    print("6. Importancia de features:")
    importancias = pd.DataFrame({
        'feature': feature_names,
        'importancia': clf.feature_importances_
    }).sort_values('importancia', ascending=False)
    
    for _, row in importancias.iterrows():
        print(f"   - {row['feature']}: {row['importancia']:.3f}")
    print()
    
    # ------------------------------------------------------------------
    # 7. EJEMPLO DE PREDICCIÓN
    # Tres textos representativos de niveles 1, 4 y 5
    # ------------------------------------------------------------------
    print("7. Ejemplo de predicción en nuevos textos:")
    print()
    
    textos_ejemplo = [
        "El perro corre rápido.",                                                           # N1 esperado
        "La situación económica actual requiere medidas fiscales excepcionales.",            # N4 esperado
        "El proceso de fotosíntesis implica la conversión de dióxido de carbono en oxígeno." # N5 esperado
    ]
    
    for texto in textos_ejemplo:
        metricas = calcular_metricas_texto(texto)
        features = np.array([[
            metricas['palabras_por_frase'],
            metricas['silabas_por_palabra'],
            metricas['longitud_promedio_palabra'],
            metricas['ratio_palabras_largas'],
            metricas['inflesz']
        ]])
        
        nivel_pred = clf.predict(features)[0]
        proba = clf.predict_proba(features)[0]
        confianza = proba[nivel_pred - 1] * 100  # Niveles 1-5, índice 0-4
        
        print(f"   Texto: '{texto}'")
        print(f"   → Nivel predicho: {nivel_pred} (confianza: {confianza:.1f}%)")
        print(f"   → INFLESZ: {metricas['inflesz']:.1f}")
        print()
    
    print("=" * 70)
    print("✓ PROCESO COMPLETADO")
    print("=" * 70)
    
    # ------------------------------------------------------------------
    # 8. GUARDAR MODELO BASELINE
    # Se guarda para comparación con el modelo final de embeddings
    # ------------------------------------------------------------------
    import pickle
    with open('modelo_clasificador_baseline.pkl', 'wb') as f:
        pickle.dump(clf, f)
    print("\n✓ Modelo baseline guardado en 'modelo_clasificador_baseline.pkl'")
    print("  (El modelo final con embeddings está en models/clasificador_embeddings.pkl)")


if __name__ == "__main__":
    main()