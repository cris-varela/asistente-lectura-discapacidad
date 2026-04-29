"""
CLASIFICADOR BÁSICO DE DIFICULTAD LECTORA
Sistema de Asistencia Lectura para Personas con Discapacidad Cognitiva
TFM IABD - Cristina Varela

Este script implementa un clasificador básico que:
1. Carga textos etiquetados con niveles de dificultad
2. Calcula métricas de legibilidad (INFLESZ, features básicos)
3. Entrena un clasificador RandomForest
4. Evalúa el rendimiento
5. Predice el nivel de nuevos textos
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
    Regla simple: cada vocal o grupo de vocales = 1 sílaba
    """
    palabra = palabra.lower()
    # Eliminar h muda
    palabra = palabra.replace('h', '')
    # Contar grupos de vocales
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
    
    return max(1, silabas)  # Mínimo 1 sílaba

def calcular_metricas_texto(texto):
    """
    Calcula métricas de legibilidad del texto.
    
    Returns:
        dict con las siguientes métricas:
        - num_palabras: número total de palabras
        - num_frases: número de frases
        - palabras_por_frase: promedio palabras/frase
        - silabas_por_palabra: promedio sílabas/palabra
        - longitud_promedio_palabra: promedio caracteres/palabra
        - ratio_palabras_largas: % palabras con >3 sílabas
        - inflesz: índice INFLESZ (adaptación española de Flesch)
    """
    # Normalizar texto
    texto = texto.strip()
    
    # Contar frases (terminan en . ! ?)
    frases = re.split(r'[.!?]+', texto)
    frases = [f.strip() for f in frases if f.strip()]
    num_frases = max(len(frases), 1)
    
    # Contar palabras
    palabras = re.findall(r'\b\w+\b', texto.lower())
    num_palabras = len(palabras)
    
    if num_palabras == 0:
        # Texto vacío - devolver valores por defecto
        return {
            'num_palabras': 0,
            'num_frases': 0,
            'palabras_por_frase': 0,
            'silabas_por_palabra': 0,
            'longitud_promedio_palabra': 0,
            'ratio_palabras_largas': 0,
            'inflesz': 0
        }
    
    # Palabras por frase
    palabras_por_frase = num_palabras / num_frases
    
    # Sílabas
    total_silabas = sum(contar_silabas(palabra) for palabra in palabras)
    silabas_por_palabra = total_silabas / num_palabras
    
    # Longitud promedio de palabra
    longitud_promedio_palabra = sum(len(p) for p in palabras) / num_palabras
    
    # Palabras largas (>3 sílabas)
    palabras_largas = sum(1 for p in palabras if contar_silabas(p) > 3)
    ratio_palabras_largas = palabras_largas / num_palabras
    
    # INFLESZ (Fernández-Huerta adaptado)
    # Fórmula: 206.84 - (60 × sílabas/palabras) - (1.02 × palabras/frases)
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
    Procesa el dataset calculando features para cada texto.
    
    Args:
        df: DataFrame con columnas 'texto' y 'nivel'
    
    Returns:
        X: array de features
        y: array de etiquetas
        feature_names: lista de nombres de features
    """
    features_list = []
    
    for texto in df['texto']:
        metricas = calcular_metricas_texto(texto)
        # Extraer solo las features numéricas (excluir conteos absolutos)
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
    print("CLASIFICADOR DE DIFICULTAD LECTORA")
    print("TFM IABD - Cristina Varela")
    print("=" * 70)
    print()
    
    # 1. CARGAR DATOS
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
    
    # 2. PROCESAR FEATURES
    print("2. Calculando features de legibilidad...")
    X, y, feature_names = procesar_dataset(df)
    print(f"   ✓ Calculados {X.shape[1]} features por texto")
    print(f"   ✓ Features: {', '.join(feature_names)}")
    print()
    
    # 3. DIVIDIR TRAIN/TEST
    print("3. Dividiendo dataset (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   ✓ Train: {len(X_train)} textos")
    print(f"   ✓ Test: {len(X_test)} textos")
    print()
    
    # 4. ENTRENAR CLASIFICADOR
    print("4. Entrenando clasificador RandomForest...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    clf.fit(X_train, y_train)
    print("   ✓ Modelo entrenado")
    print()
    
    # 5. EVALUAR
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
    
    # 6. IMPORTANCIA DE FEATURES
    print("6. Importancia de features:")
    importancias = pd.DataFrame({
        'feature': feature_names,
        'importancia': clf.feature_importances_
    }).sort_values('importancia', ascending=False)
    
    for _, row in importancias.iterrows():
        print(f"   - {row['feature']}: {row['importancia']:.3f}")
    print()
    
    # 7. EJEMPLO DE PREDICCIÓN
    print("7. Ejemplo de predicción en nuevos textos:")
    print()
    
    textos_ejemplo = [
        "El perro corre rápido.",
        "La situación económica actual requiere medidas fiscales excepcionales.",
        "El proceso de fotosíntesis implica la conversión de dióxido de carbono en oxígeno."
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
        confianza = proba[nivel_pred - 1] * 100  # Asume niveles 1-5
        
        print(f"   Texto: '{texto}'")
        print(f"   → Nivel predicho: {nivel_pred} (confianza: {confianza:.1f}%)")
        print(f"   → INFLESZ: {metricas['inflesz']:.1f}")
        print()
    
    print("=" * 70)
    print("✓ PROCESO COMPLETADO")
    print("=" * 70)
    
    # Guardar modelo (opcional)
    import pickle
    with open('modelo_clasificador.pkl', 'wb') as f:
        pickle.dump(clf, f)
    print("\n✓ Modelo guardado en 'modelo_clasificador.pkl'")

if __name__ == "__main__":
    main()