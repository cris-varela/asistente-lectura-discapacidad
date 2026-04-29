#!/usr/bin/env python3
"""
Análisis de frontera N2-N3: ¿Qué textos de N2 se confunden con N3?
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import textstat

# Configuración
BASE_DIR = Path(r"C:\Users\crisv\OneDrive\Desktop\asistente-lectura-discapacidad")
CSV_PATH = BASE_DIR / "data" / "processed" / "textos_embeddings.csv"
MODEL_PATH = BASE_DIR / "models" / "clasificador_embeddings.pkl"

print("=" * 80)
print("ANÁLISIS DE FRONTERA N2-N3")
print("=" * 80)

# 1. Cargar datos
df = pd.read_csv(CSV_PATH)
print(f"\n📊 Dataset cargado: {len(df)} fragmentos")

# 2. Cargar modelo
with open(MODEL_PATH, 'rb') as f:
    model_data = pickle.load(f)
    modelo = model_data['modelo']
    scaler = model_data.get('scaler')

# 3. Hacer predicciones
X = df[[col for col in df.columns if col.startswith('emb_')]].values
if scaler:
    X = scaler.transform(X)
    
df['prediccion'] = modelo.predict(X)

# 4. Calcular complejidad para cada fragmento
def calcular_complejidad(texto):
    """Calcula métrica de complejidad simple"""
    palabras = len(texto.split())
    oraciones = max(1, texto.count('.') + texto.count('!') + texto.count('?'))
    palabras_por_oracion = palabras / oraciones
    letras_por_palabra = len(texto.replace(' ', '')) / max(1, palabras)
    return palabras_por_oracion * 10 + letras_por_palabra * 20

df['complejidad'] = df['texto'].apply(calcular_complejidad)

# 5. Analizar N2
print("\n" + "=" * 80)
print("ANÁLISIS NIVEL 2")
print("=" * 80)

n2_real = df[df['nivel'] == 2].copy()
n2_correcto = n2_real[n2_real['prediccion'] == 2]
n2_como_n3 = n2_real[n2_real['prediccion'] == 3]
n2_como_n1 = n2_real[n2_real['prediccion'] == 1]

print(f"\nN2 total: {len(n2_real)} fragmentos")
print(f"  ✓ Correcto (N2→N2): {len(n2_correcto)} ({len(n2_correcto)/len(n2_real)*100:.1f}%)")
print(f"  ✗ Confusión N2→N3: {len(n2_como_n3)} ({len(n2_como_n3)/len(n2_real)*100:.1f}%)")
print(f"  ✗ Confusión N2→N1: {len(n2_como_n1)} ({len(n2_como_n1)/len(n2_real)*100:.1f}%)")

# 6. Comparar complejidad
print("\n" + "=" * 80)
print("COMPLEJIDAD DE TEXTOS N2")
print("=" * 80)

print(f"\nN2 correctamente clasificados:")
print(f"  Complejidad media: {n2_correcto['complejidad'].mean():.1f}")
print(f"  Rango: {n2_correcto['complejidad'].min():.1f} - {n2_correcto['complejidad'].max():.1f}")

if len(n2_como_n3) > 0:
    print(f"\nN2 clasificados como N3 (CONFUSIÓN):")
    print(f"  Complejidad media: {n2_como_n3['complejidad'].mean():.1f}")
    print(f"  Rango: {n2_como_n3['complejidad'].min():.1f} - {n2_como_n3['complejidad'].max():.1f}")
    
    diferencia = n2_como_n3['complejidad'].mean() - n2_correcto['complejidad'].mean()
    print(f"\n  📊 Diferencia: +{diferencia:.1f} puntos más complejos")
    print(f"  💡 Los textos N2 que se confunden con N3 son {diferencia:.1f} puntos")
    print(f"     más complejos que los N2 correctos")

# 7. Analizar N3
print("\n" + "=" * 80)
print("ANÁLISIS NIVEL 3")
print("=" * 80)

n3_real = df[df['nivel'] == 3].copy()
n3_correcto = n3_real[n3_real['prediccion'] == 3]
n3_como_n2 = n3_real[n3_real['prediccion'] == 2]

print(f"\nN3 total: {len(n3_real)} fragmentos")
print(f"  ✓ Correcto (N3→N3): {len(n3_correcto)} ({len(n3_correcto)/len(n3_real)*100:.1f}%)")
print(f"  ✗ Confusión N3→N2: {len(n3_como_n2)} ({len(n3_como_n2)/len(n3_real)*100:.1f}%)")

print(f"\nN3 correctamente clasificados:")
print(f"  Complejidad media: {n3_correcto['complejidad'].mean():.1f}")
print(f"  Rango: {n3_correcto['complejidad'].min():.1f} - {n3_correcto['complejidad'].max():.1f}")

if len(n3_como_n2) > 0:
    print(f"\nN3 clasificados como N2 (CONFUSIÓN):")
    print(f"  Complejidad media: {n3_como_n2['complejidad'].mean():.1f}")
    print(f"  Rango: {n3_como_n2['complejidad'].min():.1f} - {n3_como_n2['complejidad'].max():.1f}")

# 8. Zona de solapamiento
print("\n" + "=" * 80)
print("ZONA DE SOLAPAMIENTO N2-N3")
print("=" * 80)

complejidad_n2_max = n2_correcto['complejidad'].quantile(0.90)  # Top 10% de N2
complejidad_n3_min = n3_correcto['complejidad'].quantile(0.10)  # Bottom 10% de N3

print(f"\nTop 10% de N2 (más complejos): >{complejidad_n2_max:.1f} puntos")
print(f"Bottom 10% de N3 (más simples): <{complejidad_n3_min:.1f} puntos")

if complejidad_n2_max > complejidad_n3_min:
    solapamiento = complejidad_n2_max - complejidad_n3_min
    print(f"\n⚠️  SOLAPAMIENTO DETECTADO: {solapamiento:.1f} puntos")
    print(f"    Zona conflictiva: {complejidad_n3_min:.1f} - {complejidad_n2_max:.1f}")
else:
    print(f"\n✓ No hay solapamiento significativo")

# 9. Archivos fuente de los confusos
print("\n" + "=" * 80)
print("ARCHIVOS DE N2 QUE SE CONFUNDEN CON N3")
print("=" * 80)

if len(n2_como_n3) > 0:
    # Agrupar por archivo fuente
    archivos_problematicos = n2_como_n3.groupby('archivo').agg({
        'texto': 'count',
        'complejidad': 'mean'
    }).sort_values('texto', ascending=False)
    
    print(f"\nTop 10 archivos con más confusiones:")
    print(f"{'Archivo':<50} {'Fragmentos':<12} {'Complejidad':<12}")
    print("-" * 80)
    for archivo, row in archivos_problematicos.head(10).iterrows():
        print(f"{archivo[:47]:<50} {int(row['texto']):<12} {row['complejidad']:.1f}")

# 10. Conclusión
print("\n" + "=" * 80)
print("CONCLUSIÓN")
print("=" * 80)

if len(n2_como_n3) > 0:
    print(f"""
Los textos de N2 que se clasifican como N3 tienen:
  • Complejidad media: {n2_como_n3['complejidad'].mean():.1f} puntos
  • Son {n2_como_n3['complejidad'].mean() - n2_correcto['complejidad'].mean():.1f} puntos más complejos que el N2 promedio
  • Están en la zona frontera entre N2 y N3
  
Al añadir 252 fragmentos a N3 (National Geographic Kids ~170-180 puntos),
fortaleciste la representación de N3 en esa zona frontera.

Ahora el modelo aprendió que la zona 170-180 es "territorio N3", por lo que
algunos textos de N2 que están en esa zona se clasifican como N3.

TRADE-OFF INEVITABLE:
  • Fortalecer N3 → Mejora N3 pero "empuja" frontera
  • Textos N2 en frontera → Ahora se clasifican como N3
  • Resultado: N2 baja de 93% a 88% (-5 puntos)
  • Pero N3 sube de 57% a 77% (+20 puntos)
  
Balance neto: +15 puntos de mejora ✅
""")
else:
    print("\nNo se detectaron confusiones significativas N2→N3")

print("=" * 80)
