"""
DIAGNÓSTICO ULTRA-ESPECÍFICO N3 Y N4
Identifica QUÉ archivos están causando los problemas
"""

import pandas as pd
from pathlib import Path

print("=" * 80)
print("DIAGNÓSTICO ULTRA-ESPECÍFICO - NIVELES 3 Y 4")
print("=" * 80 + "\n")

# Cargar datos
BASE_DIR = Path(r'C:\Users\crisv\OneDrive\Desktop\asistente-lectura-discapacidad')
CSV_PATH = BASE_DIR / 'data' / 'processed' / 'textos_embeddings.csv'

df = pd.read_csv(CSV_PATH, encoding='utf-8')

print(f"📊 Dataset actual: {len(df)} textos\n")

# ============================================================================
# ANÁLISIS NIVEL 2 (81% recall - SOLO SUBIÓ 1 PUNTO, OBJETIVO 85%)
# ============================================================================

print("=" * 80)
print("🔍 NIVEL 2 - RECALL ACTUAL: 81% (antes 80%, objetivo 85%)")
print("=" * 80 + "\n")

nivel2_df = df[df['nivel'] == 2]

print(f"Total fragmentos N2: {len(nivel2_df)}")
print(f"Total archivos N2: {nivel2_df['fuente'].nunique()}\n")

# Calcular complejidad de cada archivo
archivos_n2 = []

for archivo in nivel2_df['fuente'].unique():
    textos = nivel2_df[nivel2_df['fuente'] == archivo]['texto'].tolist()
    
    palabras_promedio = sum(len(t.split()) for t in textos) / len(textos)
    
    longitudes = []
    for texto in textos:
        palabras = texto.split()
        longitudes.extend([len(p) for p in palabras])
    
    long_palabra = sum(longitudes) / len(longitudes) if longitudes else 0
    complejidad = palabras_promedio * long_palabra
    
    archivos_n2.append({
        'archivo': archivo,
        'fragmentos': len(textos),
        'palabras_promedio': palabras_promedio,
        'complejidad': complejidad
    })

archivos_n2_df = pd.DataFrame(archivos_n2)
archivos_n2_df = archivos_n2_df.sort_values('complejidad', ascending=False)

umbral_n2 = archivos_n2_df['complejidad'].quantile(0.75)

print("🔴 ARCHIVOS MÁS COMPLEJOS (probablemente causan confusión con N3):\n")
complejos_n2 = archivos_n2_df[archivos_n2_df['complejidad'] > umbral_n2]

for idx, row in complejos_n2.head(10).iterrows():
    print(f"  {row['archivo']}")
    print(f"    Complejidad: {row['complejidad']:.1f} | {row['fragmentos']} frag | {row['palabras_promedio']:.1f} pal/frag")
print()

# ============================================================================
# ANÁLISIS NIVEL 3 (75% recall - BAJÓ 7 PUNTOS)
# ============================================================================

print("=" * 80)
print("🔍 NIVEL 3 - RECALL ACTUAL: 75% (antes 82%)")
print("=" * 80 + "\n")

nivel3_df = df[df['nivel'] == 3]

print(f"Total fragmentos N3: {len(nivel3_df)}")
print(f"Total archivos N3: {nivel3_df['fuente'].nunique()}\n")

# Calcular complejidad de cada archivo
archivos_n3 = []

for archivo in nivel3_df['fuente'].unique():
    textos = nivel3_df[nivel3_df['fuente'] == archivo]['texto'].tolist()
    
    palabras_promedio = sum(len(t.split()) for t in textos) / len(textos)
    
    longitudes = []
    for texto in textos:
        palabras = texto.split()
        longitudes.extend([len(p) for p in palabras])
    
    long_palabra = sum(longitudes) / len(longitudes) if longitudes else 0
    complejidad = palabras_promedio * long_palabra
    
    archivos_n3.append({
        'archivo': archivo,
        'fragmentos': len(textos),
        'palabras_promedio': palabras_promedio,
        'complejidad': complejidad
    })

archivos_n3_df = pd.DataFrame(archivos_n3)
archivos_n3_df = archivos_n3_df.sort_values('complejidad', ascending=True)

print("ARCHIVOS DE NIVEL 3 (del más simple al más complejo):\n")

# Clasificar archivos
muy_simples = archivos_n3_df[archivos_n3_df['complejidad'] < archivos_n3_df['complejidad'].quantile(0.25)]
simples = archivos_n3_df[(archivos_n3_df['complejidad'] >= archivos_n3_df['complejidad'].quantile(0.25)) & 
                         (archivos_n3_df['complejidad'] < archivos_n3_df['complejidad'].quantile(0.50))]
normales = archivos_n3_df[(archivos_n3_df['complejidad'] >= archivos_n3_df['complejidad'].quantile(0.50)) & 
                          (archivos_n3_df['complejidad'] < archivos_n3_df['complejidad'].quantile(0.75))]
complejos = archivos_n3_df[archivos_n3_df['complejidad'] >= archivos_n3_df['complejidad'].quantile(0.75)]

print("🔵 MUY SIMPLES (probablemente confundidos con N2):\n")
for idx, row in muy_simples.iterrows():
    print(f"  {row['archivo']}")
    print(f"    Complejidad: {row['complejidad']:.1f} | {row['fragmentos']} frag | {row['palabras_promedio']:.1f} pal/frag")
print()

print("🟢 SIMPLES-NORMALES (pueden estar bien en N3):\n")
for idx, row in simples.iterrows():
    print(f"  {row['archivo']}")
    print(f"    Complejidad: {row['complejidad']:.1f} | {row['fragmentos']} frag | {row['palabras_promedio']:.1f} pal/frag")
print()

print("🟡 NORMALES-COMPLEJOS (deberían estar bien en N3):\n")
for idx, row in normales.iterrows():
    print(f"  {row['archivo']}")
    print(f"    Complejidad: {row['complejidad']:.1f} | {row['fragmentos']} frag | {row['palabras_promedio']:.1f} pal/frag")
print()

print("🔴 MUY COMPLEJOS (probablemente confundidos con N4):\n")
for idx, row in complejos.iterrows():
    print(f"  {row['archivo']}")
    print(f"    Complejidad: {row['complejidad']:.1f} | {row['fragmentos']} frag | {row['palabras_promedio']:.1f} pal/frag")
print()

# ============================================================================
# ANÁLISIS NIVEL 4 (83% recall - BAJÓ 4 PUNTOS)
# ============================================================================

print("=" * 80)
print("🔍 NIVEL 4 - RECALL ACTUAL: 83% (antes 87%)")
print("=" * 80 + "\n")

nivel4_df = df[df['nivel'] == 4]

print(f"Total fragmentos N4: {len(nivel4_df)}")
print(f"Total archivos N4: {nivel4_df['fuente'].nunique()}\n")

# Calcular complejidad
archivos_n4 = []

for archivo in nivel4_df['fuente'].unique():
    textos = nivel4_df[nivel4_df['fuente'] == archivo]['texto'].tolist()
    
    palabras_promedio = sum(len(t.split()) for t in textos) / len(textos)
    
    longitudes = []
    for texto in textos:
        palabras = texto.split()
        longitudes.extend([len(p) for p in palabras])
    
    long_palabra = sum(longitudes) / len(longitudes) if longitudes else 0
    complejidad = palabras_promedio * long_palabra
    
    archivos_n4.append({
        'archivo': archivo,
        'fragmentos': len(textos),
        'palabras_promedio': palabras_promedio,
        'complejidad': complejidad
    })

archivos_n4_df = pd.DataFrame(archivos_n4)
archivos_n4_df = archivos_n4_df.sort_values('complejidad', ascending=True)

print("ARCHIVOS DE NIVEL 4 (del más simple al más complejo):\n")

simples_n4 = archivos_n4_df[archivos_n4_df['complejidad'] < archivos_n4_df['complejidad'].quantile(0.50)]
complejos_n4 = archivos_n4_df[archivos_n4_df['complejidad'] >= archivos_n4_df['complejidad'].quantile(0.50)]

print("🔵 SIMPLES (pueden estar confundiéndose con N3):\n")
for idx, row in simples_n4.iterrows():
    print(f"  {row['archivo']}")
    print(f"    Complejidad: {row['complejidad']:.1f} | {row['fragmentos']} frag | {row['palabras_promedio']:.1f} pal/frag")
print()

print("🟢 COMPLEJOS (deberían estar bien en N4):\n")
for idx, row in complejos_n4.iterrows():
    print(f"  {row['archivo']}")
    print(f"    Complejidad: {row['complejidad']:.1f} | {row['fragmentos']} frag | {row['palabras_promedio']:.1f} pal/frag")
print()

# ============================================================================
# COMPARACIÓN ENTRE NIVELES
# ============================================================================

print("=" * 80)
print("📊 COMPARACIÓN ENTRE TODOS LOS NIVELES")
print("=" * 80 + "\n")

for nivel in [1, 2, 3, 4, 5]:
    nivel_df = df[df['nivel'] == nivel]
    textos = nivel_df['texto'].tolist()
    palabras_promedio = sum(len(t.split()) for t in textos) / len(textos) if textos else 0
    print(f"Nivel {nivel}: {palabras_promedio:.1f} palabras/fragmento ({len(nivel_df)} frag)")

# ============================================================================
# RECOMENDACIONES ESPECÍFICAS
# ============================================================================

print("\n" + "=" * 80)
print("💡 RECOMENDACIONES QUIRÚRGICAS ESPECÍFICAS")
print("=" * 80 + "\n")

print("PROBLEMA 1: Nivel 2 solo subió 1 punto (81%, objetivo 85%)")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("Falta +4 puntos")
if len(complejos_n2) > 0:
    print("\nArchivos MÁS COMPLEJOS en N2 (causan confusión):")
    for idx, row in complejos_n2.head(5).iterrows():
        print(f"  • {row['archivo']} ({row['fragmentos']} frag)")
print()

print("\nPROBLEMA 2: Nivel 3 bajó de 82% a 75% (-7 puntos) ← CRÍTICO")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

print("HIPÓTESIS 1: Textos MUY SIMPLES de N3 confunden al modelo")
print("Están demasiado cerca de N2")
print("SOLUCIÓN: Mover los MÁS SIMPLES de N3 → N2\n")

if len(muy_simples) > 0:
    print("Candidatos específicos para N3 → N2:")
    for idx, row in muy_simples.head(3).iterrows():
        print(f"  • {row['archivo']} ({row['fragmentos']} frag)")
    print()

print("HIPÓTESIS 2: Añadimos textos COMPLEJOS a N3 (artículos wiki)")
print("Estos confunden la frontera N3-N4")
print("SOLUCIÓN: Volver a mover artículos wiki de N4 → N3\n")

print("ESTRATEGIA RECOMENDADA:")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("1. DESHACER movimientos de artículos wiki (volver a N3)")
print("   → Esto debería recuperar N3 de 75% a ~82%")
print()
print("2. Mover 3-4 archivos MÁS COMPLEJOS de N2 → N3")
print("   → Esto debería subir N2 de 81% a ~85%")
print()
print("3. Mover SOLO 1-2 textos MUY SIMPLES de N3 → N2")
print("   → Esto refuerza N2 sin debilitar N3")
print()
print("4. Re-entrenar")
print()
print("Objetivos:")
print("  Nivel 2: 81% → 85%+")
print("  Nivel 3: 75% → 85%+")
print("  Nivel 4: 83% → 85%+")
print("  Mantener N1 (92%) y N5 (95%)")

print("\n" + "=" * 80)
print("✅ DIAGNÓSTICO COMPLETADO")
print("=" * 80 + "\n")
