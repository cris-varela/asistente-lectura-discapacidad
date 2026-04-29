"""
ANÁLISIS PROFUNDO DE CALIDAD DE TEXTOS
Identifica textos problemáticos en TODOS los niveles
"""

import pandas as pd
from pathlib import Path
import os

print("=" * 80)
print("ANÁLISIS PROFUNDO DE CALIDAD DE TEXTOS POR NIVEL")
print("=" * 80 + "\n")

BASE_DIR = Path(r'C:\Users\crisv\OneDrive\Desktop\asistente-lectura-discapacidad')

# Verificar archivos en carpetas físicas
print("🔍 VERIFICANDO ARCHIVOS EN CARPETAS FÍSICAS:\n")

niveles = {
    1: BASE_DIR / 'data' / 'raw' / 'nivel_1_lectura_facil',
    2: BASE_DIR / 'data' / 'raw' / 'nivel_2_cuentos_simple',
    3: BASE_DIR / 'data' / 'raw' / 'nivel_3_cuentos_complejo',
    4: BASE_DIR / 'data' / 'raw' / 'nivel_4_narrativa_adulta',
    5: BASE_DIR / 'data' / 'raw' / 'nivel_5_textos_tecnicos'
}

for nivel, carpeta in niveles.items():
    archivos = list(carpeta.glob('*.txt'))
    print(f"Nivel {nivel}: {len(archivos)} archivos")
    
print("\n" + "=" * 80)

# Cargar CSV procesado
CSV_PATH = BASE_DIR / 'data' / 'processed' / 'textos_embeddings.csv'
df = pd.read_csv(CSV_PATH, encoding='utf-8')

print(f"📊 Dataset CSV: {len(df)} fragmentos\n")

for nivel in [1, 2, 3, 4, 5]:
    nivel_df = df[df['nivel'] == nivel]
    archivos_unicos = nivel_df['fuente'].nunique()
    fragmentos = len(nivel_df)
    print(f"Nivel {nivel}: {fragmentos} fragmentos de {archivos_unicos} archivos")

print("\n" + "=" * 80)
print("📊 ANÁLISIS DE CALIDAD POR NIVEL")
print("=" * 80 + "\n")

# Análisis detallado por nivel
for nivel in [1, 2, 3, 4, 5]:
    print(f"\n{'=' * 80}")
    print(f"NIVEL {nivel}")
    print("=" * 80 + "\n")
    
    nivel_df = df[df['nivel'] == nivel]
    
    # Análisis por archivo
    archivos_stats = []
    
    for archivo in nivel_df['fuente'].unique():
        textos = nivel_df[nivel_df['fuente'] == archivo]['texto'].tolist()
        
        palabras_promedio = sum(len(t.split()) for t in textos) / len(textos)
        
        longitudes = []
        for texto in textos:
            palabras = texto.split()
            longitudes.extend([len(p) for p in palabras])
        
        long_palabra = sum(longitudes) / len(longitudes) if longitudes else 0
        complejidad = palabras_promedio * long_palabra
        
        archivos_stats.append({
            'archivo': archivo,
            'fragmentos': len(textos),
            'palabras_promedio': palabras_promedio,
            'longitud_palabra': long_palabra,
            'complejidad': complejidad
        })
    
    archivos_df = pd.DataFrame(archivos_stats)
    archivos_df = archivos_df.sort_values('complejidad', ascending=False)
    
    # Estadísticas generales
    print(f"Total fragmentos: {len(nivel_df)}")
    print(f"Total archivos: {len(archivos_df)}")
    print(f"Palabras/fragmento promedio: {archivos_df['palabras_promedio'].mean():.1f}")
    print(f"Complejidad promedio: {archivos_df['complejidad'].mean():.1f}\n")
    
    # Mostrar todos los archivos ordenados por complejidad
    print(f"TODOS LOS ARCHIVOS (ordenados por complejidad):\n")
    
    for idx, row in archivos_df.iterrows():
        # Marcar extremos
        if row['complejidad'] > archivos_df['complejidad'].quantile(0.75):
            emoji = "🔴"
            nota = "MUY COMPLEJO"
        elif row['complejidad'] < archivos_df['complejidad'].quantile(0.25):
            emoji = "🔵"
            nota = "MUY SIMPLE"
        else:
            emoji = "🟢"
            nota = "NORMAL"
        
        print(f"{emoji} {row['archivo']}")
        print(f"   Complejidad: {row['complejidad']:.1f} | {row['fragmentos']} frag | {row['palabras_promedio']:.1f} pal/frag | {nota}")

# Comparación entre niveles
print("\n" + "=" * 80)
print("📊 COMPARACIÓN ENTRE NIVELES")
print("=" * 80 + "\n")

for nivel in [1, 2, 3, 4, 5]:
    nivel_df = df[df['nivel'] == nivel]
    textos = nivel_df['texto'].tolist()
    palabras_promedio = sum(len(t.split()) for t in textos) / len(textos) if textos else 0
    print(f"Nivel {nivel}: {palabras_promedio:.1f} palabras/fragmento")

# Análisis de solapamiento
print("\n" + "=" * 80)
print("⚠️ ANÁLISIS DE SOLAPAMIENTO ENTRE NIVELES")
print("=" * 80 + "\n")

print("Buscando archivos con complejidad similar entre niveles...\n")

for nivel_actual in [1, 2, 3, 4]:
    nivel_sig = nivel_actual + 1
    
    nivel_actual_df = df[df['nivel'] == nivel_actual]
    nivel_sig_df = df[df['nivel'] == nivel_sig]
    
    # Calcular complejidad por archivo en ambos niveles
    archivos_actual = []
    for archivo in nivel_actual_df['fuente'].unique():
        textos = nivel_actual_df[nivel_actual_df['fuente'] == archivo]['texto'].tolist()
        palabras_promedio = sum(len(t.split()) for t in textos) / len(textos)
        longitudes = []
        for texto in textos:
            palabras = texto.split()
            longitudes.extend([len(p) for p in palabras])
        long_palabra = sum(longitudes) / len(longitudes) if longitudes else 0
        complejidad = palabras_promedio * long_palabra
        archivos_actual.append({'archivo': archivo, 'complejidad': complejidad, 'nivel': nivel_actual})
    
    archivos_sig = []
    for archivo in nivel_sig_df['fuente'].unique():
        textos = nivel_sig_df[nivel_sig_df['fuente'] == archivo]['texto'].tolist()
        palabras_promedio = sum(len(t.split()) for t in textos) / len(textos)
        longitudes = []
        for texto in textos:
            palabras = texto.split()
            longitudes.extend([len(p) for p in palabras])
        long_palabra = sum(longitudes) / len(longitudes) if longitudes else 0
        complejidad = palabras_promedio * long_palabra
        archivos_sig.append({'archivo': archivo, 'complejidad': complejidad, 'nivel': nivel_sig})
    
    df_actual = pd.DataFrame(archivos_actual)
    df_sig = pd.DataFrame(archivos_sig)
    
    max_actual = df_actual['complejidad'].max()
    min_sig = df_sig['complejidad'].min()
    
    print(f"Nivel {nivel_actual} vs Nivel {nivel_sig}:")
    print(f"  Máxima complejidad N{nivel_actual}: {max_actual:.1f}")
    print(f"  Mínima complejidad N{nivel_sig}: {min_sig:.1f}")
    
    if max_actual > min_sig:
        print(f"  ⚠️ HAY SOLAPAMIENTO: {max_actual - min_sig:.1f} puntos")
        
        # Archivos problemáticos
        problematicos_actual = df_actual[df_actual['complejidad'] > min_sig].sort_values('complejidad', ascending=False)
        problematicos_sig = df_sig[df_sig['complejidad'] < max_actual].sort_values('complejidad', ascending=True)
        
        if len(problematicos_actual) > 0:
            print(f"\n  Archivos de N{nivel_actual} MÁS COMPLEJOS que algunos de N{nivel_sig}:")
            for idx, row in problematicos_actual.head(5).iterrows():
                print(f"    • {row['archivo']} (comp: {row['complejidad']:.1f})")
        
        if len(problematicos_sig) > 0:
            print(f"\n  Archivos de N{nivel_sig} MÁS SIMPLES que algunos de N{nivel_actual}:")
            for idx, row in problematicos_sig.head(5).iterrows():
                print(f"    • {row['archivo']} (comp: {row['complejidad']:.1f})")
    else:
        print(f"  ✅ No hay solapamiento")
    
    print()

print("\n" + "=" * 80)
print("✅ ANÁLISIS COMPLETADO")
print("=" * 80 + "\n")

print("💡 PRÓXIMOS PASOS:")
print("1. Revisar archivos marcados como 🔴 MUY COMPLEJO o 🔵 MUY SIMPLE")
print("2. Considerar mover o eliminar archivos con solapamiento problemático")
print("3. Buscar nuevos textos para reemplazar archivos problemáticos")
