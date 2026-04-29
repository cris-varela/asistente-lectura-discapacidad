"""
EXTRAER RECETAS DE PDF - NIVEL 5
TFM IABD - Cristina Varela

Este script extrae recetas de un PDF y las guarda como archivos .txt individuales
"""

import pypdf
from pathlib import Path
import re

def extraer_recetas_pdf(pdf_path, output_folder, max_recetas=20):
    """
    Extrae recetas de un PDF y las guarda como archivos .txt
    
    Args:
        pdf_path: Ruta al PDF con recetas
        output_folder: Carpeta donde guardar los .txt
        max_recetas: Número máximo de recetas a extraer
    """
    print("\n" + "=" * 70)
    print("EXTRACCIÓN DE RECETAS DESDE PDF")
    print("=" * 70)
    print(f"PDF: {pdf_path}")
    print(f"Carpeta salida: {output_folder}")
    print()
    
    # Crear carpeta de salida si no existe
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Abrir PDF
    try:
        with open(pdf_path, 'rb') as file:
            reader = pypdf.PdfReader(file)
            total_paginas = len(reader.pages)
            
            print(f"Total páginas en PDF: {total_paginas}")
            print()
            
            # Extraer texto completo
            texto_completo = ""
            for i, page in enumerate(reader.pages):
                print(f"  Leyendo página {i+1}/{total_paginas}...", end=' ')
                texto = page.extract_text()
                texto_completo += texto + "\n\n"
                print("✓")
            
            print()
            print(f"Texto extraído: {len(texto_completo)} caracteres")
            print()
            
            # Guardar texto completo para inspección manual
            texto_completo_path = Path(output_folder) / "_texto_completo_pdf.txt"
            with open(texto_completo_path, 'w', encoding='utf-8') as f:
                f.write(texto_completo)
            
            print(f"✓ Texto completo guardado en: {texto_completo_path}")
            print()
            print("=" * 70)
            print("INSTRUCCIONES PARA DIVIDIR RECETAS MANUALMENTE:")
            print("=" * 70)
            print()
            print("1. Abre el archivo: _texto_completo_pdf.txt")
            print()
            print("2. Busca títulos de recetas (suelen estar en MAYÚSCULAS o con formato especial)")
            print()
            print("3. Copia cada receta completa:")
            print("   - Título")
            print("   - Ingredientes")
            print("   - Preparación")
            print()
            print("4. Pega cada receta en un archivo nuevo:")
            print("   receta_pasta_carbonara.txt")
            print("   receta_risotto_hongos.txt")
            print("   receta_ossobuco_milanese.txt")
            print("   ... etc")
            print()
            print("5. Guarda en: data/raw/nivel_5_tecnico_especializado/")
            print()
            print("=" * 70)
            print()
            
            # Buscar patrones comunes de recetas
            # (Esto es aproximado - puede necesitar ajuste manual)
            print("Buscando patrones de recetas...")
            print()
            
            # Patrones comunes en libros de recetas
            # - Títulos en mayúsculas
            # - Secciones: INGREDIENTES, PREPARACIÓN, etc.
            
            # Dividir por saltos grandes de página
            secciones = texto_completo.split('\n\n\n')
            
            recetas_encontradas = 0
            
            for i, seccion in enumerate(secciones[:max_recetas*2]):  # Revisar más secciones por si acaso
                seccion = seccion.strip()
                
                # Criterios para considerar que es una receta:
                # - Longitud razonable (200-2000 caracteres)
                # - Contiene palabras clave comunes en recetas
                
                if len(seccion) < 200 or len(seccion) > 3000:
                    continue
                
                # Palabras clave en recetas
                palabras_clave = [
                    'ingrediente', 'preparación', 'elaboración',
                    'gr', 'ml', 'cucharada', 'minuto',
                    'horno', 'sartén', 'cocinar', 'freír'
                ]
                
                tiene_palabras_clave = any(palabra in seccion.lower() for palabra in palabras_clave)
                
                if tiene_palabras_clave and recetas_encontradas < max_recetas:
                    recetas_encontradas += 1
                    
                    # Intentar extraer título (primera línea)
                    lineas = seccion.split('\n')
                    titulo = lineas[0][:50] if lineas else f"receta_{recetas_encontradas}"
                    
                    # Limpiar título para nombre de archivo
                    titulo_limpio = re.sub(r'[^\w\s-]', '', titulo.lower())
                    titulo_limpio = re.sub(r'[-\s]+', '_', titulo_limpio)
                    titulo_limpio = titulo_limpio[:40]  # Limitar longitud
                    
                    nombre_archivo = f"receta_{titulo_limpio}_{recetas_encontradas}.txt"
                    
                    # Guardar
                    output_path = Path(output_folder) / nombre_archivo
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(seccion)
                    
                    print(f"  ✓ Receta {recetas_encontradas}: {nombre_archivo}")
                    print(f"    Longitud: {len(seccion)} caracteres")
                    print()
            
            print()
            print("=" * 70)
            print("RESUMEN")
            print("=" * 70)
            print(f"Posibles recetas extraídas automáticamente: {recetas_encontradas}")
            print()
            print("⚠️  IMPORTANTE: REVISA MANUALMENTE los archivos generados")
            print("    - Algunos pueden no ser recetas completas")
            print("    - Algunos pueden estar mal divididos")
            print("    - Ajusta manualmente según sea necesario")
            print()
            print("RECOMENDACIÓN:")
            print("Usa el archivo '_texto_completo_pdf.txt' para copiar recetas")
            print("manualmente con más precisión.")
            print()
            
            return recetas_encontradas
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return 0


# ============================================================================
# CONFIGURACIÓN - EDITA AQUÍ
# ============================================================================

if __name__ == "__main__":
    
    # PASO 1: Edita estas rutas según tu PC
    PDF_PATH = 'C:/Users/crisv/Downloads/recetas-cocina-italiana.pdf'
    OUTPUT_FOLDER = 'C:/Users/crisv/OneDrive/Desktop/asistente-lectura-discapacidad/data/raw/nivel_5_tecnico_especializado'
    
    # PASO 2: Ejecuta el script
    extraer_recetas_pdf(PDF_PATH, OUTPUT_FOLDER, max_recetas=20)
    
    print("\n")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║  SIGUIENTE PASO:                                                   ║")
    print("║                                                                    ║")
    print("║  1. Revisa los archivos .txt generados                            ║")
    print("║  2. Abre '_texto_completo_pdf.txt'                                ║")
    print("║  3. Copia manualmente 15-20 recetas completas                     ║")
    print("║  4. Guarda cada una como: receta_nombre.txt                       ║")
    print("║  5. Mueve a: nivel_5_tecnico_especializado/                       ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print()
