"""
SCRIPT OCR PARA LIBROS DE BLANCA
TFM IABD - Cristina Varela

Este script convierte fotos de libros infantiles en archivos .txt
"""

import pytesseract
from PIL import Image
from pathlib import Path
import sys

def procesar_libro(carpeta_fotos, nombre_libro, carpeta_salida='.'):
    """
    Procesa todas las fotos de un libro y crea 1 archivo .txt
    
    Args:
        carpeta_fotos: Ruta a la carpeta con las fotos del libro
        nombre_libro: Nombre para el archivo de salida
        carpeta_salida: Dónde guardar el .txt (por defecto: carpeta actual)
    """
    carpeta = Path(carpeta_fotos)
    
    if not carpeta.exists():
        print(f"❌ ERROR: La carpeta '{carpeta_fotos}' no existe")
        return False
    
    # Buscar todas las imágenes (jpg, jpeg, png)
    fotos = []
    fotos.extend(sorted(carpeta.glob('*.jpg')))
    fotos.extend(sorted(carpeta.glob('*.jpeg')))
    fotos.extend(sorted(carpeta.glob('*.JPG')))
    fotos.extend(sorted(carpeta.glob('*.JPEG')))
    fotos.extend(sorted(carpeta.glob('*.png')))
    fotos.extend(sorted(carpeta.glob('*.PNG')))
    
    fotos = sorted(set(fotos))  # Eliminar duplicados y ordenar
    
    if not fotos:
        print(f"❌ ERROR: No se encontraron imágenes en '{carpeta_fotos}'")
        return False
    
    print("\n" + "=" * 70)
    print(f"📖 PROCESANDO: {nombre_libro}")
    print("=" * 70)
    print(f"Carpeta: {carpeta_fotos}")
    print(f"Páginas encontradas: {len(fotos)}")
    print()
    
    texto_completo = []
    
    for i, foto in enumerate(fotos, 1):
        print(f"  Página {i:2d}/{len(fotos)}: {foto.name:40s}", end=' ')
        
        try:
            # Leer imagen
            img = Image.open(foto)
            
            # Extraer texto en español
            texto = pytesseract.image_to_string(img, lang='spa')
            
            # Limpiar un poco
            texto = texto.strip()
            
            if texto:
                texto_completo.append(texto)
                print(f"✓ ({len(texto.split())} palabras)")
            else:
                print("⚠️  (vacío)")
                
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
    
    if not texto_completo:
        print(f"\n❌ ERROR: No se pudo extraer texto de ninguna imagen")
        return False
    
    # Unir todo el texto con saltos de página
    texto_final = '\n\n'.join(texto_completo)
    
    # Guardar archivo
    output_path = Path(carpeta_salida) / f"{nombre_libro}.txt"
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(texto_final)
        
        print()
        print("=" * 70)
        print(f"✓ COMPLETADO: {nombre_libro}")
        print("=" * 70)
        print(f"Archivo guardado: {output_path}")
        print(f"Total palabras: {len(texto_final.split())}")
        print(f"Total caracteres: {len(texto_final)}")
        print()
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR al guardar: {str(e)}")
        return False


def procesar_varios_libros(carpeta_base, libros, carpeta_salida='.'):
    """
    Procesa múltiples libros de una vez
    
    Args:
        carpeta_base: Carpeta raíz donde están las subcarpetas de cada libro
        libros: Lista de tuplas (nombre_carpeta, nombre_salida)
        carpeta_salida: Dónde guardar los .txt
    """
    print("\n" + "=" * 70)
    print("PROCESAMIENTO DE LIBROS INFANTILES - OCR")
    print("=" * 70)
    print(f"Libros a procesar: {len(libros)}")
    print()
    
    exitosos = 0
    fallidos = 0
    
    for nombre_carpeta, nombre_salida in libros:
        carpeta_fotos = Path(carpeta_base) / nombre_carpeta
        
        if procesar_libro(carpeta_fotos, nombre_salida, carpeta_salida):
            exitosos += 1
        else:
            fallidos += 1
    
    print("\n" + "=" * 70)
    print("RESUMEN FINAL")
    print("=" * 70)
    print(f"✓ Procesados correctamente: {exitosos}")
    print(f"❌ Con errores: {fallidos}")
    print()


# ============================================================================
# CONFIGURACIÓN - EDITA AQUÍ TUS RUTAS
# ============================================================================

if __name__ == "__main__":
    
    # OPCIÓN 1: PROCESAR UN SOLO LIBRO
    # ---------------------------------
    # Descomenta estas líneas y edita las rutas:
    
    # procesar_libro(
    #     carpeta_fotos='C:/fotos_libros/monstruo_colores',
    #     nombre_libro='monstruo_colores_emociones',
    #     carpeta_salida='C:/Users/crisv/OneDrive/Desktop/asistente-lectura-discapacidad/data/raw/nivel_2_cuentos_simple'
    # )
    
    
    # OPCIÓN 2: PROCESAR VARIOS LIBROS A LA VEZ
    # ------------------------------------------
    # Descomenta y edita:
    
    # CARPETA_BASE = 'C:/fotos_libros'
    # CARPETA_SALIDA = 'C:/Users/crisv/OneDrive/Desktop/asistente-lectura-discapacidad/data/raw/nivel_2_cuentos_simple'
    # 
    # LIBROS = [
    #     ('monstruo_colores', 'monstruo_colores_emociones'),
    #     ('daniela_pirata', 'daniela_pirata'),
    #     ('lobo_cole', 'lobo_fue_al_cole'),
    #     ('lobo_viajar', 'lobo_queria_viajar'),
    #     # Añade más libros aquí...
    # ]
    # 
    # procesar_varios_libros(CARPETA_BASE, LIBROS, CARPETA_SALIDA)
    
    
    # Si no editaste nada arriba, muestra ayuda
    print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║  SCRIPT OCR PARA LIBROS INFANTILES                                 ║
    ╚════════════════════════════════════════════════════════════════════╝
    
    ANTES DE EJECUTAR:
    
    1. Edita este script (procesar_libros_ocr.py)
    2. Ve a la sección "CONFIGURACIÓN" (línea ~140)
    3. Descomenta OPCIÓN 1 o OPCIÓN 2
    4. Edita las rutas según tu PC
    5. Guarda el archivo
    6. Ejecuta: python procesar_libros_ocr.py
    
    EJEMPLO OPCIÓN 1 (un libro):
    
        procesar_libro(
            carpeta_fotos='C:/fotos_libros/monstruo_colores',
            nombre_libro='monstruo_colores_emociones',
            carpeta_salida='C:/.../data/raw/nivel_2_cuentos_simple'
        )
    
    EJEMPLO OPCIÓN 2 (varios libros):
    
        CARPETA_BASE = 'C:/fotos_libros'
        CARPETA_SALIDA = 'C:/.../data/raw/nivel_2_cuentos_simple'
        
        LIBROS = [
            ('monstruo_colores', 'monstruo_colores'),
            ('daniela_pirata', 'daniela_pirata'),
            ('lobo_cole', 'lobo_cole'),
        ]
        
        procesar_varios_libros(CARPETA_BASE, LIBROS, CARPETA_SALIDA)
    
    ╔════════════════════════════════════════════════════════════════════╗
    ║  Para más ayuda, lee el archivo INSTRUCCIONES_OCR.md               ║
    ╚════════════════════════════════════════════════════════════════════╝
    """)
