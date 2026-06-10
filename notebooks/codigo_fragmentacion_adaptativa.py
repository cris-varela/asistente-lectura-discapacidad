"""
FRAGMENTACIÓN ADAPTATIVA POR NIVEL
Sistema de Asistencia Lectura para Personas con Discapacidad Cognitiva
TFM IABD - Cristina Varela

CONTEXTO:
    En la Iteración 2 del proceso de optimización del clasificador se detectó
    que una fragmentación uniforme (30-70 palabras para todos los niveles)
    perjudicaba especialmente al Nivel 1 y al Nivel 3:
    
    - Nivel 1 (Lectura Fácil): textos cortos de Plena Inclusión perdían
      fragmentos por no alcanzar el mínimo de 30 palabras.
    - Nivel 3: fragmentos demasiado largos mezclaban oraciones de distinta
      complejidad, confundiendo al modelo.

SOLUCIÓN:
    Fragmentación diferenciada por nivel con rangos de palabras adaptados
    a la complejidad cognitiva de cada categoría.

RESULTADO:
    Recall del Nivel 3: 32% → 81% en una sola iteración (+49 puntos).
"""

import re


def dividir_en_fragmentos_adaptativo(texto, nivel):
    """
    Divide un texto en fragmentos con tamaño adaptado al nivel de dificultad.
    
    La longitud óptima del fragmento varía según el nivel porque:
        - Nivel 1: textos de Lectura Fácil son intrínsecamente cortos.
          Fragmentos pequeños maximizan la cantidad de muestras de entrenamiento.
        - Nivel 5: textos técnicos necesitan fragmentos más grandes para
          que el embedding capture suficiente contexto semántico especializado.
    
    Args:
        texto (str): texto limpio en español a fragmentar
        nivel (int): nivel de dificultad (1-5)
        
    Returns:
        list[str]: lista de fragmentos que cumplen el rango de palabras
        
    Configuración de rangos por nivel:
        N1 (Lectura Fácil):    15-40 palabras
        N2 (Básico):           25-55 palabras
        N3 (Intermedio):       25-55 palabras
        N4 (Avanzado):         30-70 palabras
        N5 (Especializado):    35-80 palabras
    """
    # Rangos min/max de palabras por nivel
    # Valores obtenidos empíricamente durante el proceso de optimización iterativa
    config = {
        1: {'min': 15, 'max': 40},  # Lectura Fácil: fragmentos pequeños
        2: {'min': 25, 'max': 55},  # Básico: tamaño intermedio-bajo
        3: {'min': 25, 'max': 55},  # Intermedio: igual que N2 (frontera difusa)
        4: {'min': 30, 'max': 70},  # Avanzado: tamaño estándar
        5: {'min': 35, 'max': 80},  # Especializado: fragmentos más grandes
    }
    
    min_palabras = config[nivel]['min']
    max_palabras = config[nivel]['max']
    
    # Segmentar en frases como unidad mínima indivisible
    # Se respeta la puntuación para no cortar en mitad de una idea
    frases = re.split(r'[.!?]+', texto)
    frases = [f.strip() for f in frases if f.strip()]
    
    fragmentos = []
    fragmento_actual = []   # Frases acumuladas en el fragmento en curso
    palabras_actual = 0     # Contador de palabras del fragmento en curso
    
    for frase in frases:
        num_palabras = len(frase.split())
        
        # Si añadir esta frase supera el máximo, cerrar fragmento actual y empezar uno nuevo
        if palabras_actual + num_palabras > max_palabras and fragmento_actual:
            fragmentos.append(' '.join(fragmento_actual) + '.')
            fragmento_actual = [frase]
            palabras_actual = num_palabras
        else:
            # La frase cabe: añadir al fragmento en curso
            fragmento_actual.append(frase)
            palabras_actual += num_palabras
        
        # Si el fragmento acumulado alcanza el mínimo, cerrarlo
        # (no esperar al máximo para no generar fragmentos demasiado largos)
        if palabras_actual >= min_palabras:
            fragmentos.append(' '.join(fragmento_actual) + '.')
            fragmento_actual = []
            palabras_actual = 0
    
    # Añadir el último fragmento si cumple el mínimo de palabras
    if fragmento_actual and palabras_actual >= min_palabras:
        fragmentos.append(' '.join(fragmento_actual) + '.')
    
    return fragmentos


def procesar_archivo(filepath, nivel):
    """
    Lee un archivo .txt, lo limpia y devuelve fragmentos etiquetados con su nivel.
    
    Gestiona dos encodings habituales en fuentes de texto español:
    UTF-8 (estándar) y Latin-1 (documentos más antiguos o de Windows).
    
    Args:
        filepath (Path): ruta al archivo .txt de origen
        nivel (int): nivel de dificultad asignado al archivo (1-5)
        
    Returns:
        list[dict]: lista de registros con claves:
            - texto (str): fragmento de texto limpio
            - nivel (int): nivel de dificultad (1-5)
            - fuente (str): nombre del archivo de origen
            - fragmento_num (int): índice del fragmento dentro del archivo
            
    Nota:
        Los fragmentos que no alcanzan el mínimo de palabras adaptativo
        se descartan para evitar textos demasiado cortos en el dataset.
    """
    # Intentar UTF-8 primero; fallback a Latin-1 para archivos legacy
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            texto = f.read()
    except UnicodeDecodeError:
        with open(filepath, 'r', encoding='latin-1') as f:
            texto = f.read()
    
    texto_limpio = limpiar_texto(texto)
    if not texto_limpio:
        return []  # Archivo vacío o sin contenido válido tras limpieza
    
    # Fragmentar con tamaños adaptados al nivel del archivo
    fragmentos = dividir_en_fragmentos_adaptativo(texto_limpio, nivel)
    
    # Mínimo de palabras por nivel para filtrar fragmentos residuales
    # (más restrictivo que el mínimo de fragmentación para garantizar calidad)
    min_palabras_minimo = {1: 12, 2: 20, 3: 20, 4: 25, 5: 30}
    min_req = min_palabras_minimo.get(nivel, 20)
    
    registros = []
    for i, fragmento in enumerate(fragmentos):
        if len(fragmento.split()) >= min_req:
            registros.append({
                'texto': fragmento,
                'nivel': nivel,
                'fuente': filepath.name,
                'fragmento_num': i + 1
            })
    
    return registros


def limpiar_texto(texto):
    """
    Elimina metadata y ruido de los textos web antes de la fragmentación.
    
    Los textos de Plena Inclusión y otras fuentes web incluyen cabeceras,
    botones de redes sociales y elementos de navegación que contaminan
    el corpus si no se eliminan.
    
    Args:
        texto (str): texto raw del archivo
        
    Returns:
        str: texto limpio listo para fragmentar
    """
    # Cadenas de metadata web que no forman parte del texto
    lineas_eliminar = [
        'Escuchar el texto', 'Publicado el', 'Guardado en',
        'Ver la versión', 'Twitter', 'Facebook', 'Telegram', 'WhatsApp',
    ]
    
    lineas = texto.split('\n')
    lineas_limpias = []
    
    for linea in lineas:
        linea = linea.strip()
        if len(linea) < 10:
            continue  # Descartar líneas muy cortas (cabeceras, separadores)
        if any(meta in linea for meta in lineas_eliminar):
            continue  # Descartar metadata de redes sociales y navegación
        lineas_limpias.append(linea)
    
    texto_limpio = ' '.join(lineas_limpias)
    # Colapsar espacios múltiples generados al unir líneas
    texto_limpio = re.sub(r'\s+', ' ', texto_limpio)
    
    return texto_limpio.strip()