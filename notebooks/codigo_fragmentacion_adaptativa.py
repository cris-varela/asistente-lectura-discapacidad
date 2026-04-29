"""
SOLUCIÓN: Fragmentación diferenciada por nivel
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PROBLEMA ACTUAL:
- Misma fragmentación (30-70 palabras) para todos los niveles
- Nivel 1 y 3 pierden fragmentos porque sus textos son más cortos

SOLUCIÓN:
- Fragmentación ADAPTATIVA por nivel
- Nivel 1: Fragmentos más pequeños (15-40 palabras)
- Nivel 3: Fragmentos medianos (25-55 palabras)
- Niveles 2,4,5: Como están (30-70 palabras)
"""

# REEMPLAZAR la función dividir_en_fragmentos en el notebook
# por esta versión MEJORADA:

def dividir_en_fragmentos_adaptativo(texto, nivel):
    """
    Divide texto en fragmentos con tamaño adaptado al nivel.
    Niveles bajos = fragmentos más pequeños para maximizar cantidad.
    """
    # Configuración por nivel
    config = {
        1: {'min': 15, 'max': 40},  # Lectura fácil - MÁS PEQUEÑOS
        2: {'min': 25, 'max': 55},  # Cuentos simples
        3: {'min': 25, 'max': 55},  # Cuentos complejos - REDUCIDOS
        4: {'min': 30, 'max': 70},  # Narrativa adulta
        5: {'min': 35, 'max': 80},  # Técnico - MÁS GRANDES
    }
    
    min_palabras = config[nivel]['min']
    max_palabras = config[nivel]['max']
    
    # Dividir en frases
    frases = re.split(r'[.!?]+', texto)
    frases = [f.strip() for f in frases if f.strip()]
    
    fragmentos = []
    fragmento_actual = []
    palabras_actual = 0
    
    for frase in frases:
        num_palabras = len(frase.split())
        
        # Si añadir esta frase excede el máximo, guardar fragmento
        if palabras_actual + num_palabras > max_palabras and fragmento_actual:
            fragmentos.append(' '.join(fragmento_actual) + '.')
            fragmento_actual = [frase]
            palabras_actual = num_palabras
        else:
            fragmento_actual.append(frase)
            palabras_actual += num_palabras
        
        # Si alcanzamos el mínimo, podemos cerrar fragmento
        if palabras_actual >= min_palabras:
            fragmentos.append(' '.join(fragmento_actual) + '.')
            fragmento_actual = []
            palabras_actual = 0
    
    # Añadir último fragmento si cumple mínimo
    if fragmento_actual and palabras_actual >= min_palabras:
        fragmentos.append(' '.join(fragmento_actual) + '.')
    
    return fragmentos


# MODIFICAR la función procesar_archivo para usar la nueva fragmentación:

def procesar_archivo(filepath, nivel):
    """
    Procesa un archivo .txt y devuelve fragmentos con nivel.
    Usa fragmentación adaptativa según el nivel.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            texto = f.read()
    except UnicodeDecodeError:
        with open(filepath, 'r', encoding='latin-1') as f:
            texto = f.read()
    
    texto_limpio = limpiar_texto(texto)
    if not texto_limpio:
        return []
    
    # CAMBIO AQUÍ: usar fragmentación adaptativa
    fragmentos = dividir_en_fragmentos_adaptativo(texto_limpio, nivel)
    
    # Mínimo de palabras también adaptativo
    min_palabras_minimo = {1: 12, 2: 20, 3: 20, 4: 25, 5: 30}
    min_req = min_palabras_minimo.get(nivel, 20)
    
    registros = []
    for i, fragmento in enumerate(fragmentos):
        if len(fragmento.split()) >= min_req:
            registros.append({
                'texto': fragmento,
                'nivel': nivel,
                'fuente': filepath.name,
                'fragmento_num': i+1
            })
    
    return registros
