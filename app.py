import os
import json
import datetime
import threading
import uuid
import time
import site
import sys
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

# --- 1. CORRECCIÓN GPU NVIDIA (NO TOCAR) ---
if os.name == 'nt':
    try:
        site_pkgs = site.getsitepackages()
        for site_pkg in site_pkgs:
            base_nvidia = os.path.join(site_pkg, "nvidia")
            paths_to_add = [
                os.path.join(base_nvidia, "cublas", "bin"),
                os.path.join(base_nvidia, "cudnn", "bin")
            ]
            for path in paths_to_add:
                if os.path.exists(path):
                    try: os.add_dll_directory(path)
                    except: pass
                    os.environ["PATH"] = path + os.pathsep + os.environ["PATH"]
                    print(f"🔧 DLL inyectada: {path}")
    except Exception as e:
        print(f"⚠️ Nota DLLs: {e}")

from faster_whisper import WhisperModel
import google.generativeai as genai

app = Flask(__name__)
CORS(app)

# --- 2. CONFIGURACIÓN DE GEMINI (AUTO-DETECTAR) ---
GENAI_API_KEY = "AIzaSyCf1KJ2NA0aDOu5ChPOZALeEI3Dr9czlB4" # <--- ¡PEGÁ TU API KEY ACÁ!
genai.configure(api_key=GENAI_API_KEY)

# Función inteligente para encontrar un modelo que SÍ funcione
def get_best_model():
    print("\n🔍 Buscando modelo Gemini GRATUITO...")
    try:
        modelos = genai.list_models()
        # Lista de prioridad: Buscamos estos nombres en orden
        prioridades = ["gemini-1.5-flash", "gemini-1.5-flash-001", "gemini-1.5-flash-002", "gemini-1.0-pro"]
        
        nombres_disponibles = []
        for m in modelos:
            if 'generateContent' in m.supported_generation_methods:
                nombres_disponibles.append(m.name)
        
        print(f"📋 Modelos encontrados: {len(nombres_disponibles)}")
        
        # 1. Intentar encontrar un Flash específico
        for prioridad in prioridades:
            for nombre in nombres_disponibles:
                if prioridad in nombre:
                    print(f"✅ Modelo GRATUITO seleccionado: {nombre}")
                    return nombre
        
        # 2. Si no encuentra ninguno, forzar el estándar (Flash)
        print("⚠️ No se encontró coincidencia exacta, forzando 'models/gemini-1.5-flash'...")
        return "models/gemini-1.5-flash"
        
    except Exception as e:
        print(f"⚠️ Error listando modelos: {e}. Usando fallback seguro.")
        return "models/gemini-1.5-flash"

# Detectamos el modelo AL INICIO
MODELO_USAR = "gemini-2.5-flash"

# --- 3. CARGAR WHISPER (GPU) ---
model_size = "medium"
print(f"🚀 Cargando Whisper '{model_size}'...")
try:
    model = WhisperModel(model_size, device="cuda", compute_type="int8")
    print("✅ ¡GPU NVIDIA ACTIVADA!")
except:
    print("⚠️ Usando CPU")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

# Carpetas
DATA_DIR = "datos_clases"
os.makedirs(DATA_DIR, exist_ok=True)
HISTORIAL_FILE = os.path.join(DATA_DIR, "historial.json")
active_tasks = {}

def guardar_en_historial(titulo, materia, resumen, fecha):
    historial = []
    if os.path.exists(HISTORIAL_FILE):
        with open(HISTORIAL_FILE, 'r', encoding='utf-8') as f:
            try:
                historial = json.load(f)
            except:
                pass

    nuevo_item = {
        "id": str(uuid.uuid4()),
        "titulo": titulo,
        "materia": materia,
        "resumen": resumen,
        "fecha": fecha
    }

    historial.insert(0, nuevo_item)

    with open(HISTORIAL_FILE, 'w', encoding='utf-8') as f:
        json.dump(historial, f, ensure_ascii=False, indent=4)

    return nuevo_item

def background_processing(task_id, file_path, materia):
    try:
        # FASE 1: TRANSCRIPCIÓN
        active_tasks[task_id]['status'] = 'transcribing'
        active_tasks[task_id]['message'] = 'Transcribiendo (Whisper GPU)...'
        
        print(f"[{task_id}] 🎙️ Transcribiendo para {materia}...")
        start_time = time.time()
        
        segments, info = model.transcribe(file_path, beam_size=5, language="es")
        texto_completo = ""
        for segment in segments: texto_completo += segment.text + " "
            
        duracion = time.time() - start_time
        print(f"[{task_id}] ✅ Audio listo en {duracion:.2f}s")
        
        if len(texto_completo) < 2: raise Exception("El audio parece vacío.")

        # FASE 2: RESUMEN AVANZADO
        active_tasks[task_id]['status'] = 'summarizing'
        active_tasks[task_id]['message'] = f'Analizando estructura ({MODELO_USAR})...'
        print(f"[{task_id}] 🧠 Generando apuntes avanzados...")

        modelo_gemini = genai.GenerativeModel(MODELO_USAR)
        
        # PROMPT DE INGENIERÍA AVANZADA
        prompt = f"""
        ROL:
        Actuá como un estudiante universitario avanzado que toma apuntes claros, ordenados y útiles para rendir exámenes finales.

        CONTEXTO:
        - Materia: {materia}
        - Origen: Audio real de una clase universitaria (puede ser largo, repetitivo o desordenado).
        - Objetivo: Convertir la clase en apuntes de estudio completos, claros y bien estructurados.

        TEXTO DE LA CLASE (TRANSCRIPCIÓN):
        "{texto_completo}"

        INSTRUCCIONES IMPORTANTES:
        - Priorizá claridad por sobre literalidad.
        - Eliminá repeticiones innecesarias.
        - Si el profesor se desvía, extraé solo lo relevante.
        - Asumí que esto será usado para estudiar.

        FORMATO OBLIGATORIO (Markdown estricto):

        # 📌 TÍTULO DE LA CLASE
        (Título académico claro y representativo del contenido)

        ## 📝 RESUMEN GENERAL
        - 4 a 6 líneas explicando de qué trató la clase.
        - Qué tema central se desarrolló y con qué enfoque.

        ## 🎯 OBJETIVOS DE APRENDIZAJE
        - ¿Qué debería saber entender el estudiante después de esta clase?
        - Lista de 3 a 5 objetivos.

        ## 🔑 CONCEPTOS CLAVE Y DEFINICIONES
        ### Concepto 1
        Definición clara según la explicación del profesor.

        ### Concepto 2
        Definición clara según la explicación del profesor.

        ## 📘 DESARROLLO DE LOS TEMAS
        - Organizá el contenido en subtítulos claros.
        - Usá bullet points.
        - Incluí ejemplos dados en clase.
        - Aclaraciones importantes hechas por el profesor.

        ## ⚠️ ÉNFASIS DEL PROFESOR / ZONA DE EXAMEN
        - Frases como “esto es importante”, “esto entra”, “ojo con”.
        - Errores comunes que el profesor remarcó.
        - Fechas, trabajos prácticos o parciales si se mencionaron.
        - Si no hubo nada explícito, indicarlo.

        ## 📚 REFERENCIAS O AUTORES MENCIONADOS
        - Libros, teorías, autores, leyes o modelos.
        - Si no se mencionó nada concreto, aclararlo.

        REGLAS:
        - No inventes información.
        - No agregues contenido que no esté implícito en el audio.
        - Mantené un tono académico claro y directo.
        """
        
        response = modelo_gemini.generate_content(prompt)
        resumen_final = response.text
        
        # Limpieza de título para la UI
        titulo = "Clase de " + materia
        try:
            for linea in resumen_final.split('\n'):
                if "TÍTULO SUGERIDO" in linea or "1." in linea and "TÍTULO" in linea:
                    titulo = linea.split(':')[1].strip().replace('*', '')
                    break
        except: pass
        
        # Recortar título si es muy largo
        titulo = titulo[:70]

        item = guardar_en_historial(titulo, materia, resumen_final, datetime.datetime.now().strftime("%d/%m %H:%M"))
        
        active_tasks[task_id]['status'] = 'completed'
        active_tasks[task_id]['result'] = item
        print(f"[{task_id}] 🎉 ¡Apuntes generados!")

    except Exception as e:
        print(f"[{task_id}] ❌ ERROR: {str(e)}")
        msg = str(e)
        if "429" in msg: msg = "⚠️ Cuota excedida. Espera 1 min."
        active_tasks[task_id]['status'] = 'error'
        active_tasks[task_id]['message'] = msg
    finally:
        if os.path.exists(file_path): os.remove(file_path)

@app.route('/')
def index(): return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio"}), 400

    file = request.files['audio']
    materia = request.form.get('materia', 'General')
    
    # --- CAMBIO: Detectar extensión original o usar webm por defecto ---
    filename_orig = file.filename
    ext = os.path.splitext(filename_orig)[1]
    if not ext: ext = ".webm" # Fallback para grabaciones de micrófono
    
    task_id = str(uuid.uuid4())
    safe_filename = f"{task_id}{ext}" # Mantiene la extensión original (mp3, wav, etc)
    save_path = os.path.join(DATA_DIR, safe_filename)
    
    file.save(save_path)

    active_tasks[task_id] = {
        'status': 'queued',
        'message': 'Audio recibido.',
        'materia': materia
    }
    
    thread = threading.Thread(
        target=background_processing,
        args=(task_id, save_path, materia)
    )
    thread.daemon = True
    thread.start()
    return jsonify({"task_id": task_id})


@app.route('/status/<task_id>', methods=['GET'])
def check_status(task_id):
    task = active_tasks.get(task_id)
    if not task: return jsonify({"status": "error", "message": "No encontrado"}), 404
    return jsonify(task)

@app.route('/historial', methods=['GET'])
def obtener_historial():
    if os.path.exists(HISTORIAL_FILE):
        with open(HISTORIAL_FILE, 'r', encoding='utf-8') as f:
            try: return jsonify(json.load(f))
            except: return jsonify([])
    return jsonify([])

@app.route('/delete/<string:item_id>', methods=['DELETE'])
def delete_history_item(item_id):
    try:
        if not os.path.exists(HISTORIAL_FILE):
            return jsonify({"error": "No existe el historial"}), 404

        with open(HISTORIAL_FILE, 'r', encoding='utf-8') as f:
            history = json.load(f)

        # Filtramos: Nos quedamos con todos EXCEPTO el que tenga ese ID
        # (Chequeamos tanto 'id' como 'task_id' por seguridad)
        initial_len = len(history)
        new_history = [
            item for item in history 
            if item.get('id') != item_id and item.get('task_id') != item_id
        ]

        if len(new_history) == initial_len:
            return jsonify({"error": "Item no encontrado"}), 404

        with open(HISTORIAL_FILE, 'w', encoding='utf-8') as f:
            json.dump(new_history, f, indent=4, ensure_ascii=False)

        return jsonify({"status": "deleted", "message": "Eliminado correctamente"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("\n🚀 INICIANDO GLAMOURE AI...")
    app.run(host='0.0.0.0', port=5000, debug=True)