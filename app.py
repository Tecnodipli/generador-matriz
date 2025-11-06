import os
import io
import re
import json
import zipfile
import secrets
import pdfplumber
import pandas as pd
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from threading import Lock

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

from tenacity import retry, stop_after_attempt, wait_random_exponential

# ---------- OpenAI ----------
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# =========================
# App & CORS
# =========================
app = FastAPI(title="Generador de Matrices Dipli")

ALLOWED_ORIGINS = [
    "https://www.dipli.ai",
    "https://dipli.ai",
    "https://isagarcivill09.wixsite.com/turop",
    "https://isagarcivill09-wixsite-com.filesusr.com",
    "https://www-dipli-ai.filesusr.com",
    "https://*.filesusr.com",
    "https://*.wixsite.com",
    "https://generador-matriz.onrender.com",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"https://.*\.(filesusr|wixsite)\.com",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Descargas temporales (thread-safe)
# =========================
DOWNLOADS: dict[str, tuple[bytes, str, str, datetime]] = {}
DOWNLOAD_TTL_SECS = 900  # 15 min
ZIP_MEDIA_TYPE = "application/zip"
_DL_LOCK = Lock()

def cleanup_downloads() -> None:
    now = datetime.utcnow()
    with _DL_LOCK:
        expired = [t for t, (_, _, _, exp) in DOWNLOADS.items() if exp <= now]
        for t in expired:
            DOWNLOADS.pop(t, None)

def register_download(data: bytes, filename: str, media_type: str) -> str:
    cleanup_downloads()
    token = secrets.token_urlsafe(16)
    expires_at = datetime.utcnow() + timedelta(seconds=DOWNLOAD_TTL_SECS)
    with _DL_LOCK:
        DOWNLOADS[token] = (data, filename, media_type, expires_at)
    return token

# =========================
# Config sin .env (constantes)
# =========================
# Tamaños/cortes para acelerar el procesamiento
MAX_UPLOAD_MB = 60
MAX_PAGES_CO = 30     # páginas máximas del PDF de Contexto+Objetivos
MAX_PAGES_TR = 60     # páginas máximas del PDF de Transcripción
MAX_PAGES_GUIDE = 20  # páginas máximas del PDF de Guía (opcional)

LLM_MAX_CHUNKS = 6    # tope de llamadas por función de análisis
LLM_CHUNK_SIZE = 9000 # chars por chunk

# Costeo opcional (si no se definen variables de entorno, quedan en 0)
INPUT_COST = float(os.getenv("OPENAI_INPUT_COST_PER_1K", "0.0"))
OUTPUT_COST = float(os.getenv("OPENAI_OUTPUT_COST_PER_1K", "0.0"))

# =========================
# Utilidades PDF / Texto
# =========================
def extract_text_from_pdf(path_or_bytes, max_pages: int | None = None) -> str:
    """Acepta ruta o bytes de PDF. Lee como máximo `max_pages` páginas y concatena texto."""
    texts = []
    if isinstance(path_or_bytes, (bytes, bytearray)):
        bio = io.BytesIO(path_or_bytes)
        with pdfplumber.open(bio) as pdf:
            for idx, page in enumerate(pdf.pages):
                if max_pages is not None and idx >= max_pages:
                    break
                txt = page.extract_text() or ""
                texts.append(txt)
    else:
        with pdfplumber.open(path_or_bytes) as pdf:
            for idx, page in enumerate(pdf.pages):
                if max_pages is not None and idx >= max_pages:
                    break
                txt = page.extract_text() or ""
                texts.append(txt)
    return "\n".join(texts)

def split_into_chunks(text: str, max_chars: int = 8000) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            nl = text.rfind("\n", start, end)
            if nl != -1 and (nl - start) > max_chars * 0.5:
                end = nl
        chunks.append(text[start:end])
        start = end
    return chunks

def split_capped(text: str, max_chars: int, max_chunks: int) -> list[str]:
    """Divide en bloques de hasta max_chars y limita a max_chunks (compactando el resto)."""
    raw_chunks = split_into_chunks(text, max_chars=max_chars)
    if len(raw_chunks) <= max_chunks:
        return raw_chunks
    head = raw_chunks[:max_chunks - 1]
    tail = "\n".join(raw_chunks[max_chunks - 1:])
    return head + [tail]

# =========================
# Cliente LLM con conteo de tokens y costos
# =========================
class LLMClient:
    def __init__(self, model: Optional[str] = None):
        if OpenAI is None:
            raise RuntimeError("Falta openai>=2.x. Instala la dependencia.")
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "No hay clave de OpenAI. Define la variable de entorno OPENAI_API_KEY en tu hosting."
            )
        # Puedes cambiar el modelo por variable de entorno si quieres
        self.model = (model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip()
        self.client = OpenAI(api_key=api_key)

        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def _acc(self, resp) -> None:
        u = getattr(resp, "usage", None)
        if not u:
            return
        self.total_tokens += int(getattr(u, "total_tokens", 0) or 0)
        self.prompt_tokens += int(getattr(u, "prompt_tokens", 0) or 0)
        self.completion_tokens += int(getattr(u, "completion_tokens", 0) or 0)

    # Reintentos cortos para que no “explote” el tiempo total
    @retry(stop=stop_after_attempt(2), wait=wait_random_exponential(multiplier=0.8, max=3))
    def chat(self, system: str, user: str, temperature: float = 0.2) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        self._acc(resp)
        return resp.choices[0].message.content

    @retry(stop=stop_after_attempt(2), wait=wait_random_exponential(multiplier=0.8, max=3))
    def chat_json(self, system: str, user: str, temperature: float = 0.1) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        self._acc(resp)
        return resp.choices[0].message.content

    def cost_usd(self) -> float:
        return (self.prompt_tokens / 1000.0) * INPUT_COST + (self.completion_tokens / 1000.0) * OUTPUT_COST

# =========================
# Resumen jerárquico
# =========================
def hierarchical_summarize(llm: LLMClient, raw_text: str, label: str) -> str:
    chunks = split_capped(raw_text, max_chars=LLM_CHUNK_SIZE, max_chunks=LLM_MAX_CHUNKS)
    summaries = []
    sys = (
        "Eres un analista senior. Resume con foco en: actores, procesos, definiciones, "
        "restricciones, y términos clave. Usa viñetas concisas."
    )
    for i, ch in enumerate(chunks, 1):
        user = f"Resumen parcial {label} ({i}/{len(chunks)}):\n\n{ch}"
        summaries.append(llm.chat(sys, user, temperature=0.0))
    combined = "\n\n".join(f"- Bloque {i+1}: {s}" for i, s in enumerate(summaries))
    final_user = f"Fusiona y depura en 12–18 viñetas claras, sin redundancias:\n\n{combined}"
    final = llm.chat(sys, final_user, temperature=0.0)
    return final

# =========================
# Extracción robusta de preguntas (heurística)
# =========================
_INTERROGATIVOS = r"(qué|como|cómo|cual|cuál|cuando|cuándo|donde|dónde|quien|quién|por qué|para qué|cuanto|cuánto|cuales|cuáles)"
_cond_head_re = re.compile(r"^\s*(si\s+es\s+él|si\s+no\s+es\s+él)\s*:\s*$", re.IGNORECASE)
_is_instr_re = re.compile(r"^\s*(moderador:|se graba|\(se graba\))", re.IGNORECASE)

def _clean_line(s: str) -> str:
    s = s.strip().strip("•").strip("-").strip("·").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _is_questionish(s: str) -> bool:
    if "?" in s or "¿" in s:
        return True
    return re.match(rf"^\s*{_INTERROGATIVOS}\b", s, re.IGNORECASE) is not None

def _ends_like_question(s: str) -> bool:
    return bool(re.search(r"\?\s*$", s)) or s.endswith("?")

def extract_questions_from_text_v2(text: str) -> List[str]:
    if not text:
        return []
    lines = [_clean_line(l) for l in text.splitlines()]
    lines = [l for l in lines if l]
    preguntas = []
    condicion_actual: Optional[str] = None
    i = 0
    while i < len(lines):
        ln = lines[i]
        m = _cond_head_re.match(ln)
        if m:
            condicion_actual = m.group(1).strip()
            i += 1
            continue
        if re.match(r"^[A-ZÁÉÍÓÚÑ].{0,80}$", ln) and not _is_questionish(ln):
            condicion_actual = None
        if _is_instr_re.match(ln):
            i += 1
            continue
        if _is_questionish(ln):
            buf = [ln]
            j = i + 1
            while j < len(lines):
                nxt = lines[j]
                if _is_instr_re.match(nxt) or _cond_head_re.match(nxt):
                    break
                if not _is_questionish(nxt) and not re.match(r"^[A-ZÁÉÍÓÚÑ].{0,80}$", nxt):
                    buf.append(nxt)
                    j += 1
                else:
                    break
            merged = " ".join(buf).strip()
            merged = re.sub(r"^\s*[•\-\u2022]\s*", "", merged)
            if re.match(rf"^{_INTERROGATIVOS}\b", merged, re.IGNORECASE) and not _ends_like_question(merged):
                merged += "?"
            if condicion_actual:
                merged = f"{merged} [Condición: {condicion_actual}]"
            preguntas.append(merged)
            i = j
            continue
        i += 1

    # deduplicación laxa
    def _norm(s: str) -> str:
        t = s.casefold()
        t = t.replace("¿", "").replace("?", "")
        t = re.sub(r"[\s\.,;:()\-]+", " ", t).strip()
        repl = {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ñ": "n"}
        for a, b in repl.items():
            t = t.replace(a, b)
        return t
    seen, out = set(), []
    for q in preguntas:
        k = _norm(q)
        if k in seen:
            continue
        seen.add(k)
        out.append(q)
    return out

# =========================
# Extracción de preguntas desde TRANSCRIPCIÓN con LLM (complemento)
# =========================
def extract_questions_from_transcript_llm(llm: LLMClient, transcript_text: str) -> List[str]:
    """
    Usa LLM para:
      - Normalizar preguntas implícitas del moderador aunque no tengan signos.
      - Inferir 'prompts' de indagación que aparecen fragmentados.
    Devuelve lista de strings (sin duplicados obvios).
    """
    chunks = split_capped(transcript_text, max_chars=LLM_CHUNK_SIZE, max_chunks=LLM_MAX_CHUNKS)
    sys = (
        "Eres un analista cualitativo. Extrae las PREGUNTAS (prompts del moderador) "
        "que aparecen explícitas o implícitas en la transcripción. Devuelve JSON: "
        '{"preguntas": ["..."]}. No reformules innecesariamente; normaliza a preguntas claras.'
    )
    preguntas: List[str] = []
    for i, ch in enumerate(chunks, 1):
        user = f"Transcripción (bloque {i}/{len(chunks)}):\n\n{ch}\n\nDevuelve solo JSON."
        try:
            raw = llm.chat_json(sys, user, temperature=0.0)
            data = json.loads(raw)
            for q in data.get("preguntas", []) or []:
                if isinstance(q, str) and q.strip():
                    preguntas.append(q.strip())
        except Exception:
            continue
    # dedupe simple
    seen, out = set(), []
    for q in preguntas:
        k = re.sub(r"[\s\?¿]+", " ", q.casefold()).strip()
        if k in seen:
            continue
        seen.add(k)
        out.append(q)
    return out

# =========================
# Plantilla de Informe (más informativa)
# =========================
REPORT_SCHEMA = [
    {
        "titulo": "Panorama del Estudio",
        "subcapitulos": [
            {"titulo": "Contexto y alcance del proyecto"},
            {"titulo": "Objetivos y preguntas guía"},
            {"titulo": "Metodología y muestra"},
        ],
    },
    {
        "titulo": "Audiencia y Segmentos",
        "subcapitulos": [
            {"titulo": "Perfil, comportamientos y necesidades"},
            {"titulo": "Segmentaciones relevantes"},
        ],
    },
    {
        "titulo": "Jornadas y Tareas del Usuario",
        "subcapitulos": [
            {"titulo": "Etapas y momentos clave"},
            {"titulo": "Tareas, fricciones y atajos"},
        ],
    },
    {
        "titulo": "Motivadores y Barreras",
        "subcapitulos": [
            {"titulo": "Drivers de adopción y uso"},
            {"titulo": "Barreras, dolores y riesgos"},
            {"titulo": "Trade-offs y condicionantes"},
        ],
    },
    {
        "titulo": "Percepción de la Solución y la Marca",
        "subcapitulos": [
            {"titulo": "Valor percibido y expectativas"},
            {"titulo": "Satisfactores, insatisfactores y comparativos"},
        ],
    },
    {
        "titulo": "Oportunidades de Valor",
        "subcapitulos": [
            {"titulo": "Insights accionables"},
            {"titulo": "Oportunidades priorizadas"},
        ],
    },
    {
        "titulo": "Recomendaciones y Roadmap",
        "subcapitulos": [
            {"titulo": "Quick wins (0–90 días)"},
            {"titulo": "Iniciativas estratégicas (90–180 días)"},
        ],
    },
    {
        "titulo": "Anexos Analíticos",
        "subcapitulos": [
            {"titulo": "Hipótesis y supuestos"},
            {"titulo": "Preguntas abiertas y siguientes pasos"},
        ],
    },
]

MATRIX_INSTRUCTIONS = f"""
Devuelve un JSON válido con esta estructura EXACTA:
{{
  "capitulos": [
    {{
      "titulo": "string",
      "subcapitulos": [
        {{"titulo": "string", "preguntas": ["string", ...]}}
      ]
    }}
  ]
}}

REGLAS:
- Debes usar EXCLUSIVAMENTE los siguientes capítulos y subcapítulos (no inventes ni renombres):
{json.dumps(REPORT_SCHEMA, ensure_ascii=False, indent=2)}

- Asigna TODAS las preguntas a algún subcapítulo.
- Mantén el orden original de las preguntas en lo posible.
- Si una pregunta encaja en varias categorías, elige la que mejor contribuya a un INFORME claro y útil.
- Títulos de capítulos/subcapítulos deben ser EXACTAMENTE los de la plantilla.
- Flujo analítico global: panorama -> audiencia -> jornadas/tareas -> motivadores/barreras -> percepción -> oportunidades -> recomendaciones -> anexos.
"""

def extract_json_block(text: str) -> Optional[str]:
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None

def safe_json_loads(s: str) -> Dict[str, Any]:
    if not s:
        raise ValueError("Respuesta vacía del LLM.")
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`")
        s = s.replace("json\n", "", 1).replace("json\r\n", "", 1)
    try:
        return json.loads(s)
    except Exception:
        pass
    blk = extract_json_block(s)
    if blk:
        return json.loads(blk)
    raise ValueError("No se pudo parsear JSON.")

def build_matrix_with_llm(
    llm: LLMClient,
    contexto_sum: str,
    objetivos_sum: str,
    preguntas: Optional[List[str]],
    n_if_no_guide: int = 60,
) -> Dict[str, Any]:
    sys = "Eres un Research Lead experto en cualitativo. Diseñas estructuras analíticas de INFORME limpias y accionables."
    if preguntas:
        pregunta_blob = "\n".join(f"- {p}" for p in preguntas)
        user = f"""CONTEXTO (resumen):
{contexto_sum}

OBJETIVOS (resumen):
{objetivos_sum}

PREGUNTAS (de guía y/o derivadas de TRANSCRIPCIÓN; no edites el texto):
{pregunta_blob}

{MATRIX_INSTRUCTIONS}
"""
    else:
        user = f"""CONTEXTO (resumen):
{contexto_sum}

OBJETIVOS (resumen):
{objetivos_sum}

No se detectaron preguntas. Genera exactamente {n_if_no_guide} preguntas totales alineadas al informe.
{MATRIX_INSTRUCTIONS}
"""
    try:
        raw = llm.chat_json(sys, user, temperature=0.15)
        return safe_json_loads(raw)
    except Exception:
        raw_fallback = llm.chat(sys, user, temperature=0.15)
        try:
            return safe_json_loads(raw_fallback)
        except Exception:
            fix = llm.chat_json(
                "Eres un validador estricto de JSON. Devuelve únicamente un objeto JSON válido.",
                f"Corrige a JSON válido esta salida (no añadas texto fuera del objeto):\n\n{raw_fallback}",
                temperature=0.0,
            )
            return safe_json_loads(fix)

def to_dataframe(matrix: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for cap in matrix.get("capitulos", []):
        cap_title = (cap.get("titulo") or "").strip()
        for sub in cap.get("subcapitulos", []):
            sub_title = (sub.get("titulo") or "").strip()
            for q in sub.get("preguntas", []) or []:
                rows.append({"Capitulo": cap_title, "Subcapitulo": sub_title, "Preguntas": (q or "").strip()})
    return pd.DataFrame(rows, columns=["Capitulo", "Subcapitulo", "Preguntas"])

# ===== utilidades de estructura =====
def _norm_light(s: str) -> str:
    if s is None:
        return ""
    t = s.strip()
    t = re.sub(r"\s+", " ", t)
    t = t.replace("¿", "").replace("?", "")
    return t.lower()

def _collect_assigned_questions(matrix: Dict[str, Any]) -> list[str]:
    out = []
    for cap in matrix.get("capitulos", []):
        for sub in cap.get("subcapitulos", []) or []:
            for q in sub.get("preguntas", []) or []:
                out.append(q)
    return out

def _structure_snapshot(matrix: Dict[str, Any]) -> List[Dict[str, Any]]:
    snap = []
    for cap in matrix.get("capitulos", []):
        snap.append({
            "capitulo": (cap.get("titulo") or "").strip(),
            "subcapitulos": [ (s.get("titulo") or "").strip() for s in cap.get("subcapitulos", []) ]
        })
    return snap

def _append_in_order(matrix: Dict[str, Any], cap_title: str, sub_title: str, question: str) -> None:
    caps = matrix.get("capitulos", [])
    for cap in caps:
        if (cap.get("titulo") or "").strip().lower() == cap_title.strip().lower():
            for sub in cap.get("subcapitulos", []) or []:
                if (sub.get("titulo") or "").strip().lower() == sub_title.strip().lower():
                    lst = sub.get("preguntas") or []
                    lst.append(question)
                    sub["preguntas"] = lst
                    return
            cap.setdefault("subcapitulos", []).append({"titulo": sub_title, "preguntas": [question]})
            return
    caps.append({"titulo": cap_title, "subcapitulos": [{"titulo": sub_title, "preguntas": [question]}]})
    matrix["capitulos"] = caps

def assign_missing_with_llm(llm: LLMClient, matrix: Dict[str, Any],
                            missing: List[str], contexto_sum: str, objetivos_sum: str) -> Dict[str, Any]:
    snap = _structure_snapshot(matrix)
    sys = "Eres Research Lead. Clasificas preguntas de guía en una estructura dada (capítulos y subcapítulos)."
    if snap:
        user = f"""Estructura actual (úsala tal cual; NO inventes nuevas):
{json.dumps(snap, ensure_ascii=False, indent=2)}

Asigna CADA pregunta a un par (capitulo, subcapitulo) EXISTENTE. Mantén el orden.
Devuelve JSON con:
{{"assignments":[{{"question":"...","capitulo":"...","subcapitulo":"..."}}, ...]}}

Preguntas:
- """ + "\n- ".join(missing)
        raw = llm.chat_json(sys, user, temperature=0.1)
        data = safe_json_loads(raw)
        for it in data.get("assignments", []):
            q = (it.get("question") or "").strip()
            cap = (it.get("capitulo") or "").strip() or "Jornadas y Tareas del Usuario"
            sub = (it.get("subcapitulo") or "").strip() or "Etapas y momentos clave"
            if q:
                _append_in_order(matrix, cap, sub, q)
        return matrix
    else:
        # Si no hay estructura, fuerza la plantilla del informe y agrega por defecto
        matrix["capitulos"] = REPORT_SCHEMA
        for q in missing:
            _append_in_order(matrix, "Jornadas y Tareas del Usuario", "Etapas y momentos clave", q)
        return matrix

def enforce_subchapter_limits(matrix: Dict[str, Any], min_sub: int = 2, max_sub: int = 3) -> Dict[str, Any]:
    # Mantiene 2–3 subcapítulos; si hay más, combina los excedentes en un "· Combinado"
    for cap in matrix.get("capitulos", []):
        subs = cap.get("subcapitulos", []) or []
        all_q = []
        for s in subs:
            all_q.extend(s.get("preguntas", []) or [])
        if len(subs) == 0:
            n_parts = min_sub
            chunk = max(1, len(all_q) // n_parts) if all_q else 0
            new_subs, pos = [], 0
            for i in range(n_parts):
                chunk_q = all_q[pos:pos + chunk] if all_q else []
                pos += chunk
                new_subs.append({"titulo": f"Bloque {i + 1}", "preguntas": chunk_q})
            cap["subcapitulos"] = new_subs
        elif len(subs) == 1:
            target = 3 if len(all_q) > 15 else 2
            target = max(min_sub, min(max_sub, target))
            chunk = max(1, (len(all_q) + target - 1) // target)
            new_subs, pos = [], 0
            base_title = subs[0].get('titulo', 'Bloque').split("·")[0].strip()
            for i in range(target):
                chunk_q = all_q[pos:pos + chunk]
                pos += chunk
                new_subs.append({"titulo": f"{base_title} · Parte {i + 1}", "preguntas": chunk_q})
            cap["subcapitulos"] = new_subs
        elif len(subs) > max_sub:
            keep = subs[:max_sub - 1]
            merged = {"titulo": f"{subs[max_sub - 1].get('titulo','Bloque')} · Combinado", "preguntas": []}
            for s in subs[max_sub - 1:]:
                merged["preguntas"].extend(s.get("preguntas", []) or [])
            cap["subcapitulos"] = keep + [merged]
    return matrix

def _sanitize_filename(name: Optional[str], default: str) -> str:
    base = (name or default).strip()
    base = re.sub(r'[^\w\-. ]+', '_', base)
    return base or default

# =========================
# Endpoints
# =========================
@app.get("/")
def root():
    return {"status": "ok", "service": "dipli-matrix-generator"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/generate_matrix")
async def generate_matrix(
    request: Request,
    contexto_objetivos: UploadFile = File(...),     # <-- UN SOLO PDF (contexto+objetivos)
    transcripcion: UploadFile = File(...),          # <-- TRANSCRIPCIÓN OBLIGATORIA
    guia: UploadFile | None = File(None),           # <-- GUÍA OPCIONAL
    num_questions: int | None = Form(None),         # backup si no se detectan preguntas y no hay guía
    filename_base: str | None = Form(None),
):
    # Límite de subida (aprox por content-length)
    try:
        content_len = int(request.headers.get("content-length", "0"))
        if content_len and content_len > MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"Payload demasiado grande (> {MAX_UPLOAD_MB} MB).")
    except ValueError:
        pass

    start_dt = datetime.utcnow()

    try:
        co_bytes = await contexto_objetivos.read()
        tr_bytes = await transcripcion.read()
        guia_bytes = await guia.read() if guia is not None else None

        if not contexto_objetivos.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Se requiere PDF combinado de CONTEXTO+OBJETIVOS.")
        if not transcripcion.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Se requiere PDF de TRANSCRIPCIÓN.")
        if guia_bytes is not None and guia and (not guia.filename.lower().endswith(".pdf")):
            raise HTTPException(status_code=400, detail="La GUÍA debe ser PDF.")

        # 1) Extraer texto (con límites de páginas)
        co_raw = extract_text_from_pdf(co_bytes, max_pages=MAX_PAGES_CO)
        tr_raw = extract_text_from_pdf(tr_bytes, max_pages=MAX_PAGES_TR)
        guide_raw = extract_text_from_pdf(guia_bytes, max_pages=MAX_PAGES_GUIDE) if guia_bytes else ""

        if not co_raw.strip():
            raise HTTPException(status_code=422, detail="El PDF de CONTEXTO+OBJETIVOS no contiene texto extraíble.")
        if not tr_raw.strip():
            raise HTTPException(status_code=422, detail="La TRANSCRIPCIÓN no contiene texto extraíble.")

        # 2) LLM (resúmenes compactados)
        llm = LLMClient(model=None)
        co_sum = hierarchical_summarize(llm, co_raw, "Contexto+Objetivos")
        obj_sum = co_sum  # si luego quieres separar, se puede refinar

        # 3) Preguntas desde GUÍA (si hay) y desde TRANSCRIPCIÓN (siempre)
        preguntas_heur = extract_questions_from_text_v2(guide_raw) if guide_raw else []
        preguntas_heur_tr = extract_questions_from_text_v2(tr_raw)
        preguntas_llm_tr = extract_questions_from_transcript_llm(llm, tr_raw)

        # Unir y deduplicar
        all_q: List[str] = []
        for q in (preguntas_heur + preguntas_heur_tr + preguntas_llm_tr):
            if isinstance(q, str) and q.strip():
                all_q.append(q.strip())

        def _dedupe_keep_order(items: List[str]) -> List[str]:
            seen, out = set(), []
            for it in items:
                k = re.sub(r"[\s\?¿\.,;:()\-]+", " ", it.casefold()).strip()
                k = k.replace("á","a").replace("é","e").replace("í","i").replace("ó","o").replace("ú","u").replace("ñ","n")
                if k in seen:
                    continue
                seen.add(k); out.append(it)
            return out

        preguntas = _dedupe_keep_order(all_q)

        # 4) Generar matriz con LLM
        has_guide = bool(guia_bytes and guia and guia.filename)
        n_if_no_guide = None
        if not has_guide and not preguntas:
            # no hay guía y no se detectaron preguntas (ni en transcripción)
            # clamp 40..120; por defecto 60
            n = 60 if (num_questions is None) else int(num_questions)
            n_if_no_guide = max(40, min(120, n))

        matrix = build_matrix_with_llm(llm, co_sum, obj_sum, preguntas, n_if_no_guide or 60)

        # 5) Cobertura total
        if preguntas:
            ya = {_norm_light(x) for x in _collect_assigned_questions(matrix)}
            faltantes = [q for q in preguntas if _norm_light(q) not in ya]
            if faltantes:
                matrix = assign_missing_with_llm(llm, matrix, faltantes, co_sum, obj_sum)
        matrix = enforce_subchapter_limits(matrix, min_sub=2, max_sub=3)

        # 6) Excel
        df = to_dataframe(matrix)
        if df.empty:
            raise HTTPException(status_code=500, detail="La matriz resultó vacía.")
        out_xlsx = io.BytesIO()
        df.to_excel(out_xlsx, index=False, engine="openpyxl")
        out_xlsx.seek(0)
        excel_bytes = out_xlsx.getvalue()

        # 7) Métricas
        end_dt = datetime.utcnow()
        duration_min = (end_dt - start_dt).total_seconds() / 60.0
        used_questions = len(_collect_assigned_questions(matrix))
        metrics_txt = (
            "=== METRICS ===\n"
            f"Start (UTC): {start_dt.isoformat()}Z\n"
            f"End   (UTC): {end_dt.isoformat()}Z\n"
            f"Duration min: {duration_min:.2f}\n"
            f"Detected questions (guia): {len(preguntas_heur) if preguntas_heur else 0}\n"
            f"Detected questions (transcripcion-heur): {len(preguntas_heur_tr)}\n"
            f"Detected questions (transcripcion-LLM): {len(preguntas_llm_tr)}\n"
            f"Assigned questions (final): {used_questions}\n"
            f"Prompt tokens: {llm.prompt_tokens}\n"
            f"Completion tokens: {llm.completion_tokens}\n"
            f"Total tokens: {llm.total_tokens}\n"
            f"Cost USD (est): {llm.cost_usd():.6f}\n"
        )

        # 8) ZIP (xlsx + metrics.txt)
        base_name = _sanitize_filename(filename_base, "matriz")
        zip_name = f"{base_name}.zip"
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr(f"{base_name}.xlsx", excel_bytes)
            z.writestr("metrics.txt", metrics_txt.encode("utf-8"))
        buf.seek(0)
        zip_bytes = buf.getvalue()

        token = register_download(zip_bytes, zip_name, ZIP_MEDIA_TYPE)
        base_url = str(request.base_url).rstrip("/")
        return JSONResponse(
            {
                "download_url": f"{base_url}/download/{token}",
                "filename": zip_name,
                "expires_in_seconds": DOWNLOAD_TTL_SECS,
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando matriz: {e}")

@app.get("/download/{token}")
def download_token(token: str):
    cleanup_downloads()
    item = DOWNLOADS.get(token)
    if not item:
        raise HTTPException(status_code=404, detail="Link expirado o inválido")
    data, filename, media_type, exp = item
    if exp <= datetime.utcnow():
        with _DL_LOCK:
            DOWNLOADS.pop(token, None)
        raise HTTPException(status_code=410, detail="Link expirado")
    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
        "Cache-Control": "no-store",
    }
    return StreamingResponse(io.BytesIO(data), media_type=media_type, headers=headers)
