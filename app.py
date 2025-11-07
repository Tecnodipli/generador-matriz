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
# Config (sin .env)
# =========================
MAX_UPLOAD_MB = 60
MAX_PAGES_CO = 30     # máx páginas de Contexto+Objetivos
MAX_PAGES_TR = 60     # máx páginas de Transcripción (solo contexto)
MAX_PAGES_GUIDE = 20  # máx páginas de Guía (opcional)

LLM_MAX_CHUNKS = 6
LLM_CHUNK_SIZE = 9000

INPUT_COST = float(os.getenv("OPENAI_INPUT_COST_PER_1K", "0.0"))
OUTPUT_COST = float(os.getenv("OPENAI_OUTPUT_COST_PER_1K", "0.0"))

# =========================
# Utilidades PDF / Texto
# =========================
def extract_text_from_pdf(path_or_bytes, max_pages: int | None = None) -> str:
    texts = []
    if isinstance(path_or_bytes, (bytes, bytearray)):
        bio = io.BytesIO(path_or_bytes)
        with pdfplumber.open(bio) as pdf:
            for idx, page in enumerate(pdf.pages):
                if max_pages is not None and idx >= max_pages:
                    break
                texts.append(page.extract_text() or "")
    else:
        with pdfplumber.open(path_or_bytes) as pdf:
            for idx, page in enumerate(pdf.pages):
                if max_pages is not None and idx >= max_pages:
                    break
                texts.append(page.extract_text() or "")
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
    raw_chunks = split_into_chunks(text, max_chars=max_chars)
    if len(raw_chunks) <= max_chunks:
        return raw_chunks
    head = raw_chunks[:max_chunks - 1]
    tail = "\n".join(raw_chunks[max_chunks - 1:])
    return head + [tail]

# =========================
# Cliente LLM
# =========================
class LLMClient:
    def __init__(self, model: Optional[str] = None):
        if OpenAI is None:
            raise RuntimeError("Falta openai>=2.x. Instala la dependencia.")
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("Define la variable de entorno OPENAI_API_KEY en tu hosting.")
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
        "restricciones y términos clave. Usa viñetas concisas."
    )
    for i, ch in enumerate(chunks, 1):
        user = f"Resumen parcial {label} ({i}/{len(chunks)}):\n\n{ch}"
        summaries.append(llm.chat(sys, user, temperature=0.0))
    combined = "\n\n".join(f"- Bloque {i+1}: {s}" for i, s in enumerate(summaries))
    final_user = f"Fusiona y depura en 12–18 viñetas claras, sin redundancias:\n\n{combined}"
    final = llm.chat(sys, final_user, temperature=0.0)
    return final

# =========================
# Extracción de preguntas SOLO desde GUÍA (SIN deduplicar)
# =========================
_INTERROGATIVOS = r"(qué|como|cómo|cual|cuál|cuando|cuándo|donde|dónde|quien|quién|por qué|para qué|cuanto|cuánto|cuales|cuáles)"
_IS_INSTR = re.compile(r"^\s*(moderador:|se graba|\(se graba\))", re.IGNORECASE)

def _clean_line(s: str) -> str:
    s = s.strip().strip("•").strip("-").strip("·").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _is_questionish(s: str) -> bool:
    if "?" in s or "¿" in s:
        return True
    return re.match(rf"^\s*{_INTERROGATIVOS}\b", s, re.IGNORECASE) is not None

def extract_questions_from_guide(text: str) -> List[str]:
    """Extrae preguntas de la GUÍA manteniendo el ORDEN y SIN deduplicar."""
    if not text:
        return []
    lines = [_clean_line(l) for l in text.splitlines()]
    lines = [l for l in lines if l and not _IS_INSTR.match(l)]
    out = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if _is_questionish(ln):
            buf = [ln]
            j = i + 1
            while j < len(lines):
                nxt = lines[j]
                if _is_questionish(nxt):
                    break
                if re.match(r"^[A-ZÁÉÍÓÚÑ].{0,80}$", nxt):  # encabezado
                    break
                buf.append(nxt)
                j += 1
            q = " ".join(buf).strip()
            if re.match(rf"^{_INTERROGATIVOS}\b", q, re.IGNORECASE) and not q.endswith("?"):
                q += "?"
            out.append(q)
            i = j
        else:
            i += 1
    return out  # ¡sin dedup!

# =========================
# Instrucciones DINÁMICAS (dos pasos cuando hay guía)
# =========================
STRUCTURE_ONLY_PROMPT = """
Devuelve SOLO un objeto JSON con la estructura:
{
  "capitulos": [
    { "titulo": "string", "subcapitulos": [ { "titulo": "string" }, ... ] }
  ]
}
REGLAS:
- Propón capítulos y subcapítulos DINÁMICOS a tono de informe, basados en CONTEXTO y TRANSCRIPCIÓN (esta última sólo como referencia de tono/secuencia).
- No agregues campo "preguntas" en este paso.
- Títulos breves (≤ 70 caracteres), informativos y sin jerga innecesaria.
"""

CLASSIFY_PROMPT = """
Tienes una estructura de capítulos/subcapítulos y una lista de PREGUNTAS (de una GUÍA). 
Asigna CADA pregunta EXACTAMENTE a un (capítulo, subcapítulo) de la estructura SIN modificar el texto ni el conteo.
Si alguna no encaja, colócala en "Preguntas sin asignar" dentro del primer capítulo (créalo si no existe).
Devuelve JSON con:
{
  "capitulos": [
    { "titulo": "string", "subcapitulos": [ { "titulo": "string", "preguntas": ["..."] } ] }
  ]
}
No inventes preguntas nuevas.
"""

MATRIX_INSTRUCTIONS_DYNAMIC = """
Estructura objetivo:
{
  "capitulos": [
    {
      "titulo": "string",
      "subcapitulos": [
        {"titulo": "string", "preguntas": ["string", ...]}
      ]
    }
  ]
}
REGLAS:
- CAPÍTULOS y SUBCAPÍTULOS DINÁMICOS, tono de informe.
- La TRANSCRIPCIÓN se usa SOLO como CONTEXTO (no extraigas diálogos).
- Si NO hay GUÍA: genera EXACTAMENTE N preguntas, claras y neutrales.
- Mantén el orden en lo posible.
- 2–5 subcapítulos por capítulo es un rango razonable; ajusta según contenido.
- Títulos ≤ 70 caracteres.
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

# ---------- Paso A (estructura) + Paso B (clasificación) cuando HAY guía ----------
def build_structure_only(llm: LLMClient, co_sum: str, transcript_sum: str) -> Dict[str, Any]:
    sys = "Eres Research Lead. Defines capítulos y subcapítulos informativos."
    user = f"""CONTEXTO (resumen):
{co_sum}

TRANSCRIPCIÓN (resumen como contexto; no extraer diálogos):
{transcript_sum}

{STRUCTURE_ONLY_PROMPT}
"""
    raw = llm.chat_json(sys, user, temperature=0.15)
    data = safe_json_loads(raw)
    # normaliza forma: subcapitulos deben ser objetos {"titulo": "..."}
    caps = []
    for cap in data.get("capitulos", []):
        title = (cap.get("titulo") or "").strip() or "Capítulo 1"
        subs = cap.get("subcapitulos", []) or []
        norm_subs = []
        for s in subs:
            if isinstance(s, dict):
                t = (s.get("titulo") or "").strip() or "Bloque"
            else:
                t = str(s).strip() or "Bloque"
            norm_subs.append({"titulo": t})
        if not norm_subs:
            norm_subs = [{"titulo": "Bloque 1"}]
        caps.append({"titulo": title, "subcapitulos": norm_subs})
    if not caps:
        caps = [{"titulo": "Capítulo 1", "subcapitulos": [{"titulo": "Bloque 1"}]}]
    return {"capitulos": caps}

def classify_questions_into_structure(
    llm: LLMClient,
    structure: Dict[str, Any],
    questions: List[str],
    co_sum: str,
    transcript_sum: str,
) -> Dict[str, Any]:
    sys = "Eres Research Lead. Clasificas preguntas en capítulos/subcapítulos dados sin cambiar el texto."
    struct_min = json.dumps(structure, ensure_ascii=False)
    qs_blob = "\n".join(f"- {q}" for q in questions)
    user = f"""ESTRUCTURA:
{struct_min}

PREGUNTAS (usa exactamente estas; no inventes ni edites):
{qs_blob}

Referencia (contexto):
- CONTEXTO: {co_sum[:900]}
- TRANSCRIPCIÓN: {transcript_sum[:900]}

{CLASSIFY_PROMPT}
"""
    raw = llm.chat_json(sys, user, temperature=0.1)
    data = safe_json_loads(raw)

    # seguridad: aseguro que TODAS las preguntas estén en la salida
    assigned = []
    for cap in data.get("capitulos", []):
        for sub in cap.get("subcapitulos", []) or []:
            for q in sub.get("preguntas", []) or []:
                assigned.append(q)

    # Preguntas faltantes -> agregarlas a "Preguntas sin asignar"
    missing = [q for q in questions if q not in assigned]

    if missing:
        # busca primer capítulo / subcapítulo "Preguntas sin asignar"; si no existe, créalo
        if not data.get("capitulos"):
            data["capitulos"] = [{"titulo": "Capítulo 1", "subcapitulos": []}]
        first_cap = data["capitulos"][0]
        found_bucket = None
        for sub in first_cap.get("subcapitulos", []):
            if (sub.get("titulo") or "").strip().lower() == "preguntas sin asignar":
                found_bucket = sub
                break
        if not found_bucket:
            found_bucket = {"titulo": "Preguntas sin asignar", "preguntas": []}
            first_cap.setdefault("subcapitulos", []).append(found_bucket)
        found_bucket.setdefault("preguntas", []).extend(missing)

    return data

# ---------- Caso sin guía (generación dinámica completa) ----------
def build_matrix_no_guide(
    llm: LLMClient,
    co_sum: str,
    transcript_sum: str,
    n_questions: int,
) -> Dict[str, Any]:
    sys = "Eres Research Lead experto en cualitativo. Diseñas estructuras de INFORME y generas preguntas."
    user = f"""CONTEXTO (resumen):
{co_sum}

TRANSCRIPCIÓN (resumen como contexto; no extraer diálogos):
{transcript_sum}

NO hay guía. Genera EXACTAMENTE {n_questions} preguntas y organízalas en capítulos/subcapítulos.
{MATRIX_INSTRUCTIONS_DYNAMIC}
"""
    raw = llm.chat_json(sys, user, temperature=0.15)
    return safe_json_loads(raw)

def to_dataframe(matrix: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for cap in matrix.get("capitulos", []):
        cap_title = (cap.get("titulo") or "").strip()
        for sub in cap.get("subcapitulos", []) or []:
            sub_title = (sub.get("titulo") or "").strip()
            for q in sub.get("preguntas", []) or []:
                rows.append({"Capitulo": cap_title, "Subcapitulo": sub_title, "Preguntas": (q or "").strip()})
    return pd.DataFrame(rows, columns=["Capitulo", "Subcapitulo", "Preguntas"])

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
    contexto_objetivos: UploadFile = File(...),     # UN SOLO PDF (contexto+objetivos)
    transcripcion: UploadFile = File(...),          # TRANSCRIPCIÓN (contexto; NO extrae preguntas)
    guia: UploadFile | None = File(None),           # GUÍA (opcional)
    num_questions: int | None = Form(None),         # requerido SOLO si no hay guía (40..120; default 60)
    filename_base: str | None = Form(None),
):
    # Límite de carga
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

        # 2) LLM: resúmenes (transcripción solo como contexto)
        llm = LLMClient(model=None)
        co_sum = hierarchical_summarize(llm, co_raw, "Contexto+Objetivos")
        tr_sum = hierarchical_summarize(llm, tr_raw, "Transcripción (contexto)")

        # 3) ¿Hay guía?
        preguntas: Optional[List[str]] = None
        if guide_raw:
            preguntas = extract_questions_from_guide(guide_raw)  # SIN dedup
        has_guide = bool(preguntas and len(preguntas) > 0)

        # 4) Construcción de la matriz
        if has_guide:
            # Paso A: estructura sin preguntas
            structure = build_structure_only(llm, co_sum, tr_sum)
            # Paso B: clasificar EXACTAMENTE las preguntas de la guía en esa estructura
            matrix = classify_questions_into_structure(llm, structure, preguntas, co_sum, tr_sum)
        else:
            n = 60 if (num_questions is None) else int(num_questions)
            n = max(40, min(120, n))
            matrix = build_matrix_no_guide(llm, co_sum, tr_sum, n_questions=n)

        # 5) Excel
        df = to_dataframe(matrix)
        if df.empty:
            raise HTTPException(status_code=500, detail="La matriz resultó vacía.")
        out_xlsx = io.BytesIO()
        df.to_excel(out_xlsx, index=False, engine="openpyxl")
        out_xlsx.seek(0)
        excel_bytes = out_xlsx.getvalue()

        # 6) Métricas
        end_dt = datetime.utcnow()
        duration_min = (end_dt - start_dt).total_seconds() / 60.0
        used_questions = len(df.index)
        metrics_txt = (
            "=== METRICS ===\n"
            f"Start (UTC): {start_dt.isoformat()}Z\n"
            f"End   (UTC): {end_dt.isoformat()}Z\n"
            f"Duration min: {duration_min:.2f}\n"
            f"Guide questions (if any): {len(preguntas) if has_guide else 0}\n"
            f"Assigned questions (final): {used_questions}\n"
            f"Prompt tokens: {llm.prompt_tokens}\n"
            f"Completion tokens: {llm.completion_tokens}\n"
            f"Total tokens: {llm.total_tokens}\n"
            f"Cost USD (est): {llm.cost_usd():.6f}\n"
        )

        # 7) ZIP (xlsx + metrics.txt)
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
