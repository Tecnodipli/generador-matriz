import os
import io
import re
import json
import secrets
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

import pdfplumber
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

# =========================
# CORS (ajusta dominios según tu Wix)
# =========================
ALLOWED_ORIGINS = [
    "https://www.dipli.ai",
    "https://dipli.ai",
    "https://isagarcivill09.wixsite.com",
    "https://isagarcivill09.wixsite.com/turop",
    "https://isagarcivill09.wixsite.com/turop/tienda",
    "https://isagarcivill09-wixsite-com.filesusr.com",
    "https://www-dipli-ai.filesusr.com",
]

app = FastAPI(title="Generador de Matriz Dipli API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_origin_regex=r"https://.*\.filesusr\.com",
)

# =========================
# OpenAI
# =========================
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


class LLMClient:
    def __init__(self, model: Optional[str]):
        load_dotenv()
        if OpenAI is None:
            raise RuntimeError("Falta la librería openai>=1.30.0. Instálala con: pip install openai")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Debes definir OPENAI_API_KEY en entorno o .env")

        self.model = (model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip()
        self.client = OpenAI(api_key=api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(1, 8))
    def chat(self, system: str, user: str, temperature: float = 0.2) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content

    @retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(1, 8))
    def chat_json(self, system: str, user: str, temperature: float = 0.1) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content


# =========================
# Utilidades de PDF / texto
# =========================
def extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
    if not file_bytes:
        return ""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        return "\n".join((page.extract_text() or "") for page in pdf.pages)


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


def hierarchical_summarize(llm: LLMClient, raw_text: str, label: str) -> str:
    chunks = split_into_chunks(raw_text)
    summaries = []
    sys = ("Eres un analista senior. Resume con foco en: actores, procesos, definiciones, "
           "restricciones y términos clave. Usa viñetas concisas.")
    for i, ch in enumerate(chunks, 1):
        user = f"Resumen parcial {label} ({i}/{len(chunks)}):\n\n{ch}"
        summaries.append(llm.chat(sys, user, temperature=0.0))
    combined = "\n\n".join(f"- Bloque {i+1}: {s}" for i, s in enumerate(summaries))
    return llm.chat(sys, f"Fusiona y depura en 12–18 viñetas claras, sin redundancias:\n\n{combined}", temperature=0.0)


def extract_questions_from_text(text: str) -> List[str]:
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    q = []
    for l in lines:
        if l.endswith("?") or l.startswith("¿") or re.search(r"\?\s*$", l):
            q.append(l)
    out, seen = [], set()
    for s in q:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


# =========================
# Generación de matriz
# =========================
MATRIX_INSTRUCTIONS = """
Devuelve un JSON válido con esta estructura:
{
  "capitulos": [
    {
      "titulo": "string",
      "subcapitulos": [
        {
          "titulo": "string",
          "preguntas": ["string", ...]
        }
      ]
    }
  ]
}
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
        blk = extract_json_block(s)
        if blk:
            return json.loads(blk)
        raise


def build_matrix_with_llm(llm: LLMClient, contexto_sum: str, objetivos_sum: str, preguntas: Optional[List[str]]) -> Dict[str, Any]:
    sys = "Eres un Research Lead experto en cualitativo. Diseñas guías y estructuras analíticas limpias y accionables."
    if preguntas:
        pregunta_blob = "\n".join(f"- {p}" for p in preguntas)
        user = f"""CONTEXTO (resumen):
{contexto_sum}

OBJETIVOS (resumen):
{objetivos_sum}

GUIA (preguntas originales, no edites el texto):
{pregunta_blob}

{MATRIX_INSTRUCTIONS}
"""
    else:
        user = f"""CONTEXTO (resumen):
{contexto_sum}

OBJETIVOS (resumen):
{objetivos_sum}

NO se proporcionó guía. Debes proponer capítulos, subcapítulos y preguntas.
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


# =========================
# Descarga por token (memoria)
# =========================
TOKEN_TTL_SECONDS = int(os.getenv("TOKEN_TTL_SECONDS", "1800"))  # 30 min por defecto
DOWNLOADS: Dict[str, tuple[bytes, str, str, datetime]] = {}


def cleanup_downloads():
    """Elimina entradas expiradas del almacenamiento en memoria."""
    now = datetime.utcnow()
    expired = [k for k, (_, _, _, exp) in DOWNLOADS.items() if exp <= now]
    for k in expired:
        DOWNLOADS.pop(k, None)


def push_download(data: bytes, filename: str, media_type: str) -> str:
    """Crea un token y guarda en memoria el binario + metadatos."""
    cleanup_downloads()
    token = secrets.token_urlsafe(16)
    exp = datetime.utcnow() + timedelta(seconds=TOKEN_TTL_SECONDS)
    DOWNLOADS[token] = (data, filename, media_type, exp)
    return token


@app.get("/download/{token}")
def download_token(token: str):
    """Entrega el archivo asociado a un token válido"""
    cleanup_downloads()
    item = DOWNLOADS.get(token)
    if not item:
        raise HTTPException(status_code=404, detail="Link expirado o inválido")
    data, filename, media_type, exp = item
    if exp <= datetime.utcnow():
        DOWNLOADS.pop(token, None)
        raise HTTPException(status_code=410, detail="Link expirado")
    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
        "Cache-Control": "no-store",
    }
    return StreamingResponse(io.BytesIO(data), media_type=media_type, headers=headers)


# =========================
# Endpoints
# =========================
@app.get("/health")
def health():
    return {"status": "ok", "utc": datetime.utcnow().isoformat()}

@app.post("/generate_matrix")
async def generate_matrix(
    request: Request,
    contexto: UploadFile = File(...),
    objetivos: UploadFile = File(...),
    guia: Optional[UploadFile] = None,
    model: Optional[str] = "gpt-4o-mini",
):
    """
    Sube CONTEXTO.pdf, OBJETIVOS.pdf y opcionalmente GUIA.pdf.
    Respuesta: JSON con { download_url, token, expires_at } para descargar matriz.xlsx
    """
    try:
        contexto_bytes = await contexto.read()
        objetivos_bytes = await objetivos.read()
        guia_bytes = await guia.read() if guia else None

        llm = LLMClient(model)

        # Resúmenes
        ctx_text = extract_text_from_pdf_bytes(contexto_bytes)
        obj_text = extract_text_from_pdf_bytes(objetivos_bytes)
        if not ctx_text.strip():
            raise HTTPException(status_code=400, detail="El CONTEXTO no contiene texto detectable.")
        if not obj_text.strip():
            raise HTTPException(status_code=400, detail="Los OBJETIVOS no contienen texto detectable.")

        ctx_sum = hierarchical_summarize(llm, ctx_text, "Contexto")
        obj_sum = hierarchical_summarize(llm, obj_text, "Objetivos")

        # Preguntas opcionales desde GUIA
        preguntas = extract_questions_from_text(extract_text_from_pdf_bytes(guia_bytes)) if guia_bytes else None

        # Matriz
        matrix = build_matrix_with_llm(llm, ctx_sum, obj_sum, preguntas)
        df = to_dataframe(matrix)
        if df.empty:
            raise HTTPException(status_code=422, detail="La matriz resultó vacía. Revisa los documentos de entrada.")

        # Excel a memoria + token
        buf = io.BytesIO()
        df.to_excel(buf, index=False)
        data = buf.getvalue()

        filename = "matriz.xlsx"
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        token = push_download(data, filename, media_type)
        exp = DOWNLOADS[token][3].isoformat()

        # URL absoluta basada en la request
        download_url = request.url_for("download_token", token=token)

        return JSONResponse({"download_url": str(download_url), "token": token, "expires_at": exp})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
