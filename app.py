# app.py
import os
import re
import io
import json
import secrets
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

import pdfplumber
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential_jitter
from dotenv import load_dotenv

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# =========================
# CONFIGURACIÓN GENERAL
# =========================
app = FastAPI(title="Generador de Matrices – Dipli")

ALLOWED_ORIGINS = [
    "https://www.dipli.ai",
    "https://dipli.ai",
    "https://isagarcivill09.wixsite.com/turop",
    "https://isagarcivill09.wixsite.com/turop/tienda",
    "https://isagarcivill09-wixsite-com.filesusr.com",
    "https://www.dipli.ai/preparaci%C3%B3n",
    "https://www-dipli-ai.filesusr.com",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_origin_regex=r"https://.*\.filesusr\.com",
)

# =========================
# DOWNLOADS TEMPORALES
# =========================
DOWNLOADS: Dict[str, Any] = {}
TTL_MINUTES = 30

def cleanup_downloads():
    now = datetime.utcnow()
    expired = [k for k, (_, _, _, exp) in DOWNLOADS.items() if exp <= now]
    for k in expired:
        DOWNLOADS.pop(k, None)

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
# FUNCIONES BASE
# =========================
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

def extract_text_from_pdf_bytes(data: bytes) -> str:
    texts = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            texts.append(txt)
    return "\n".join(texts)


# =========================
# CLIENTE OPENAI (con conteo de tokens)
# =========================
class LLMClient:
    def __init__(self, model: Optional[str], api_key_override: Optional[str] = None):
        load_dotenv()
        if OpenAI is None:
            raise RuntimeError("Falta la librería openai>=1.30.0. Instálala con: pip install openai")

        api_key = api_key_override or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY no configurada. Pásala en el form-data (openai_api_key) o variable de entorno.")
        os.environ["OPENAI_API_KEY"] = api_key

        self.model = (model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip()
        self.client = OpenAI()
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def _accumulate_usage(self, resp):
        try:
            u = getattr(resp, "usage", None)
            if not u:
                return
            if hasattr(u, "total_tokens"):
                self.total_tokens += int(getattr(u, "total_tokens", 0) or 0)
                self.prompt_tokens += int(getattr(u, "prompt_tokens", 0) or 0)
                self.completion_tokens += int(getattr(u, "completion_tokens", 0) or 0)
            elif isinstance(u, dict):
                self.total_tokens += int(u.get("total_tokens") or 0)
                self.prompt_tokens += int(u.get("prompt_tokens") or 0)
                self.completion_tokens += int(u.get("completion_tokens") or 0)
        except Exception:
            pass

    @retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(1, 8))
    def chat(self, system: str, user: str, temperature: float = 0.2) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        self._accumulate_usage(resp)
        return resp.choices[0].message.content

    @retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(1, 8))
    def chat_json(self, system: str, user: str, temperature: float = 0.1) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        self._accumulate_usage(resp)
        return resp.choices[0].message.content


# =========================
# RESUMEN JERÁRQUICO
# =========================
def hierarchical_summarize(llm: LLMClient, raw_text: str, label: str) -> str:
    chunks = split_into_chunks(raw_text)
    summaries = []
    sys = "Eres un analista senior. Resume con foco en actores, procesos, restricciones y términos clave. Usa viñetas concisas."
    for i, ch in enumerate(chunks, 1):
        user = f"Resumen parcial {label} ({i}/{len(chunks)}):\n\n{ch}"
        summaries.append(llm.chat(sys, user))
    combined = "\n\n".join(f"- Bloque {i+1}: {s}" for i, s in enumerate(summaries))
    final = llm.chat(sys, f"Fusiona y depura en 12–18 viñetas claras:\n\n{combined}")
    return final


# =========================
# EXTRACCIÓN DE PREGUNTAS (VERSIÓN AVANZADA)
# =========================
_INTERROGATIVOS = (
    r"(qué|como|cómo|cual|cuál|cuando|cuándo|donde|dónde|quien|quién|por qué|para qué|cuanto|cuánto|cuales|cuáles)"
)

def _clean_line(s: str) -> str:
    s = s.strip("•-· ").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _is_questionish(s: str) -> bool:
    """Detecta si una línea suena a pregunta aunque no tenga signos."""
    if "?" in s or "¿" in s:
        return True
    if re.match(rf"^\s*{_INTERROGATIVOS}\b", s, re.IGNORECASE):
        return True
    if re.match(r"^\s*(cuéntame|háblame|dime|menciona|explícame|relata|descríbeme|cómo fue|qué pasó|qué sintió|qué pensó|por favor cuéntame)", s, re.IGNORECASE):
        return True
    if re.match(r"^\s*(piensa|recuerda|comparte|cuenta|mencione|narre|indique|explique|detalla|describe)", s, re.IGNORECASE):
        return True
    return False

def extract_questions_from_text_v3(text: str) -> List[str]:
    """
    Extrae preguntas incluso si no tienen signo de interrogación,
    basándose en tono y estructura gramatical.
    """
    if not text:
        return []
    lines = [_clean_line(l) for l in text.splitlines() if l.strip()]
    candidates = []
    buffer = []
    for l in lines:
        if _is_questionish(l):
            if buffer:
                candidates.append(" ".join(buffer))
                buffer = []
            buffer.append(l)
        elif buffer:
            if len(l.split()) < 20 and not re.match(r"^[A-ZÁÉÍÓÚÑ]{2,}$", l):
                buffer.append(l)
            else:
                candidates.append(" ".join(buffer))
                buffer = []
    if buffer:
        candidates.append(" ".join(buffer))

    # Limpieza y deduplicado
    seen = set()
    out = []
    for q in candidates:
        q = q.strip()
        if len(q) < 4:
            continue
        norm = re.sub(r"[\W_]+", "", q.lower())
        if norm not in seen:
            seen.add(norm)
            if not q.endswith("?"):
                q += "?"
            out.append(q)
    return out


# =========================
# CONSTRUCCIÓN MATRIZ
# =========================
def build_matrix_with_llm(llm: LLMClient, contexto_sum: str, objetivos_sum: str, preguntas: Optional[List[str]]) -> Dict[str, Any]:
    sys = "Eres Research Lead. Diseñas guías cualitativas limpias y estructuradas."
    if preguntas:
        pregunta_blob = "\n".join(f"- {p}" for p in preguntas)
        user = (
            f"CONTEXTO:\n{contexto_sum}\n\n"
            f"OBJETIVOS:\n{objetivos_sum}\n\n"
            f"GUIA:\n{pregunta_blob}\n\n"
            "Devuelve JSON con capítulos, subcapítulos y preguntas."
        )
    else:
        user = (
            f"CONTEXTO:\n{contexto_sum}\n\n"
            f"OBJETIVOS:\n{objetivos_sum}\n\n"
            "Genera estructura analítica JSON (capítulos y subcapítulos)."
        )
    raw = llm.chat_json(sys, user)
    return json.loads(raw)

def to_dataframe(matrix: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for c in matrix.get("capitulos", []):
        cap = (c.get("titulo") or "").strip()
        for s in c.get("subcapitulos", []):
            sub = (s.get("titulo") or "").strip()
            for q in s.get("preguntas", []):
                rows.append({"Capitulo": cap, "Subcapitulo": sub, "Preguntas": q})
    return pd.DataFrame(rows)


# =========================
# ENDPOINT PRINCIPAL
# =========================
@app.post("/generate-matrix")
async def generate_matrix(
    contexto: UploadFile = File(...),
    objetivos: UploadFile = File(...),
    guia: Optional[UploadFile] = File(None),
    openai_api_key: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
    filename_base: Optional[str] = Form("matriz.xlsx"),
):
    cleanup_downloads()

    ctx_bytes = await contexto.read()
    obj_bytes = await objetivos.read()
    guia_bytes = await guia.read() if guia else None

    llm = LLMClient(model=model, api_key_override=openai_api_key)

    # Resúmenes
    ctx_raw = extract_text_from_pdf_bytes(ctx_bytes)
    obj_raw = extract_text_from_pdf_bytes(obj_bytes)
    ctx_sum = hierarchical_summarize(llm, ctx_raw, "Contexto")
    obj_sum = hierarchical_summarize(llm, obj_raw, "Objetivos")

    # Preguntas detectadas (v3)
    preguntas = extract_questions_from_text_v3(extract_text_from_pdf_bytes(guia_bytes)) if guia_bytes else []

    # Matriz
    matrix = build_matrix_with_llm(llm, ctx_sum, obj_sum, preguntas)
    df = to_dataframe(matrix)
    if df.empty:
        raise HTTPException(status_code=500, detail="Matriz vacía")

    # Excel en memoria
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Matriz", index=False)
        pd.DataFrame(
            [
                ["Preguntas detectadas", len(preguntas)],
                ["Tokens totales", llm.total_tokens],
                ["Prompt tokens", llm.prompt_tokens],
                ["Completion tokens", llm.completion_tokens],
            ],
            columns=["Métrica", "Valor"],
        ).to_excel(writer, sheet_name="Métricas", index=False)
    output.seek(0)

    # Registrar para descarga temporal
    token = secrets.token_urlsafe(16)
    filename = filename_base if filename_base.endswith(".xlsx") else f"{filename_base}.xlsx"
    DOWNLOADS[token] = (
        output.getvalue(),
        filename,
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        datetime.utcnow() + timedelta(minutes=TTL_MINUTES),
    )

    return JSONResponse(
        {
            "status": "ok",
            "download_url": f"/download/{token}",
            "filename": filename,
            "detected_questions": len(preguntas),
            "tokens": {
                "total": llm.total_tokens,
                "prompt": llm.prompt_tokens,
                "completion": llm.completion_tokens,
            },
            "expires_in_minutes": TTL_MINUTES,
        }
    )


# =========================
# HEALTHCHECK
# =========================
@app.get("/healthz")
def health():
    cleanup_downloads()
    return {"ok": True}
