import os
import io
import re
import json
import tempfile
import pdfplumber
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from tenacity import retry, stop_after_attempt, wait_exponential_jitter
from dotenv import load_dotenv

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# =====================================================
# CONFIG
# =====================================================
load_dotenv()
app = FastAPI(title="Generador de Matrices - Dipli")

ALLOWED_ORIGINS = [
    "https://www.dipli.ai",
    "https://dipli.ai",
    "https://isagarcivill09.wixsite.com",
    "https://isagarcivill09.wixsite.com/turop",
    "https://isagarcivill09-wixsite-com.filesusr.com",
    "https://www-dipli-ai.filesusr.com",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DOWNLOADS: Dict[str, tuple] = {}
TOKEN_EXPIRATION_MINUTES = 30


# =====================================================
# UTILS
# =====================================================
def cleanup_downloads():
    """Elimina descargas expiradas."""
    now = datetime.utcnow()
    expired = [k for k, v in DOWNLOADS.items() if v[3] <= now]
    for k in expired:
        DOWNLOADS.pop(k, None)


def generate_token() -> str:
    import secrets
    return secrets.token_urlsafe(16)


def extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
    texts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for p in pdf.pages:
            txt = p.extract_text() or ""
            texts.append(txt)
    return "\n".join(texts)


# =====================================================
# LLM CLIENT (con conteo de tokens)
# =====================================================
class LLMClient:
    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        if OpenAI is None:
            raise RuntimeError("Falta la librería openai.")
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("No hay clave de OpenAI.")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def _accumulate_usage(self, resp):
        try:
            u = getattr(resp, "usage", None)
            if not u:
                return
            self.total_tokens += int(getattr(u, "total_tokens", 0) or 0)
            self.prompt_tokens += int(getattr(u, "prompt_tokens", 0) or 0)
            self.completion_tokens += int(getattr(u, "completion_tokens", 0) or 0)
        except Exception:
            pass

    @retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(1, 6))
    def chat(self, system: str, user: str, temperature: float = 0.1) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature,
        )
        self._accumulate_usage(resp)
        return resp.choices[0].message.content

    @retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(1, 6))
    def chat_json(self, system: str, user: str, temperature: float = 0.1) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        self._accumulate_usage(resp)
        return resp.choices[0].message.content


# =====================================================
# EXTRACCIÓN DE PREGUNTAS (INTELIGENTE)
# =====================================================
_INTERROGATIVOS = (
    r"(qué|como|cómo|cual|cuál|cuando|cuándo|donde|dónde|quien|quién|por qué|para qué|cuanto|cuánto|cuales|cuáles)"
)

def _clean_line(s: str) -> str:
    s = s.strip("•-· ").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _is_questionish(s: str) -> bool:
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


# =====================================================
# CONSTRUCCIÓN DE MATRIZ
# =====================================================
MATRIX_INSTRUCTIONS = """
Devuelve un JSON válido con esta estructura EXACTA:
{
  "capitulos": [
    {
      "titulo": "string",
      "subcapitulos": [
        { "titulo": "string", "preguntas": ["string", ...] }
      ]
    }
  ]
}

Reglas:
- Cada capítulo debe tener 2–3 subcapítulos.
- TODAS las preguntas de la GUIA deben incluirse, sin editar su texto.
- No dejes preguntas a nivel del capítulo.
- No devuelvas objetos, solo strings en "preguntas".
- Mantén orden lógico y títulos breves.
"""

def safe_json_loads(s: str) -> Dict[str, Any]:
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`").replace("json\n", "").replace("json\r\n", "")
    try:
        return json.loads(s)
    except Exception:
        match = re.search(r"\{.*\}", s, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ValueError("No se pudo parsear JSON.")


def build_matrix_with_llm(llm: LLMClient, contexto_sum: str, objetivos_sum: str, preguntas: List[str]) -> Dict[str, Any]:
    sys = "Eres un Research Lead experto en investigación cualitativa."
    pregunta_blob = "\n".join(f"- {p}" for p in preguntas)
    user = f"""CONTEXTO:
{contexto_sum}

OBJETIVOS:
{objetivos_sum}

PREGUNTAS DE LA GUÍA:
{pregunta_blob}

{MATRIX_INSTRUCTIONS}
"""
    try:
        raw = llm.chat_json(sys, user)
        return safe_json_loads(raw)
    except Exception:
        raw = llm.chat(sys, user)
        return safe_json_loads(raw)


# =====================================================
# NORMALIZACIÓN DE ESTRUCTURA
# =====================================================
def _question_to_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, (list, tuple)):
        return " ".join(str(t).strip() for t in x if str(t).strip())
    if isinstance(x, dict):
        for k in ("pregunta", "question", "texto", "text", "q", "enunciado"):
            if k in x and isinstance(x[k], (str, list, tuple)):
                return _question_to_str(x[k])
        return str(x)
    return str(x).strip()

def _distribute(lst: List[str], parts: int) -> List[List[str]]:
    n = max(1, parts)
    size = (len(lst) + n - 1) // n
    return [lst[i:i+size] for i in range(0, len(lst), size)]

def normalize_matrix_structure(matrix: Dict[str, Any], preguntas: List[str]) -> Dict[str, Any]:
    caps = matrix.get("capitulos") or []
    if not caps:
        # crear estructura desde preguntas
        parts_caps = _distribute(preguntas, 3)
        caps = []
        for i, cap_q in enumerate(parts_caps, start=1):
            subs = [{"titulo": f"Bloque {j+1}", "preguntas": cap_q[j::3]} for j in range(3)]
            caps.append({"titulo": f"Capítulo {i}", "subcapitulos": subs})
        matrix["capitulos"] = caps

    new_caps = []
    for i, cap in enumerate(caps, start=1):
        ctitle = (cap.get("titulo") or "").strip() or f"Capítulo {i}"
        subs = cap.get("subcapitulos") or []
        if not subs:
            subs = [{"titulo": f"Bloque 1", "preguntas": []}, {"titulo": f"Bloque 2", "preguntas": []}]
        norm_subs = []
        for j, s in enumerate(subs, start=1):
            stitle = (s.get("titulo") or "").strip() or f"Bloque {j}"
            qraw = s.get("preguntas") or []
            qlist = [_question_to_str(q) for q in qraw if _question_to_str(q)]
            norm_subs.append({"titulo": stitle, "preguntas": qlist})
        new_caps.append({"titulo": ctitle, "subcapitulos": norm_subs})
    matrix["capitulos"] = new_caps
    return matrix


def to_dataframe(matrix: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for ci, cap in enumerate(matrix.get("capitulos", []), start=1):
        ctitle = (cap.get("titulo") or "").strip() or f"Capítulo {ci}"
        for si, sub in enumerate(cap.get("subcapitulos") or [], start=1):
            stitle = (sub.get("titulo") or "").strip() or f"Bloque {si}"
            for q in sub.get("preguntas") or []:
                rows.append({"Capitulo": ctitle, "Subcapitulo": stitle, "Preguntas": _question_to_str(q)})
    return pd.DataFrame(rows, columns=["Capitulo", "Subcapitulo", "Preguntas"])


# =====================================================
# ENDPOINT PRINCIPAL
# =====================================================
@app.post("/generate-matrix")
async def generate_matrix(
    contexto: UploadFile = File(...),
    objetivos: UploadFile = File(...),
    guia: Optional[UploadFile] = File(None),
    filename_base: Optional[str] = Form("matriz.xlsx"),
    openai_api_key: Optional[str] = Form(None),
):
    try:
        contexto_bytes = await contexto.read()
        objetivos_bytes = await objetivos.read()
        guia_bytes = await guia.read() if guia else None

        llm = LLMClient(api_key=openai_api_key)

        ctx_text = extract_text_from_pdf_bytes(contexto_bytes)
        obj_text = extract_text_from_pdf_bytes(objetivos_bytes)

        # Detectar preguntas
        preguntas = extract_questions_from_text_v3(extract_text_from_pdf_bytes(guia_bytes)) if guia_bytes else []
        print(f"Preguntas detectadas: {len(preguntas)}")

        # Construcción
        matrix = build_matrix_with_llm(llm, ctx_text, obj_text, preguntas)
        matrix = normalize_matrix_structure(matrix, preguntas)
        df = to_dataframe(matrix)

        # Exportar Excel
        buf = io.BytesIO()
        df.to_excel(buf, index=False)
        buf.seek(0)

        token = generate_token()
        exp = datetime.utcnow() + timedelta(minutes=TOKEN_EXPIRATION_MINUTES)
        DOWNLOADS[token] = (buf.getvalue(), filename_base or "matriz.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", exp)

        cleanup_downloads()
        download_url = f"/download/{token}"

        return JSONResponse({
            "status": "ok",
            "download_url": download_url,
            "filename": filename_base,
            "detected_questions": len(preguntas),
            "tokens": {
                "total": llm.total_tokens,
                "prompt": llm.prompt_tokens,
                "completion": llm.completion_tokens
            },
            "expires_in_minutes": TOKEN_EXPIRATION_MINUTES
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================
# ENDPOINT DE DESCARGA
# =====================================================
@app.get("/download/{token}")
def download_token(token: str):
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
