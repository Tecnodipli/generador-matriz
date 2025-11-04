import os
import io
import re
import json
import unicodedata
import math
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
# CONFIGURACIÓN BASE
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
# UTILIDADES
# =====================================================
def cleanup_downloads():
    now = datetime.utcnow()
    for k, v in list(DOWNLOADS.items()):
        if v[3] <= now:
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
# CLIENTE OPENAI (cuenta tokens)
# =====================================================
class LLMClient:
    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        if OpenAI is None:
            raise RuntimeError("Falta la librería openai.")
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("No hay OPENAI_API_KEY configurada.")
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
# EXTRACTOR DE PREGUNTAS (v5 CONFIGURABLE)
# =====================================================
def _load_list_env(key: str, base: list[str]) -> list[str]:
    raw = os.getenv(key, "")
    if not raw.strip():
        return base
    extra = [t.strip() for t in raw.split(",") if t.strip()]
    return list(dict.fromkeys(base + extra))

IMPERATIVOS_BASE = [
    "cuéntame","cuentame","háblame","hablame","dime","menciona","explícame","explicame",
    "relata","descríbeme","describeme","indique","indica","mencione","narre",
    "explique","detalla","detalle","describe","comparte","piensa","recuerda","cuenta",
    "comente","profundice","aclare","justifique","argumente","enumere","motive",
]
INSTRUCCIONES_BASE = [
    "DILIGENCIAR","LEA","ACLAR","MOSTRAR","ROTACIÓN","APLICAR","GRABACIÓN","GRABACION",
    "CUESTIONARIO","MANTENGA VISIBLE","NO LEER","LEER","MUESTRE","ENTREGUE","PROYECTE",
    "PASE A","CAMBIE A","LEA EN VOZ ALTA",
]
MOD_TAGS_BASE = ["ENT.","MOD.","ENTREVISTADOR","FACILITADOR","GUIA"]

IMPERATIVOS = _load_list_env("EXTRACTOR_IMPERATIVOS", IMPERATIVOS_BASE)
INSTRUCCIONES = _load_list_env("EXTRACTOR_INSTRUCCIONES", INSTRUCCIONES_BASE)
MOD_TAGS = _load_list_env("EXTRACTOR_MOD_TAGS", MOD_TAGS_BASE)

_INTERROGATIVOS_RE = r"(qué|como|cómo|cual|cuál|cuando|cuándo|donde|dónde|quien|quién|por qué|para qué|cuanto|cuánto|cuales|cuáles)"
_IMPERATIVOS_RE = r"(" + "|".join(re.escape(v) for v in IMPERATIVOS) + r")"
_HEADER_LINE = re.compile(r"^[A-ZÁÉÍÓÚÑ0-9\s\-\(\)\/\#\.\:]+$")
_MODERATOR_TAG = re.compile(r"^\s*\(?\s*(?:" + "|".join(re.escape(t) for t in MOD_TAGS) + r")", re.IGNORECASE)
_ONLY_INSTRUCTION = re.compile(r"(" + "|".join(re.escape(v) for v in INSTRUCCIONES) + r")", re.IGNORECASE)

def _strip_bullets(s: str) -> str:
    s = re.sub(r"^\s*([•●○\-–—]|(\d+[\.\)]|[a-zA-Z][\.\)]))\s*", "", s or "")
    return s.strip()

def _normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def _casefold(s: str) -> str:
    return unicodedata.normalize("NFKC", (s or "")).casefold().strip()

def _looks_like_header(line: str) -> bool:
    txt = line.strip()
    if len(txt) <= 2:
        return False
    return bool(_HEADER_LINE.match(txt)) and "?" not in txt and "¿" not in txt

def _is_questionish_start(line: str) -> bool:
    cf = _casefold(line)
    return (
        re.match(rf"^\s*{_INTERROGATIVOS_RE}\b", cf)
        or re.match(rf"^\s*{_IMPERATIVOS_RE}\b", cf)
    )

def _has_any_question(line: str) -> bool:
    cf = _casefold(line)
    return ("?" in line or "¿" in line
            or re.search(rf"\b{_INTERROGATIVOS_RE}\b", cf)
            or re.match(rf"^\s*{_IMPERATIVOS_RE}\b", cf))

def segment_questions_from_line(line: str) -> List[str]:
    parts = []
    segments = re.split(r"(¿[^?]*\?)", line)
    for seg in segments:
        seg = _normalize_spaces(seg)
        if not seg:
            continue
        if seg.startswith("¿") and seg.endswith("?"):
            parts.append(seg)
        else:
            if _is_questionish_start(seg):
                if not seg.endswith("?"):
                    seg += "?"
                parts.append(seg)
    return [_strip_bullets(p) for p in parts if len(p) > 3]

def extract_questions_from_text_v5(text: str) -> List[str]:
    if not text:
        return []
    lines = [_strip_bullets(l) for l in text.splitlines()]
    lines = [l for l in lines if _normalize_spaces(l)]
    preguntas, buffer = [], []

    def flush():
        if not buffer:
            return
        merged = _normalize_spaces(" ".join(buffer))
        segs = segment_questions_from_line(merged)
        if not segs and _has_any_question(merged):
            if not merged.endswith("?"):
                merged += "?"
            segs = [merged]
        preguntas.extend(segs)
        buffer.clear()

    i = 0
    while i < len(lines):
        ln = _normalize_spaces(lines[i])
        if _looks_like_header(ln):
            flush(); i += 1; continue
        if _MODERATOR_TAG.match(ln):
            if _ONLY_INSTRUCTION.search(ln) and not _has_any_question(ln):
                flush(); i += 1; continue
            if _has_any_question(ln):
                flush()
                preguntas.extend(segment_questions_from_line(ln))
                i += 1; continue
        if re.search(r"¿[^?]*\?", ln):
            flush()
            preguntas.extend(segment_questions_from_line(ln))
            i += 1; continue
        if _is_questionish_start(ln):
            flush(); buffer.append(ln)
            j = i + 1
            while j < len(lines):
                nxt = _normalize_spaces(lines[j])
                if not nxt or _looks_like_header(nxt) or re.search(r"¿[^?]*\?", nxt) or _is_questionish_start(nxt):
                    break
                if _ONLY_INSTRUCTION.search(nxt) and not _has_any_question(nxt):
                    break
                buffer.append(nxt); j += 1
            flush(); i = j; continue
        if buffer:
            nxt = ln
            if not (_looks_like_header(nxt) or _MODERATOR_TAG.match(nxt)):
                if not _ONLY_INSTRUCTION.search(nxt) or _has_any_question(nxt):
                    buffer.append(nxt); i += 1; continue
        i += 1
    flush()

    seen, out = set(), []
    for q in preguntas:
        key = re.sub(r"[\W_]+", "", _casefold(q))
        if key not in seen:
            seen.add(key)
            out.append(q)
    return out


# =====================================================
# GENERACIÓN DE MATRIZ (estructura y reparto exacto)
# =====================================================
STRUCTURE_ONLY_INSTRUCTIONS = """
Devuelve SOLO títulos, sin preguntas. Estructura:
{
 "capitulos":[
   {"titulo":"string","subcapitulos":[{"titulo":"string"}, {"titulo":"string"}]}
 ]
}
Reglas:
- 2–3 subcapítulos por capítulo.
- Títulos claros y breves.
- No incluyas campo "preguntas".
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

def ensure_titles(struct: Dict[str, Any]) -> Dict[str, Any]:
    """Asegura títulos no vacíos en capítulos y subcapítulos."""
    caps = struct.get("capitulos") or []
    if not caps:
        struct["capitulos"] = [{"titulo": "Capítulo 1", "subcapitulos": [{"titulo": "Bloque 1"}, {"titulo": "Bloque 2"}]}]
        return struct
    fixed = []
    for ci, cap in enumerate(caps, start=1):
        ct = (cap.get("titulo") or "").strip() or f"Capítulo {ci}"
        subs = cap.get("subcapitulos") or []
        subs_fixed = []
        if not subs:
            subs = [{"titulo": "Bloque 1"}, {"titulo": "Bloque 2"}]
        for si, sub in enumerate(subs, start=1):
            st = (sub.get("titulo") or "").strip() or f"Bloque {si}"
            subs_fixed.append({"titulo": st, **({k:v for k,v in sub.items() if k!='titulo'})})
        fixed.append({"titulo": ct, "subcapitulos": subs_fixed})
    struct["capitulos"] = fixed
    return struct

def build_structure_with_llm(llm: LLMClient, contexto_sum: str, objetivos_sum: str) -> Dict[str, Any]:
    sys = "Eres Research Lead experto en investigación cualitativa."
    user = f"""CONTEXTO:
{contexto_sum}

OBJETIVOS:
{objetivos_sum}

{STRUCTURE_ONLY_INSTRUCTIONS}
"""
    raw = llm.chat_json(sys, user, temperature=0.1)
    return ensure_titles(safe_json_loads(raw))

def ensure_2_3_subchap_per_chapter(struct: Dict[str, Any]) -> Dict[str, Any]:
    struct = ensure_titles(struct)
    caps = struct.get("capitulos") or []
    fixed = []
    for idx, cap in enumerate(caps, start=1):
        ctitle = (cap.get("titulo") or "").strip() or f"Capítulo {idx}"
        subs = cap.get("subcapitulos") or []
        subs = [{"titulo": (s.get("titulo") or "").strip() or f"Bloque {i+1}"} for i, s in enumerate(subs)]
        if len(subs) < 2:
            subs = subs + [{"titulo": f"Bloque {len(subs)+1}"}]
        if len(subs) > 3:
            subs = subs[:3]
        fixed.append({"titulo": ctitle, "subcapitulos": subs})
    struct["capitulos"] = fixed
    return ensure_titles(struct)

def build_default_structure(preguntas_count: int) -> Dict[str, Any]:
    if preguntas_count <= 12:
        num_chapters = 3; subs_per_ch = 2
    elif preguntas_count <= 30:
        num_chapters = 3; subs_per_ch = 3
    else:
        num_chapters = min(5, math.ceil(preguntas_count / 15)); subs_per_ch = 3
    caps = []
    for i in range(num_chapters):
        subs = [{"titulo": f"Bloque {j+1}"} for j in range(subs_per_ch)]
        caps.append({"titulo": f"Capítulo {i+1}", "subcapitulos": subs})
    return ensure_titles({"capitulos": caps})

def distribute_questions_exact(preguntas: List[str], struct: Dict[str, Any]) -> Dict[str, Any]:
    struct = ensure_2_3_subchap_per_chapter(struct)
    caps = struct.get("capitulos") or []
    sub_refs = []
    for cap in caps:
        cap_title = (cap.get("titulo") or "").strip() or "Capítulo"
        subs = cap.get("subcapitulos") or []
        fixed_subs = []
        for s in subs:
            st = (s.get("titulo") or "").strip() or "Bloque"
            fixed_subs.append({"titulo": st, "preguntas": []})
        cap["titulo"] = cap_title
        cap["subcapitulos"] = fixed_subs
        for s in cap["subcapitulos"]:
            sub_refs.append(s)

    if not sub_refs:
        struct = build_default_structure(len(preguntas))
        sub_refs = []
        for cap in struct["capitulos"]:
            for s in cap["subcapitulos"]:
                s["preguntas"] = []
                sub_refs.append(s)

    total_subs = len(sub_refs) or 1
    for idx, q in enumerate(preguntas):
        sub_refs[idx % total_subs]["preguntas"].append(q)

    return ensure_titles(struct)

def to_dataframe(matrix: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    caps = matrix.get("capitulos") or []
    for ci, cap in enumerate(caps, start=1):
        cap_title = (cap.get("titulo") or "").strip() or f"Capítulo {ci}"
        subs = cap.get("subcapitulos") or []
        for si, sub in enumerate(subs, start=1):
            sub_title = (sub.get("titulo") or "").strip() or f"Bloque {si}"
            for q in sub.get("preguntas") or []:
                rows.append({"Capitulo": cap_title, "Subcapitulo": sub_title, "Preguntas": q})
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
    num_questions: Optional[int] = Form(None),
):
    contexto_bytes = await contexto.read()
    objetivos_bytes = await objetivos.read()
    guia_bytes = await guia.read() if guia else None

    llm = LLMClient(api_key=openai_api_key)
    ctx_text = extract_text_from_pdf_bytes(contexto_bytes)
    obj_text = extract_text_from_pdf_bytes(objetivos_bytes)

    if guia_bytes:
        preguntas = extract_questions_from_text_v5(extract_text_from_pdf_bytes(guia_bytes))
        # Estructura de títulos (sin preguntas)
        try:
            structure = build_structure_with_llm(llm, ctx_text, obj_text)
            structure = ensure_2_3_subchap_per_chapter(structure)
        except Exception:
            structure = build_default_structure(len(preguntas))
        matrix = distribute_questions_exact(preguntas, structure)

        df = to_dataframe(matrix)
        if len(df) != len(preguntas):
            assigned = [row["Preguntas"] for _, row in df.iterrows()]
            assigned_norm = set(re.sub(r"\s+", " ", a).strip() for a in assigned)
            missing = [p for p in preguntas if re.sub(r"\s+", " ", p).strip() not in assigned_norm]
            if missing:
                last_sub = None
                if matrix.get("capitulos") and matrix["capitulos"][-1].get("subcapitulos"):
                    last_sub = matrix["capitulos"][-1]["subcapitulos"][-1]
                if last_sub is None:
                    matrix = build_default_structure(len(preguntas))
                    last_sub = matrix["capitulos"][-1]["subcapitulos"][-1]
                last_sub.setdefault("preguntas", []).extend(missing)
                df = to_dataframe(matrix)
                if len(df) != len(preguntas):
                    raise RuntimeError("No se pudo igualar exactamente el total de preguntas.")
    else:
        # Sin guía: generar exactamente n preguntas + estructura
        n = int(num_questions or 16)
        n = max(8, min(60, n))
        try:
            structure = build_structure_with_llm(llm, ctx_text, obj_text)
            structure = ensure_2_3_subchap_per_chapter(structure)
        except Exception:
            structure = build_default_structure(n)

        sys = "Eres Research Lead. Generas preguntas claras y neutrales para una guía de entrevista."
        user = f"""CONTEXTO:
{ctx_text}

OBJETIVOS:
{obj_text}

Genera exactamente {n} preguntas (ni más ni menos), en estilo cualitativo.
Devuelve JSON: {{"preguntas": ["..."]}} sin texto adicional.
"""
        try:
            raw = llm.chat_json(sys, user, temperature=0.2)
            data = safe_json_loads(raw)
            gen_q = data.get("preguntas") or []
            preguntas = [str(x).strip() if isinstance(x, str) else str(x) for x in gen_q]
            preguntas = [q if q.endswith("?") else q + "?" for q in preguntas]
        except Exception:
            preguntas = [f"Pregunta {i+1}?" for i in range(n)]

        preguntas = preguntas[:n]
        matrix = distribute_questions_exact(preguntas, structure)
        df = to_dataframe(matrix)
        if len(df) != n:
            assigned = [row["Preguntas"] for _, row in df.iterrows()]
            assigned_norm = set(re.sub(r"\s+", " ", a).strip() for a in assigned)
            missing = [p for p in preguntas if re.sub(r"\s+", " ", p).strip() not in assigned_norm]
            if missing:
                last_sub = matrix["capitulos"][-1]["subcapitulos"][-1]
                last_sub.setdefault("preguntas", []).extend(missing)
                df = to_dataframe(matrix)
                if len(df) != n:
                    raise RuntimeError("No se pudo igualar exactamente el total de preguntas generadas.")

    # Exportar Excel
    buf = io.BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")
    buf.seek(0)

    # Nombre forzado .xlsx
    safe_name = (filename_base or "matriz.xlsx").strip()
    safe_name = re.sub(r'[\\/:*?"<>|]+', "_", safe_name)
    if not safe_name.lower().endswith(".xlsx"):
        safe_name += ".xlsx"

    token = generate_token()
    exp = datetime.utcnow() + timedelta(minutes=TOKEN_EXPIRATION_MINUTES)
    DOWNLOADS[token] = (buf.getvalue(), safe_name,
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", exp)
    cleanup_downloads()

    return JSONResponse({
        "status": "ok",
        "download_url": f"/download/{token}",
        "filename": safe_name,
        "detected_questions": len(df),  # filas en Excel
        "tokens": {
            "total": llm.total_tokens,
            "prompt": llm.prompt_tokens,
            "completion": llm.completion_tokens
        },
    })


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
