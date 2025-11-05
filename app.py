import os
import io
import re
import json
import math
import zipfile
import unicodedata
import pdfplumber
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from decimal import Decimal, ROUND_HALF_UP, getcontext

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from tenacity import retry, stop_after_attempt, wait_exponential_jitter
from dotenv import load_dotenv

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# =================== Config Decimal (coste exacto) ===================
getcontext().prec = 28  # alta precisión para evitar errores de coma flotante

# =====================================================
# CONFIG
# =====================================================
load_dotenv()
app = FastAPI(title="Generador de Matrices - Dipli (ZIP + Coste exacto)")

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
# CLIENTE OPENAI (cuenta tokens + desglose por llamada)
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
        self.calls = []  # desglose por llamada

    def _accumulate_usage(self, resp):
        try:
            u = getattr(resp, "usage", None)
            if not u:
                return
            model_name = getattr(resp, "model", self.model)
            pt = int(getattr(u, "prompt_tokens", 0) or 0)
            ct = int(getattr(u, "completion_tokens", 0) or 0)
            tt = int(getattr(u, "total_tokens", 0) or 0)
            self.total_tokens += tt
            self.prompt_tokens += pt
            self.completion_tokens += ct
            self.calls.append({
                "model": model_name,
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "total_tokens": tt,
            })
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
                flush(); preguntas.extend(segment_questions_from_line(ln)); i += 1; continue
        if re.search(r"¿[^?]*\?", ln):
            flush(); preguntas.extend(segment_questions_from_line(ln)); i += 1; continue
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
            seen.add(key); out.append(q)
    return out


# =====================================================
# UTIL/ESTRUCTURA
# =====================================================
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
    caps = struct.get("capitulos") or []
    if not caps:
        struct["capitulos"] = [{"titulo": "Capítulo 1", "subcapitulos": [{"titulo": "Bloque 1"}, {"titulo": "Bloque 2"}]}]
        return struct
    fixed = []
    for ci, cap in enumerate(caps, start=1):
        ct = (cap.get("titulo") or "").strip() or f"Capítulo {ci}"
        subs = cap.get("subcapitulos") or []
        if not subs:
            subs = [{"titulo": "Bloque 1"}, {"titulo": "Bloque 2"}]
        subs_fixed = []
        for si, sub in enumerate(subs, start=1):
            st = (sub.get("titulo") or "").strip() or f"Bloque {si}"
            d = {k: v for k, v in sub.items() if k != "titulo"}
            subs_fixed.append({"titulo": st, **d})
        fixed.append({"titulo": ct, "subcapitulos": subs_fixed})
    struct["capitulos"] = fixed
    return struct

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


# =====================================================
# ESTRUCTURA SEMÁNTICA (LLM)
# =====================================================
STRUCTURE_SEMANTIC_PROMPT = """
Devuelve SOLO títulos (sin preguntas) que reflejen la estructura temática de la GUÍA.
Formato JSON:
{
 "capitulos":[
   {"titulo":"string","subcapitulos":[{"titulo":"string"},{"titulo":"string"}]}
 ]
}
Reglas:
- Usa nombres cortos, claros y analíticos (máx. 60 caracteres).
- 2–3 subcapítulos por capítulo.
- Basado en los bloques/temas que se infieren de la GUÍA (encabezados, transiciones, etc.).
- NO incluyas "preguntas".
"""

def build_semantic_structure(llm: LLMClient, ctx: str, obj: str, guia_text: str) -> Dict[str, Any]:
    sys = "Eres Research Lead. Identificas bloques temáticos reales de una guía de entrevistas."
    user = f"""CONTEXTO:
{ctx}

OBJETIVOS:
{obj}

GUÍA (texto):
{guia_text}

{STRUCTURE_SEMANTIC_PROMPT}
"""
    raw = llm.chat_json(sys, user, temperature=0.2)
    return ensure_2_3_subchap_per_chapter(safe_json_loads(raw))

CLASSIFY_PROMPT = """
Te doy una estructura fija (capítulos/subcapítulos) y una lista de preguntas.
Devuelve asignaciones semánticas: cada pregunta debe ir a EXACTAMENTE 1 par (capitulo, subcapitulo).
No modifiques el texto de la pregunta.
Formato JSON:
{
 "assignments":[
   {"question":"...", "capitulo":"...", "subcapitulo":"..."}
 ]
}
Usa exclusivamente los títulos dados (exact match).
Mantén el orden de entrada en la salida.
"""

def classify_questions_to_structure(llm: LLMClient, struct: Dict[str, Any], preguntas: List[str]) -> List[Dict[str, str]]:
    snapshot = {
        "capitulos": [
            {
                "titulo": cap["titulo"],
                "subcapitulos": [ {"titulo": s["titulo"]} for s in cap.get("subcapitulos", []) ]
            }
            for cap in struct.get("capitulos", [])
        ]
    }
    sys = "Eres Research Lead. Clasificas preguntas en la estructura dada, por afinidad temática."
    q_blob = "\n".join(f"- {p}" for p in preguntas)
    user = f"""ESTRUCTURA:
{json.dumps(snapshot, ensure_ascii=False, indent=2)}

PREGUNTAS:
{q_blob}

{CLASSIFY_PROMPT}
"""
    raw = llm.chat_json(sys, user, temperature=0.1)
    data = safe_json_loads(raw)
    assigns = data.get("assignments") or []
    valid_caps = { cap["titulo"]: {s["titulo"] for s in cap.get("subcapitulos", [])} for cap in struct.get("capitulos", []) }
    cleaned = []
    for it in assigns:
        q = (it.get("question") or "").strip()
        c = (it.get("capitulo") or "").strip()
        s = (it.get("subcapitulo") or "").strip()
        if q and c in valid_caps and s in valid_caps[c]:
            cleaned.append({"question": q, "capitulo": c, "subcapitulo": s})
    return cleaned


# =====================================================
# REPARTO Y EXPORTACIÓN
# =====================================================
def apply_assignments(struct: Dict[str, Any], preguntas: List[str], assigns: List[Dict[str, str]]) -> Dict[str, Any]:
    struct = ensure_2_3_subchap_per_chapter(struct)
    cap_map = {}
    for cap in struct["capitulos"]:
        ctitle = cap["titulo"]
        cap_map[ctitle] = {}
        for sub in cap["subcapitulos"]:
            sub["preguntas"] = []
            cap_map[ctitle][sub["titulo"]] = sub["preguntas"]

    assigned_set = set()
    for it in assigns:
        q = it["question"]
        c = it["capitulo"]
        s = it["subcapitulo"]
        cap_map[c][s].append(q)
        assigned_set.add(q.strip())

    leftovers = [q for q in preguntas if q.strip() not in assigned_set]
    if leftovers:
        sub_refs = []
        for cap in struct["capitulos"]:
            for sub in cap.get("subcapitulos", []):
                sub_refs.append(sub["preguntas"])
        if not sub_refs:
            struct = build_default_structure(len(preguntas))
            sub_refs = []
            for cap in struct["capitulos"]:
                for sub in cap.get("subcapitulos"):
                    sub_refs.append(sub["preguntas"])
        for i, q in enumerate(leftovers):
            sub_refs[i % len(sub_refs)].append(q)

    return struct

def to_dataframe(matrix: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    caps = matrix.get("capitulos") or []
    for ci, cap in enumerate(caps, start=1):
        cap_title = (cap.get("titulo") or "").strip() or f"Capítulo {ci}"
        subs = cap.get("subcapitulos") or []
        for si, sub in enumerate(subs, start=1):
            sub_title = (sub.get("titulo") or "").strip() or f"Bloque {si}"
            for q in (sub.get("preguntas") or []):
                rows.append({"Capitulo": cap_title, "Subcapitulo": sub_title, "Preguntas": q})
    return pd.DataFrame(rows, columns=["Capitulo", "Subcapitulo", "Preguntas"])


# =====================================================
# COSTO EXACTO (Decimal)
# =====================================================
def estimate_cost_usd(prompt_tokens: int, completion_tokens: int,
                      in_per_1k: str = None, out_per_1k: str = None) -> Decimal:
    in_price = Decimal((in_per_1k if in_per_1k is not None else os.getenv("OPENAI_INPUT_COST_PER_1K", "0")) or "0")
    out_price = Decimal((out_per_1k if out_per_1k is not None else os.getenv("OPENAI_OUTPUT_COST_PER_1K", "0")) or "0")
    prompt_cost = (Decimal(prompt_tokens) / Decimal(1000)) * in_price
    completion_cost = (Decimal(completion_tokens) / Decimal(1000)) * out_price
    total = (prompt_cost + completion_cost).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)
    return total


# =====================================================
# ENDPOINT PRINCIPAL
# =====================================================
@app.post("/generate-matrix")
async def generate_matrix(
    contexto: UploadFile = File(...),
    objetivos: UploadFile = File(...),
    guia: Optional[UploadFile] = File(None),
    filename_base: Optional[str] = Form("matriz"),
    openai_api_key: Optional[str] = Form(None),
    num_questions: Optional[int] = Form(None),
):
    process_start = datetime.utcnow()

    # Leer archivos
    contexto_bytes = await contexto.read()
    objetivos_bytes = await objetivos.read()
    guia_bytes = await guia.read() if guia else None

    # Textos base
    ctx_text = extract_text_from_pdf_bytes(contexto_bytes)
    obj_text = extract_text_from_pdf_bytes(objetivos_bytes)

    # Cliente LLM
    llm = LLMClient(api_key=openai_api_key)

    # ===== CON GUÍA
    if guia_bytes:
        guia_text = extract_text_from_pdf_bytes(guia_bytes)
        preguntas = extract_questions_from_text_v5(guia_text)
        try:
            structure = build_semantic_structure(llm, ctx_text, obj_text, guia_text)
        except Exception:
            structure = build_default_structure(len(preguntas))
        try:
            assigns = classify_questions_to_structure(llm, structure, preguntas)
        except Exception:
            assigns = []
        matrix = apply_assignments(structure, preguntas, assigns)
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
    # ===== SIN GUÍA
    else:
        n = int(num_questions or 50)
        n = max(50, min(150, n))
        try:
            structure = build_semantic_structure(llm, ctx_text, obj_text, guia_text="")
        except Exception:
            structure = build_default_structure(n)
        # Generar preguntas (exactamente n)
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
        try:
            assigns = classify_questions_to_structure(llm, structure, preguntas)
        except Exception:
            assigns = []
        matrix = apply_assignments(structure, preguntas, assigns)
        df = to_dataframe(matrix)
        if len(df) != n:
            assigned = [row["Preguntas"] for _, row in df.iterrows()]
            assigned_norm = set(re.sub(r"\s+", " ", a).strip() for a in assigned)
            missing = [p for p in preguntas if re.sub(r"\s+", " ", p).strip() not in assigned_norm]
            if missing:
                last_sub = matrix["capitulos"][-1]["subcapitulos"][-1]
                last_sub.setdefault("preguntas", []).append(missing)
                df = to_dataframe(matrix)
                if len(df) != n:
                    raise RuntimeError("No se pudo igualar exactamente el total de preguntas generadas.")

    # ===== Crear Excel en memoria
    excel_buf = io.BytesIO()
    df.to_excel(excel_buf, index=False, engine="openpyxl")
    excel_buf.seek(0)

    # ===== Métricas y TXT (con Decimal)
    process_end = datetime.utcnow()
    duration_min = round((process_end - process_start).total_seconds() / 60.0, 2)
    total_q = len(df)
    total_tokens = llm.total_tokens
    prompt_tokens = llm.prompt_tokens
    completion_tokens = llm.completion_tokens
    est_cost_dec = estimate_cost_usd(prompt_tokens, completion_tokens)
    est_cost_str = f"{est_cost_dec.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)}"
    in_unit = os.getenv("OPENAI_INPUT_COST_PER_1K", "0")
    out_unit = os.getenv("OPENAI_OUTPUT_COST_PER_1K", "0")

    start_str = process_start.strftime("%Y-%m-%d %H:%M:%S UTC")
    end_str = process_end.strftime("%Y-%m-%d %H:%M:%S UTC")

    txt_report = []
    txt_report.append("=== DIPLI · MATRIZ - REPORTE ===")
    txt_report.append(f"Inicio:            {start_str}")
    txt_report.append(f"Fin:               {end_str}")
    txt_report.append(f"Duración (min):    {duration_min}")
    txt_report.append("")
    txt_report.append(f"Preguntas totales: {total_q}")
    txt_report.append("")
    txt_report.append("=== TOKENS (OpenAI) ===")
    txt_report.append(f"Prompt tokens:     {prompt_tokens}")
    txt_report.append(f"Completion tokens: {completion_tokens}")
    txt_report.append(f"Total tokens:      {total_tokens}")
    txt_report.append("")
    txt_report.append("=== PRECIOS (USD por 1K tokens) ===")
    txt_report.append(f"Entrada (prompt):  {in_unit}")
    txt_report.append(f"Salida (completion): {out_unit}")
    txt_report.append("")
    txt_report.append("=== COSTO ESTIMADO ===")
    txt_report.append(f"USD exacto (6 dec): {est_cost_dec}")
    txt_report.append(f"USD redondeado (¢):  {est_cost_str}")
    txt_report.append("Nota: cálculo determinístico con precios de entorno.")
    if llm.calls:
        txt_report.append("")
        txt_report.append("=== DESGLOSE POR LLAMADA ===")
        txt_report.append("Modelo;Prompt;Completion;Total;USD(aprox)")
        for c in llm.calls:
            c_cost = estimate_cost_usd(c["prompt_tokens"], c["completion_tokens"])
            c_cost_str = f"{c_cost.quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP)}"
            line = f'{c["model"]};{c["prompt_tokens"]};{c["completion_tokens"]};{c["total_tokens"]};{c_cost_str}'
            txt_report.append(line)
    txt_bytes = ("\n".join(txt_report)).encode("utf-8")

    # ===== Preparar ZIP (Excel + TXT)
    base_name = (filename_base or "matriz").strip()
    base_name = re.sub(r'[\\/:*?"<>|]+', "_", base_name)
    if base_name.lower().endswith(".xlsx"):
        base_name = base_name[:-5]
    if base_name.lower().endswith(".zip"):
        base_name = base_name[:-4]

    excel_filename = f"{base_name}.xlsx"
    txt_filename = f"{base_name}_reporte.txt"
    zip_filename = f"{base_name}.zip"

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(excel_filename, excel_buf.getvalue())
        zf.writestr(txt_filename, txt_bytes)
    zip_buf.seek(0)

    # ===== Registrar descarga temporal (.zip)
    token = generate_token()
    exp = datetime.utcnow() + timedelta(minutes=TOKEN_EXPIRATION_MINUTES)
    DOWNLOADS[token] = (zip_buf.getvalue(), zip_filename, "application/zip", exp)
    cleanup_downloads()

    return JSONResponse({
        "status": "ok",
        "download_url": f"/download/{token}",
        "filename": zip_filename,
        "detected_questions": total_q,
        "semantic_order": True,
        "metrics": {
            "started_at_utc": start_str,
            "ended_at_utc": end_str,
            "duration_minutes": duration_min,
            "tokens": {
                "total": total_tokens,
                "prompt": prompt_tokens,
                "completion": completion_tokens
            },
            "estimated_cost_usd": float(est_cost_dec)  # si prefieres exactitud como string: str(est_cost_dec)
        }
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
