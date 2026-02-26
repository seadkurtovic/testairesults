import base64
import io
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer as st_pdf_viewer

# ----------------------------
# Helpers: parsing + schema
# ----------------------------

@dataclass
class FieldPred:
    value: Optional[str]
    confidence: Optional[float] = None

@dataclass
class DocPred:
    doc_id: str
    fields: Dict[str, FieldPred]

def normalize_doc_id(name: str) -> str:
    # Normalize to make matching more robust (case, whitespace)
    return re.sub(r"\s+", " ", name.strip().lower())

def safe_str(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    return s if s != "" else None

def parse_results_json(raw: str) -> List[DocPred]:
    """
    Accepts several common shapes:
    1) {"documents":[{"doc_id":"x.pdf","fields":{"A":{"value":"1","confidence":0.9},"B":"2"}}]}
    2) [{"doc_id":"x.pdf","fields":{...}}, ...]
    3) {"x.pdf":{"A":"1","B":"2"}, "y.pdf":{...}}  (doc_id as keys)
    """
    data = json.loads(raw)

    def parse_fields(fields_obj: Any) -> Dict[str, FieldPred]:
        out: Dict[str, FieldPred] = {}
        if not isinstance(fields_obj, dict):
            return out
        for k, v in fields_obj.items():
            if isinstance(v, dict):
                out[str(k)] = FieldPred(
                    value=safe_str(v.get("value")),
                    confidence=(float(v["confidence"]) if "confidence" in v and v["confidence"] is not None else None)
                )
            else:
                out[str(k)] = FieldPred(value=safe_str(v), confidence=None)
        return out

    docs: List[DocPred] = []

    if isinstance(data, dict) and "documents" in data and isinstance(data["documents"], list):
        for d in data["documents"]:
            if not isinstance(d, dict):
                continue
            doc_id = safe_str(d.get("doc_id") or d.get("document_id") or d.get("filename"))
            if not doc_id:
                continue
            fields = parse_fields(d.get("fields") or d.get("predictions") or {})
            docs.append(DocPred(doc_id=doc_id, fields=fields))
        return docs

    if isinstance(data, list):
        for d in data:
            if not isinstance(d, dict):
                continue
            doc_id = safe_str(d.get("doc_id") or d.get("document_id") or d.get("filename"))
            if not doc_id:
                continue
            fields = parse_fields(d.get("fields") or d.get("predictions") or {})
            docs.append(DocPred(doc_id=doc_id, fields=fields))
        return docs

    if isinstance(data, dict):
        # doc_id as keys
        for doc_id, fields_obj in data.items():
            if isinstance(fields_obj, dict):
                docs.append(DocPred(doc_id=str(doc_id), fields=parse_fields(fields_obj)))
        return docs

    return docs

def parse_results_csv(file_bytes: bytes, id_column: str | None = None) -> list[DocPred]:
    df = pd.read_csv(io.BytesIO(file_bytes))

    # Normalize column lookup
    cols_norm = {c.lower().strip(): c for c in df.columns}

    def has_col(*names: str) -> bool:
        return any(n in cols_norm for n in names)

    # ---- Case A: "long format" (one row per field)
    # Required: doc_id + field + value (names flexible)
    if has_col("field", "key", "name") and has_col("value", "pred_value", "prediction"):
        # choose doc column
        doc_candidates = ["doc_id", "document_id", "filename", "file", "id", "sp_num", "spnummer", "sp", "itemid"]
        c_doc = None
        for cand in doc_candidates:
            if cand in cols_norm:
                c_doc = cols_norm[cand]
                break
        if c_doc is None and id_column is not None and id_column in df.columns:
            c_doc = id_column
        if c_doc is None:
            raise ValueError("Long-CSV erkannt, aber keine Dokument-ID Spalte gefunden. Bitte id_column angeben.")

        # choose field/value/conf columns
        c_field = cols_norm.get("field") or cols_norm.get("key") or cols_norm.get("name")
        c_value = cols_norm.get("value") or cols_norm.get("pred_value") or cols_norm.get("prediction")
        c_conf = cols_norm.get("confidence") or cols_norm.get("score") or cols_norm.get("prob")

        docs_map: dict[str, dict[str, FieldPred]] = {}
        for _, row in df.iterrows():
            doc_id = safe_str(row[c_doc])
            field = safe_str(row[c_field])
            value = safe_str(row[c_value])
            if not doc_id or not field:
                continue
            conf = None
            if c_conf and pd.notna(row[c_conf]):
                try:
                    conf = float(row[c_conf])
                except Exception:
                    conf = None
            docs_map.setdefault(doc_id, {})[field] = FieldPred(value=value, confidence=conf)

        return [DocPred(doc_id=k, fields=v) for k, v in docs_map.items()]

    # ---- Case B: "wide format" (one row per document, columns are fields)
    # Determine id column
    if id_column is None:
        # Heuristik: wenn es eine typische ID-Spalte gibt, nimm die; sonst nimm die erste Spalte
        typical = ["sp_num", "spnummer", "itemid", "id", "doc_id", "document_id", "filename", "file"]
        for t in typical:
            if t in cols_norm:
                id_column = cols_norm[t]
                break
        if id_column is None:
            id_column = df.columns[0]  # fallback

    if id_column not in df.columns:
        raise ValueError(f"id_column '{id_column}' existiert nicht in der CSV.")

    field_cols = [c for c in df.columns if c != id_column]

    preds: list[DocPred] = []
    for _, row in df.iterrows():
        doc_id = safe_str(row[id_column])
        if not doc_id:
            continue
        fields: dict[str, FieldPred] = {}
        for c in field_cols:
            fields[str(c)] = FieldPred(value=safe_str(row[c]), confidence=None)
        preds.append(DocPred(doc_id=str(doc_id), fields=fields))

    return preds

def build_match_index(pdf_files: List[Tuple[str, bytes]], preds: List[DocPred]) -> Tuple[List[str], Dict[str, bytes], Dict[str, DocPred], List[str]]:
    """
    Returns:
      ordered_doc_ids (based on PDFs), pdf_map (doc_id -> bytes), pred_map (doc_id -> DocPred), unmatched_pdf_ids
    Matching strategy:
      - exact match on normalized doc_id (including .pdf)
      - if not found: try match by basename without extension
    """
    pred_by_norm: Dict[str, DocPred] = {normalize_doc_id(p.doc_id): p for p in preds}

    # also index by basename
    pred_by_base: Dict[str, DocPred] = {}
    for p in preds:
        base = normalize_doc_id(re.sub(r"\.pdf$", "", p.doc_id.strip(), flags=re.I))
        pred_by_base[base] = p

    ordered_ids: List[str] = []
    pdf_map: Dict[str, bytes] = {}
    pred_map: Dict[str, DocPred] = {}
    unmatched: List[str] = []

    for fname, b in pdf_files:
        doc_id = fname
        pdf_map[doc_id] = b
        ordered_ids.append(doc_id)

        norm = normalize_doc_id(doc_id)
        base = normalize_doc_id(re.sub(r"\.pdf$", "", doc_id, flags=re.I))

        p = pred_by_norm.get(norm) or pred_by_base.get(base)
        if p:
            pred_map[doc_id] = p  # key by PDF filename, so UI stays consistent
        else:
            unmatched.append(doc_id)

    return ordered_ids, pdf_map, pred_map, unmatched

def pdf_viewer(pdf_bytes: bytes):
    st.download_button(
        label="📄 Open PDF in new tab",
        data=pdf_bytes,
        file_name="document.pdf",
        mime="application/pdf",
        use_container_width=True
    )

    st.write("Preview:")
    st.markdown(
        f'<embed src="data:application/pdf;base64,{base64.b64encode(pdf_bytes).decode()}" '
        f'width="100%" height="700" type="application/pdf">',
        unsafe_allow_html=True
    )

def now_iso_berlin() -> str:
    # simple ISO timestamp (local tz not essential for MVP)
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


# ----------------------------
# Metrics
# ----------------------------

def compute_metrics(validations: List[dict]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """
    validations rows:
      {doc_id, field, pred_value, confidence, verdict, true_value}
    verdict in {"correct","incorrect","missing","skip"}
    """
    df = pd.DataFrame(validations)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), {"overall_accuracy_on_predicted": 0.0}

    # Field-level counts
    def is_tp(v): return v == "correct"
    def is_fp(v): return v == "incorrect"
    def is_fn(v): return v == "missing"

    df["TP"] = df["verdict"].apply(lambda v: 1 if is_tp(v) else 0)
    df["FP"] = df["verdict"].apply(lambda v: 1 if is_fp(v) else 0)
    df["FN"] = df["verdict"].apply(lambda v: 1 if is_fn(v) else 0)

    field_grp = df.groupby("field", dropna=False)[["TP", "FP", "FN"]].sum().reset_index()
    field_grp["precision"] = field_grp.apply(lambda r: r["TP"] / (r["TP"] + r["FP"]) if (r["TP"] + r["FP"]) > 0 else None, axis=1)
    field_grp["recall"] = field_grp.apply(lambda r: r["TP"] / (r["TP"] + r["FN"]) if (r["TP"] + r["FN"]) > 0 else None, axis=1)
    field_grp["f1"] = field_grp.apply(
        lambda r: (2 * r["precision"] * r["recall"] / (r["precision"] + r["recall"]))
        if (r["precision"] is not None and r["recall"] is not None and (r["precision"] + r["recall"]) > 0)
        else None,
        axis=1
    )

    # Document-level: how many correct/incorrect/missing per doc
    doc_grp = df.groupby("doc_id")[["TP", "FP", "FN"]].sum().reset_index()
    doc_grp["total_judged"] = doc_grp["TP"] + doc_grp["FP"] + doc_grp["FN"]
    doc_grp["doc_accuracy"] = doc_grp.apply(lambda r: r["TP"] / r["total_judged"] if r["total_judged"] > 0 else None, axis=1)

    # Overall accuracy (only on predicted fields i.e. correct vs incorrect)
    judged_pred = df[df["verdict"].isin(["correct", "incorrect"])]
    overall_acc = (judged_pred["verdict"] == "correct").mean() if len(judged_pred) else 0.0

    overall = {"overall_accuracy_on_predicted": float(overall_acc)}
    return field_grp, doc_grp, overall


# ----------------------------
# Streamlit App
# ----------------------------

st.set_page_config(page_title="Test AI Results", layout="wide")

st.title("Test AI Results")

if "stage" not in st.session_state:
    st.session_state.stage = "upload"

if "pdf_files" not in st.session_state:
    st.session_state.pdf_files = []  # List[(filename, bytes)]
if "preds" not in st.session_state:
    st.session_state.preds = []       # List[DocPred]
if "ordered_ids" not in st.session_state:
    st.session_state.ordered_ids = []
if "pdf_map" not in st.session_state:
    st.session_state.pdf_map = {}
if "pred_map" not in st.session_state:
    st.session_state.pred_map = {}
if "idx" not in st.session_state:
    st.session_state.idx = 0
if "validations" not in st.session_state:
    st.session_state.validations = []  # list of dict rows
if "run_meta" not in st.session_state:
    st.session_state.run_meta = {
        "run_name": f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "model_version": "",
        "prompt_version": "",
        "created_at": now_iso_berlin(),
    }

with st.sidebar:
    st.header("Run Meta")
    st.session_state.run_meta["run_name"] = st.text_input("Run Name", st.session_state.run_meta["run_name"])
    st.session_state.run_meta["model_version"] = st.text_input("Model Version (optional)", st.session_state.run_meta["model_version"])
    st.session_state.run_meta["prompt_version"] = st.text_input("Prompt Version (optional)", st.session_state.run_meta["prompt_version"])
    st.caption("Tipp: Diese Metadaten landen im Export, damit dein Test reproduzierbar ist.")

# ----------------------------
# Stage 1: Upload
# ----------------------------
if st.session_state.stage == "upload":
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.subheader("Upload PDFs")
        pdfs = st.file_uploader("PDFs auswählen", type=["pdf"], accept_multiple_files=True)
        if pdfs:
            st.session_state.pdf_files = [(f.name, f.read()) for f in pdfs]
            st.success(f"{len(st.session_state.pdf_files)} PDF(s) geladen.")

    with col2:
        st.subheader("Upload AI Results")
        res_file = st.file_uploader("JSON oder CSV auswählen", type=["json", "csv"])

        if res_file:
            try:
                if res_file.name.lower().endswith(".json"):
                    raw = res_file.getvalue().decode("utf-8")
                    st.session_state.preds = parse_results_json(raw)

                else:
                    csv_bytes = res_file.getvalue()
                    df_preview = pd.read_csv(io.BytesIO(csv_bytes))

                    st.write("CSV Columns:", list(df_preview.columns))

                    id_col_choice = st.selectbox(
                        "Welche Spalte ist die Dokument-ID? (Dateiname-Spalte auswählen)",
                        df_preview.columns
                    )

                    st.session_state.preds = parse_results_csv(csv_bytes, id_column=id_col_choice)

                st.success(f"{len(st.session_state.preds)} Dokument-Result(s) geladen.")

            except Exception as e:
                st.error(f"Konnte Results nicht lesen: {e}")

    st.divider()

    if st.session_state.pdf_files and st.session_state.preds:
        ordered_ids, pdf_map, pred_map, unmatched = build_match_index(st.session_state.pdf_files, st.session_state.preds)
        st.session_state.ordered_ids = ordered_ids
        st.session_state.pdf_map = pdf_map
        st.session_state.pred_map = pred_map

        if unmatched:
            st.warning(f"Für {len(unmatched)} PDF(s) wurde kein passendes Result gefunden. (Du kannst trotzdem starten.)")
            with st.expander("Unmatched PDFs anzeigen"):
                st.write(unmatched)

        if st.button("Start Validation ▶️", type="primary", use_container_width=True):
            st.session_state.idx = 0
            st.session_state.validations = []
            st.session_state.stage = "quiz"
            st.rerun()
    else:
        st.info("Bitte zuerst PDFs und AI Results hochladen.")

# ----------------------------
# Stage 2: Quiz / Validation
# ----------------------------
elif st.session_state.stage == "quiz":
    ordered = st.session_state.ordered_ids
    i = st.session_state.idx

    if not ordered:
        st.error("Keine PDFs gefunden. Geh zurück zum Upload.")
        if st.button("⬅️ Zurück"):
            st.session_state.stage = "upload"
            st.rerun()
    else:
        doc_id = ordered[i]
        pdf_bytes = st.session_state.pdf_map[doc_id]
        pred = st.session_state.pred_map.get(doc_id)

        top = st.columns([2, 1, 1, 1])
        top[0].subheader(f"Document {i+1}/{len(ordered)} — {doc_id}")

        if top[1].button("⬅️ Prev", disabled=(i == 0), use_container_width=True):
            st.session_state.idx = max(0, i - 1)
            st.rerun()

        if top[2].button("Next ➡️", disabled=(i >= len(ordered) - 1), use_container_width=True):
            st.session_state.idx = min(len(ordered) - 1, i + 1)
            st.rerun()

        if top[3].button("Finish ✅", use_container_width=True):
            st.session_state.stage = "results"
            st.rerun()

        left, right = st.columns([1.1, 0.9], gap="large")

        with left:
            st.caption("PDF")
            st_pdf_viewer(pdf_bytes, height=720)

        with right:
            st.caption("Felder (✅ korrekt / ❌ falsch / ✏️ korrigieren / ∅ missing / ⏭ skip)")
            if pred is None:
                st.warning("Kein Result für dieses PDF gematched. Du kannst skippen oder zurück zum Upload.")
                if st.button("⏭ Skip this doc"):
                    st.session_state.idx = min(len(ordered) - 1, i + 1)
                    st.rerun()
            else:
                # Key to keep widget state per doc
                st.session_state.setdefault("doc_notes", {})
                st.session_state.doc_notes.setdefault(doc_id, "")

                # Optional per-doc note
                st.session_state.doc_notes[doc_id] = st.text_area(
                    "Notiz (optional)",
                    value=st.session_state.doc_notes[doc_id],
                    height=80,
                    key=f"note_{doc_id}"
                )

                fields = list(pred.fields.keys())
                if not fields:
                    st.info("Keine Felder im Result gefunden.")
                else:
                    # Show fields sorted
                    fields_sorted = sorted(fields, key=lambda x: x.lower())

                    # Quick actions
                    qa = st.columns(3)
                    if qa[0].button("✅ All Correct", use_container_width=True):
                        for f in fields_sorted:
                            st.session_state[f"verdict_{doc_id}_{f}"] = "correct"
                            st.session_state[f"true_{doc_id}_{f}"] = ""
                        st.rerun()
                    if qa[1].button("⏭ All Skip", use_container_width=True):
                        for f in fields_sorted:
                            st.session_state[f"verdict_{doc_id}_{f}"] = "skip"
                            st.session_state[f"true_{doc_id}_{f}"] = ""
                        st.rerun()
                    if qa[2].button("❌ All Incorrect", use_container_width=True):
                        for f in fields_sorted:
                            st.session_state[f"verdict_{doc_id}_{f}"] = "incorrect"
                            st.rerun()

                    st.divider()

                    # Per field UI
                    for f in fields_sorted:
                        pred_val = pred.fields[f].value
                        conf = pred.fields[f].confidence

                        st.markdown(f"**{f}**")
                        st.write(f"Predicted: `{pred_val}`" if pred_val is not None else "Predicted: _(empty)_")
                        if conf is not None:
                            st.caption(f"confidence: {conf:.3f}")

                        verdict_key = f"verdict_{doc_id}_{f}"
                        true_key = f"true_{doc_id}_{f}"

                        if verdict_key not in st.session_state:
                            st.session_state[verdict_key] = "skip"
                        if true_key not in st.session_state:
                            st.session_state[true_key] = ""

                        vcols = st.columns([1, 1, 1, 1, 2])
                        if vcols[0].button("✅", key=f"btn_ok_{doc_id}_{f}"):
                            st.session_state[verdict_key] = "correct"
                        if vcols[1].button("❌", key=f"btn_no_{doc_id}_{f}"):
                            st.session_state[verdict_key] = "incorrect"
                        if vcols[2].button("✏️", key=f"btn_edit_{doc_id}_{f}"):
                            st.session_state[verdict_key] = "incorrect"
                        if vcols[3].button("∅", key=f"btn_missing_{doc_id}_{f}"):
                            st.session_state[verdict_key] = "missing"

                        st.selectbox(
                            "Status",
                            ["correct", "incorrect", "missing", "skip"],
                            index=["correct", "incorrect", "missing", "skip"].index(st.session_state[verdict_key]),
                            key=verdict_key,
                            label_visibility="collapsed"
                        )

                        if st.session_state[verdict_key] in ["incorrect", "missing"]:
                            st.text_input(
                                "True value (optional bei missing/incorrect)",
                                key=true_key,
                                label_visibility="collapsed",
                                placeholder="korrekter Wert (wenn du ihn kennst)"
                            )

                        st.divider()

                    # Save doc validations
                    if st.button("💾 Save this document", type="primary", use_container_width=True):
                        # Remove existing validations for this doc to allow re-save
                        st.session_state.validations = [r for r in st.session_state.validations if r["doc_id"] != doc_id]

                        for f in fields_sorted:
                            pred_val = pred.fields[f].value
                            conf = pred.fields[f].confidence
                            verdict = st.session_state.get(f"verdict_{doc_id}_{f}", "skip")
                            true_val = st.session_state.get(f"true_{doc_id}_{f}", "")

                            st.session_state.validations.append({
                                "doc_id": doc_id,
                                "field": f,
                                "pred_value": pred_val,
                                "confidence": conf,
                                "verdict": verdict,
                                "true_value": true_val if true_val.strip() != "" else None
                            })

                        # Auto-advance
                        st.session_state.idx = min(len(ordered) - 1, i + 1)
                        st.success("Gespeichert.")
                        st.rerun()

# ----------------------------
# Stage 3: Results + Export
# ----------------------------
else:
    st.subheader("Results")

    validations = st.session_state.validations
    field_df, doc_df, overall = compute_metrics(validations)

    kcols = st.columns(3)
    kcols[0].metric("Validations", len(validations))
    kcols[1].metric("Docs (PDFs)", len(st.session_state.ordered_ids))
    kcols[2].metric("Overall accuracy (predicted fields)", f"{overall['overall_accuracy_on_predicted']*100:.1f}%")

    st.divider()

    st.write("### Field metrics")
    if field_df.empty:
        st.info("Noch keine Validations gespeichert.")
    else:
        st.dataframe(field_df.sort_values(["f1", "precision", "recall"], ascending=False), use_container_width=True)

    st.write("### Document metrics")
    if not doc_df.empty:
        st.dataframe(doc_df.sort_values(["doc_accuracy"], ascending=False), use_container_width=True)

    st.divider()
    st.write("### Export")

    export_obj = {
        "run_meta": st.session_state.run_meta,
        "notes_by_doc": st.session_state.get("doc_notes", {}),
        "validations": validations
    }
    export_json = json.dumps(export_obj, ensure_ascii=False, indent=2).encode("utf-8")

    st.download_button(
        "⬇️ Download validations.json",
        data=export_json,
        file_name=f"{st.session_state.run_meta['run_name']}_validations.json",
        mime="application/json",
        use_container_width=True
    )

    if not field_df.empty:
        st.download_button(
            "⬇️ Download field_metrics.csv",
            data=field_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{st.session_state.run_meta['run_name']}_field_metrics.csv",
            mime="text/csv",
            use_container_width=True
        )

    if not doc_df.empty:
        st.download_button(
            "⬇️ Download doc_metrics.csv",
            data=doc_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{st.session_state.run_meta['run_name']}_doc_metrics.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.divider()
    c1, c2 = st.columns(2)
    if c1.button("⬅️ Back to quiz", use_container_width=True):
        st.session_state.stage = "quiz"
        st.rerun()
    if c2.button("🔄 New run (reset)", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()