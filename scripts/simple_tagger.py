"""Simple Tagger - a minimal image tagger tab (ONNX only, no tensorflow)."""
import csv
import json
import os

import gradio as gr
import numpy as np
from PIL import Image

from modules import script_callbacks
from modules.paths_internal import data_path, models_path

EXT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SETTINGS_PATH = os.path.join(EXT_ROOT, "settings.json")
LOCALES_DIR = os.path.join(EXT_ROOT, "locales")
EXT_MODELS_DIR = os.path.join(EXT_ROOT, "models")

SAVE_DIR = os.path.join(data_path, "output", "tagger_output", "simple-tagger")
IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp")

_SETTINGS_DEFAULTS = {
    "model": None,  # resolved to DEFAULT_MODEL in on_ui_tabs
    "gen_thresh": 0.35,
    "char_thresh": 0.85,
    "exclude": "",
    "save": True,
    "lang": "en",
    "custom_dir": "",
}


def _safe_float(v, default):
    try:
        return float(v)
    except Exception:
        return default


def load_settings():
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        merged = dict(_SETTINGS_DEFAULTS)
        for k in _SETTINGS_DEFAULTS:
            if k in data:
                merged[k] = data[k]
        print(f"[simple-tagger] load: {merged} <- {SETTINGS_PATH}")
        return merged
    except Exception as e:
        print(f"[simple-tagger] load failed ({type(e).__name__}: {e}), defaults used. Path: {SETTINGS_PATH}")
        return dict(_SETTINGS_DEFAULTS)


def save_settings(data):
    try:
        os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
        # Atomic write: load_settings never sees a partially written file.
        tmp_path = SETTINGS_PATH + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, SETTINGS_PATH)
        print(f"[simple-tagger] saved: {data} -> {SETTINGS_PATH}")
        return True
    except Exception as e:
        print(f"[simple-tagger] SAVE FAILED ({type(e).__name__}: {e}). Path: {SETTINGS_PATH}")
        return False


# ---------- i18n ----------

_LOCALE_CACHE = {}
_LANG = "en"  # active UI language, set from settings in on_ui_tabs


def _load_locale_dict(code):
    """Load one locale file, with English as the fallback for missing keys."""
    if code not in _LOCALE_CACHE:
        data = {}
        try:
            with open(os.path.join(LOCALES_DIR, code + ".json"), encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[simple-tagger] locale '{code}' not loaded ({type(e).__name__}: {e})")
        merged = {}
        if code != "en":
            merged.update(_load_locale_dict("en"))
        merged.update({k: v for k, v in data.items() if not k.startswith("_")})
        _LOCALE_CACHE[code] = merged
    return _LOCALE_CACHE[code]


def available_languages():
    """Every *.json in locales/ automatically becomes a selectable language."""
    langs = {}
    try:
        for fn in sorted(os.listdir(LOCALES_DIR)):
            if not fn.endswith(".json"):
                continue
            code = fn[:-5]
            try:
                with open(os.path.join(LOCALES_DIR, fn), encoding="utf-8") as f:
                    data = json.load(f)
                langs[code] = data.get("_name", code)
            except Exception:
                continue
    except Exception:
        pass
    if "en" not in langs:
        langs["en"] = "English"
    return langs


def T(key, lang=None):
    """Translate a key; falls back to English, then to the key itself."""
    code = lang or _LANG
    text = _load_locale_dict(code).get(key)
    if text is None and code != "en":
        text = _load_locale_dict("en").get(key)
    return key if text is None else text


# ---------- Model search order ----------
# 1. user custom folder (from settings)
# 2. <models_path>/tagger_models (shared Forge model storage)
# 3. models/ inside this extension
# 4. HuggingFace download into <models_path>/tagger_models/<dirname>

_CUSTOM_DIR = ""


def _wd_search_dirs():
    dirs = []
    if _CUSTOM_DIR and os.path.isdir(_CUSTOM_DIR):
        dirs.append(_CUSTOM_DIR)
    dirs.append(os.path.join(models_path, "tagger_models"))
    dirs.append(EXT_MODELS_DIR)
    return dirs


def _camie_search_dirs():
    return list(_wd_search_dirs())


def _resolve_wd_local(basename, dirname):
    """Find basename.onnx + basename.csv in the search dirs or the HF cache layout."""
    for d in _wd_search_dirs():
        try:
            onnx = os.path.join(d, basename + ".onnx")
            csv = os.path.join(d, basename + ".csv")
        except Exception:
            continue
        if os.path.exists(onnx) and os.path.exists(csv):
            return onnx, csv
    # Canonical HuggingFace cache layout: tagger_models/<dirname>/model.onnx
    if dirname:
        onnx = os.path.join(models_path, "tagger_models", dirname, "model.onnx")
        csv = os.path.join(models_path, "tagger_models", dirname, "selected_tags.csv")
        if os.path.exists(onnx) and os.path.exists(csv):
            return onnx, csv
    return None, None


def _resolve_camie_local(basename):
    """Find basename.onnx + basename-metadata.json in the search dirs."""
    for d in _camie_search_dirs():
        try:
            onnx = os.path.join(d, basename + ".onnx")
            meta = os.path.join(d, basename + "-metadata.json")
        except Exception:
            continue
        if os.path.exists(onnx) and os.path.exists(meta):
            return onnx, meta
    return None, None


MODELS = {
    "WD ViT v3 (local)": {
        "type": "wd",
        "repo": "SmilingWolf/wd-vit-tagger-v3",
        "dirname": "wd-vit-tagger-v3",
        "basename": "wd-vit-tagger-v3",
    },
    "Camie v2 (local)": {
        "type": "camie",
        "basename": "camie-tagger-v2",
    },
    "WD EVA02 Large v3": {
        "type": "wd",
        "repo": "SmilingWolf/wd-eva02-large-tagger-v3",
        "dirname": "wd-eva02-large-tagger-v3",
        "basename": "wd-eva02-large-tagger-v3",
    },
    "WD SwinV2 v3": {
        "type": "wd",
        "repo": "SmilingWolf/wd-swinv2-tagger-v3",
        "dirname": "wd-swinv2-tagger-v3",
        "basename": "wd-swinv2-tagger-v3",
    },
    "WD ConvNext v3": {
        "type": "wd",
        "repo": "SmilingWolf/wd-convnext-tagger-v3",
        "dirname": "wd-convnext-tagger-v3",
        "basename": "wd-convnext-tagger-v3",
    },
    "WD MoAT v2": {
        # The only v2-family model without a direct v3 successor, lightweight (~326MB).
        "type": "wd",
        "repo": "SmilingWolf/wd-v1-4-moat-tagger-v2",
        "dirname": "wd-v1-4-moat-tagger-v2",
        "basename": "wd-v1-4-moat-tagger-v2",
    },
    # --- Deliberately NOT added, to avoid duplicates: ---
    # The v3 dataset (up to 2024-02-28) is newer than v2 (2023) in vocabulary and F1,
    # so the same-named v3 models supersede wd-v1-4-vit-tagger-v2,
    # wd-v1-4-swinv2-tagger-v2 and wd-v1-4-convnext-tagger-v2.
    # wd-v1-4-convnextv2-tagger-v2 is an intermediate arch, superseded by ConvNext v3.
    # wd-v1-4-vit-tagger, wd-v1-4-convnext-tagger (v1 without -v2) are outdated.
}

_AUTO_LABELS = set()


def _scan_extra_locals():
    """Any extra *.onnx + *.csv (WD) or *-metadata.json (Camie) found in the
    search dirs shows up in the model list automatically."""
    for label in list(_AUTO_LABELS):
        MODELS.pop(label, None)
    _AUTO_LABELS.clear()
    known_basenames = {
        cfg.get("basename")
        for cfg in MODELS.values()
        if cfg.get("type") == "wd" and cfg.get("basename")
    }
    for d in _wd_search_dirs():
        if not d or not os.path.isdir(d):
            continue
        try:
            files = os.listdir(d)
        except Exception:
            continue
        for fn in files:
            if not fn.endswith(".onnx"):
                continue
            base = fn[:-5]
            if base in known_basenames:
                continue
            csv_path = os.path.join(d, base + ".csv")
            if not os.path.exists(csv_path):
                continue
            label = f"{base} (auto-local)"
            if label not in MODELS:
                MODELS[label] = {
                    "type": "wd",
                    "basename": base,
                    "dirname": base,
                    # Best-effort guess at the standard SmilingWolf naming, so that
                    # _ensure_wd can re-download if the file is deleted.
                    "repo": f"SmilingWolf/{base}",
                }
                _AUTO_LABELS.add(label)
                known_basenames.add(base)
    # Camie: *.onnx + *-metadata.json
    for d in _camie_search_dirs():
        if not d or not os.path.isdir(d):
            continue
        try:
            files = os.listdir(d)
        except Exception:
            continue
        for fn in files:
            if not fn.endswith(".onnx"):
                continue
            base = fn[:-5]  # e.g. camie-tagger-v2
            meta = os.path.join(d, base + "-metadata.json")
            if not os.path.exists(meta):
                continue
            if any(
                cfg.get("type") == "camie" and cfg.get("basename") == base
                for cfg in MODELS.values()
            ):
                continue
            label = f"{base} (auto-local)"
            if label not in MODELS:
                MODELS[label] = {"type": "camie", "basename": base}
                _AUTO_LABELS.add(label)


def rescan_models():
    """Re-read the custom folder setting and re-scan; returns sorted model names."""
    global _CUSTOM_DIR
    try:
        _CUSTOM_DIR = (load_settings().get("custom_dir") or "").strip()
    except Exception:
        _CUSTOM_DIR = ""
    _scan_extra_locals()
    return sorted(MODELS.keys())


DEFAULT_MODEL = "WD ViT v3 (local)"

_sessions = {}
_wd_labels = {}
_camie_metas = {}


def _providers():
    import onnxruntime as ort

    available = ort.get_available_providers()
    providers = []
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers


def _get_session(key, onnx_path):
    if key not in _sessions:
        import onnxruntime as ort

        _sessions[key] = ort.InferenceSession(onnx_path, providers=_providers())
    return _sessions[key]


# ---------- WD (csv) ----------

def _ensure_wd(cfg):
    onnx_path, csv_path = _resolve_wd_local(cfg.get("basename", ""), cfg.get("dirname", ""))
    if onnx_path:
        return onnx_path, csv_path
    if not cfg.get("repo") or not cfg.get("dirname"):
        raise FileNotFoundError(T("msg_no_repo"))
    model_dir = os.path.join(models_path, "tagger_models", cfg["dirname"])
    onnx_path = os.path.join(model_dir, "model.onnx")
    csv_path = os.path.join(model_dir, "selected_tags.csv")
    if not (os.path.exists(onnx_path) and os.path.exists(csv_path)):
        os.makedirs(model_dir, exist_ok=True)
        from huggingface_hub import hf_hub_download

        hf_hub_download(repo_id=cfg["repo"], filename="model.onnx", local_dir=model_dir)
        hf_hub_download(repo_id=cfg["repo"], filename="selected_tags.csv", local_dir=model_dir)
    return onnx_path, csv_path


def _load_wd_labels(key, csv_path):
    if key not in _wd_labels:
        names, cats = [], []
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                names.append(row["name"])
                try:
                    cats.append(int(row["category"]))
                except (KeyError, ValueError):
                    cats.append(0)
        _wd_labels[key] = (names, cats)
    return _wd_labels[key]


def _wd_input_size(session):
    try:
        for dim in session.get_inputs()[0].shape[1:3]:
            if isinstance(dim, int) and dim > 0:
                return dim
    except Exception:
        pass
    return 448


def _preprocess_wd(image: Image.Image, size: int) -> np.ndarray:
    img = image.convert("RGB")
    w, h = img.size
    ratio = size / max(w, h)
    new_size = (max(1, int(w * ratio)), max(1, int(h * ratio)))
    img = img.resize(new_size, Image.LANCZOS)
    square = Image.new("RGB", (size, size), (255, 255, 255))
    square.paste(img, ((size - new_size[0]) // 2, (size - new_size[1]) // 2))
    arr = np.asarray(square, dtype=np.float32)[:, :, ::-1]  # RGB->BGR
    return np.expand_dims(arr, axis=0)


def _run_wd(model_name, cfg, image, gen_thresh, char_thresh, excluded):
    onnx_path, csv_path = _ensure_wd(cfg)
    names, cats = _load_wd_labels(model_name, csv_path)
    session = _get_session(model_name, onnx_path)
    size = _wd_input_size(session)
    probs = session.run(None, {session.get_inputs()[0].name: _preprocess_wd(image, size)})[0][0]
    scored = []
    for name, cat, p in zip(names, cats, probs):
        if cat == 9:
            continue
        if float(p) < (char_thresh if cat in (3, 4) else gen_thresh):
            continue
        if name.strip().lower().replace(" ", "_") in excluded:
            continue
        scored.append((name, float(p), cat))
    scored.sort(key=lambda x: (x[2] not in (3, 4), -x[1]))
    out = ", ".join(n for n, _, _ in scored)
    info = T("msg_found").format(model=model_name, n=len(scored), size=size)
    return out, info


# ---------- Camie (json meta) ----------

def _resolve_camie_cfg(model_name, cfg):
    """Return (onnx_path, meta_path) or raise with a user-facing message."""
    basename = cfg.get("basename", "")
    found = _resolve_camie_local(basename) if basename else (None, None)
    if found[0]:
        return found
    # Back-compat: explicit absolute paths (old settings / old configs).
    if cfg.get("local_onnx") and os.path.exists(cfg["local_onnx"]) \
            and cfg.get("local_meta") and os.path.exists(cfg["local_meta"]):
        return cfg["local_onnx"], cfg["local_meta"]
    raise FileNotFoundError(T("msg_not_found").format(path=basename + ".onnx"))


def _load_camie_meta(model_name, meta_path):
    key = (model_name, meta_path)
    if key not in _camie_metas:
        with open(meta_path, encoding="utf-8") as f:
            _camie_metas[key] = json.load(f)
    return _camie_metas[key]


def _preprocess_camie(image: Image.Image, size: int) -> np.ndarray:
    img = image.convert("RGB").resize((size, size), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    return np.expand_dims(np.transpose(arr, (2, 0, 1)), axis=0)


def _run_camie(model_name, cfg, image, thresh, excluded):
    onnx_path, meta_path = _resolve_camie_cfg(model_name, cfg)
    meta = _load_camie_meta(model_name, meta_path)
    session = _get_session(model_name, onnx_path)
    size = int(meta["model_info"]["img_size"])
    logits = session.run(
        [session.get_outputs()[0].name],
        {session.get_inputs()[0].name: _preprocess_camie(image, size)},
    )[0][0]
    probs = 1 / (1 + np.exp(-logits))

    mapping = meta["dataset_info"]["tag_mapping"]
    if "idx_to_tag" in mapping:
        idx_map = {int(k): v for k, v in mapping["idx_to_tag"].items()}
        tags = [idx_map[i] for i in range(len(idx_map))]
    else:
        tags = [t for t, _ in sorted(mapping["tag_to_idx"].items(), key=lambda kv: kv[1])]
    tag_to_cat = mapping.get("tag_to_category", {})

    scored = []
    for i, p in enumerate(probs):
        if float(p) < thresh:
            continue
        name = tags[i]
        if name.strip().lower() in excluded:
            continue
        scored.append((name, float(p), tag_to_cat.get(name, "general")))
    scored.sort(key=lambda x: -x[1])
    out = ", ".join(n for n, _, _ in scored)
    info = T("msg_found").format(model=model_name, n=len(scored), size=size)
    return out, info


def normalize_excluded(text):
    return {t.strip().lower().replace(" ", "_") for t in text.split(",") if t.strip()}


def open_save_dir():
    import subprocess
    import sys

    try:
        os.makedirs(SAVE_DIR, exist_ok=True)
        if os.name == "nt":
            os.startfile(SAVE_DIR)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", SAVE_DIR])
        else:
            subprocess.Popen(["xdg-open", SAVE_DIR])
        return T("msg_folder").format(path=SAVE_DIR)
    except Exception as e:
        return T("msg_folder_failed").format(path=SAVE_DIR, err=e)


def list_saved_images():
    """File names from SAVE_DIR, newest first."""
    try:
        files = [
            f for f in os.listdir(SAVE_DIR)
            if f.lower().endswith(IMG_EXTS) and os.path.isfile(os.path.join(SAVE_DIR, f))
        ]
    except Exception:
        return []
    files.sort(key=lambda f: os.path.getmtime(os.path.join(SAVE_DIR, f)), reverse=True)
    return files


_gallery_files = []  # file order matches thumbnail order, newest first


def build_gallery_value():
    global _gallery_files
    _gallery_files = list_saved_images()
    return [(os.path.join(SAVE_DIR, f), f) for f in _gallery_files]


def on_gallery_select(evt: gr.SelectData):
    try:
        idx = evt.index
        if isinstance(idx, tuple):
            idx = idx[0]
        path = os.path.join(SAVE_DIR, _gallery_files[idx])
    except Exception:
        return gr.update(), gr.update(), gr.update()
    if not os.path.isfile(path):
        return gr.update(), gr.update(), gr.update()
    try:
        # Picking an image closes the modal right away.
        return Image.open(path).convert("RGB"), gr.update(visible=False), False
    except Exception:
        return gr.update(), gr.update(), gr.update()


def refresh_gallery():
    return gr.update(value=build_gallery_value())


def open_gallery(is_open):
    is_open = not is_open
    if is_open:
        return gr.update(visible=True), gr.update(value=build_gallery_value()), is_open
    return gr.update(visible=False), gr.update(), is_open


def close_gallery():
    return gr.update(visible=False), False


def find_duplicate_image(image):
    """Find a pixel-identical image in SAVE_DIR to avoid piling up copies."""
    import hashlib

    try:
        rgb = image.convert("RGB")
        digest = hashlib.md5(rgb.tobytes()).hexdigest()
        size = rgb.size
    except Exception:
        return None
    try:
        files = os.listdir(SAVE_DIR)
    except Exception:
        return None
    for f in files[:1000]:
        if not f.lower().endswith(IMG_EXTS):
            continue
        p = os.path.join(SAVE_DIR, f)
        try:
            with Image.open(p) as im:
                cand = im.convert("RGB")
                if cand.size != size:
                    continue
                if hashlib.md5(cand.tobytes()).hexdigest() == digest:
                    return p
        except Exception:
            continue
    return None


def save_result(image, tags, model_name):
    import datetime
    import re

    os.makedirs(SAVE_DIR, exist_ok=True)
    dup = find_duplicate_image(image)
    if dup:
        return dup, True
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_model = re.sub(r"[^\w\-]+", "_", model_name).strip("_")[:40]
    base = f"{stamp}_{safe_model}"
    img_path = os.path.join(SAVE_DIR, base + ".png")
    # Don't overwrite if the second collided.
    i = 1
    while os.path.exists(img_path):
        i += 1
        img_path = os.path.join(SAVE_DIR, f"{base}_{i}.png")
    image.convert("RGB").save(img_path)
    return img_path, False


def _collect_ui_state(model_name, gen_thresh, char_thresh, exclude_text, save_enabled):
    """Pack UI values; language and custom_dir are managed by their own handlers."""
    s = load_settings()
    return {
        "model": model_name,
        "gen_thresh": float(gen_thresh),
        "char_thresh": float(char_thresh),
        "exclude": exclude_text or "",
        "save": bool(save_enabled),
        "lang": s.get("lang", "en"),
        "custom_dir": s.get("custom_dir", ""),
    }


def save_ui_state(model_name, gen_thresh, char_thresh, exclude_text, save_enabled):
    """Autosave on any parameter change (without running inference).

    Returns a short status into a hidden textbox: Gradio 3.x requires a real
    output component, events with outputs=None silently do nothing.
    """
    try:
        ok = save_settings(_collect_ui_state(model_name, gen_thresh, char_thresh, exclude_text, save_enabled))
        return "ok" if ok else "save failed (see console)"
    except Exception as e:
        print(f"[simple-tagger] save_ui_state skipped ({type(e).__name__}: {e})")
        return "skip"


def interrogate(image, model_name, gen_thresh, char_thresh, exclude_text, save_enabled):
    # Persist the user's choice so it survives a UI restart.
    try:
        save_settings(_collect_ui_state(model_name, gen_thresh, char_thresh, exclude_text, save_enabled))
    except Exception:
        pass
    dropdown_update = gr.update()
    if image is None:
        return "", T("msg_need_image"), dropdown_update
    if model_name not in MODELS:
        model_name = DEFAULT_MODEL if DEFAULT_MODEL in MODELS else sorted(MODELS.keys())[0]
    cfg = MODELS[model_name]
    excluded = normalize_excluded(exclude_text or "")
    try:
        if cfg["type"] == "camie":
            out, info = _run_camie(model_name, cfg, image, gen_thresh, excluded)
        else:
            out, info = _run_wd(model_name, cfg, image, gen_thresh, char_thresh, excluded)
    except Exception as e:
        return "", T("msg_error").format(err=f"{type(e).__name__}: {e}"), dropdown_update
    if save_enabled and out:
        try:
            saved, dup = save_result(image, out, model_name)
            dropdown_update = gr.update(value=build_gallery_value())
            info += T("msg_already").format(path=saved) if dup else T("msg_saved_to").format(path=saved)
        except Exception as e:
            info += T("msg_save_failed").format(err=e)
    return out, info, dropdown_update


# ---------- Settings section handlers ----------

def set_language(lang):
    """Persist the language; the tab relabels itself live, no restart needed."""
    global _LANG
    langs = available_languages()
    if lang not in langs:
        lang = "en"
    _LANG = lang
    try:
        s = load_settings()
        s["lang"] = lang
        save_settings(s)
    except Exception:
        pass
    return T("msg_lang_applied").format(lang=langs[lang])


def set_custom_dir(custom_dir, model_name):
    """Persist the custom models folder, rescan, and refresh the model list."""
    global _CUSTOM_DIR
    _CUSTOM_DIR = (custom_dir or "").strip()
    try:
        s = load_settings()
        s["custom_dir"] = _CUSTOM_DIR
        save_settings(s)
    except Exception:
        pass
    names = rescan_models()
    value = model_name if model_name in MODELS else (DEFAULT_MODEL if DEFAULT_MODEL in MODELS else names[0])
    return (
        gr.update(choices=names, value=value),
        T("msg_custom_saved").format(n=len(names)),
    )


def rescan_models_ui(model_name):
    names = rescan_models()
    value = model_name if model_name in MODELS else (DEFAULT_MODEL if DEFAULT_MODEL in MODELS else names[0])
    return (
        gr.update(choices=names, value=value),
        T("msg_rescanned").format(n=len(names)),
    )


def download_selected_model(model_name):
    """Download the selected model from HuggingFace, then rescan the list."""
    cfg = MODELS.get(model_name)
    if cfg is None:
        return T("msg_error").format(err=model_name), gr.update()
    try:
        if cfg["type"] == "camie":
            raise FileNotFoundError(T("msg_no_repo"))
        onnx_path, _ = _ensure_wd(cfg)
        names = rescan_models()
        return (
            T("msg_downloaded").format(path=os.path.dirname(onnx_path)),
            gr.update(choices=names, value=model_name),
        )
    except Exception as e:
        return T("msg_download_failed").format(err=f"{type(e).__name__}: {e}"), gr.update()


# SVG icons (inline, no emoji): CSS-mask inherits the theme color, so they look
# correct in both dark and light Forge themes.
_FOLDER_SVG = "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'><path d='M10 4H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2h-8l-2-2z'/></svg>"
_REFRESH_SVG = "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'><path d='M17.65 6.35C16.2 4.9 14.21 4 12 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08c-.82 2.33-3.04 4-5.65 4-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z'/></svg>"
_GRID_SVG = "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'><path d='M3 3v8h8V3H3zm6 6H5V5h4v4zm-6 4v8h8v-8H3zm6 6H5v-4h4v4zm4-16v8h8V3h-8zm6 6h-4V5h4v4zm-6 4v8h8v-8h-8zm6 6h-4v-4h4v4z'/></svg>"
_ICON_CSS = (
    "<style>"
    "#simple_tagger_open_folder::before, #simple_tagger_refresh_list::before, #simple_tagger_open_gallery::before {"
    "content:''; display:inline-block; width:1em; height:1em; margin-right:.45em;"
    "vertical-align:-0.125em; background-color:currentColor;"
    "-webkit-mask:no-repeat center/contain; mask:no-repeat center/contain;}"
    "#simple_tagger_open_folder::before {"
    "-webkit-mask-image:url(\"" + _FOLDER_SVG + "\"); mask-image:url(\"" + _FOLDER_SVG + "\");}"
    "#simple_tagger_refresh_list::before {"
    "-webkit-mask-image:url(\"" + _REFRESH_SVG + "\"); mask-image:url(\"" + _REFRESH_SVG + "\");}"
    "#simple_tagger_open_gallery::before {"
    "-webkit-mask-image:url(\"" + _GRID_SVG + "\"); mask-image:url(\"" + _GRID_SVG + "\");}"
    "#simple_tagger_gallery_wrap {"
    "position:fixed; top:50%; left:50%; transform:translate(-50%,-50%);"
    "width:min(920px,94vw); max-height:92vh; overflow-y:auto; z-index:1000;"
    "background:var(--background-fill-primary,#1f2937);"
    "border:1px solid var(--border-color-primary,#444); border-radius:10px; padding:12px;"
    "box-shadow:0 0 0 9999px rgba(0,0,0,.65), 0 12px 48px rgba(0,0,0,.55);}"
    "</style>"
)


def on_ui_tabs():
    global _LANG, _CUSTOM_DIR
    s = load_settings()
    _LANG = s.get("lang") or "en"
    if _LANG not in available_languages():
        _LANG = "en"
    _CUSTOM_DIR = (s.get("custom_dir") or "").strip()
    _scan_extra_locals()
    model_names = sorted(MODELS.keys())
    initial_model = s.get("model") if s.get("model") in MODELS else DEFAULT_MODEL
    if initial_model not in MODELS:
        initial_model = model_names[0]
    lang_names = available_languages()
    with gr.Blocks(analytics_enabled=False) as tab:
        title = gr.Markdown(T("title_md"))
        gr.HTML(_ICON_CSS)
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label=T("image_label"))
                open_btn = gr.Button(T("saved_images_btn"), elem_id="simple_tagger_open_gallery")
                modal_state = gr.State(value=False)
                with gr.Column(visible=False, elem_id="simple_tagger_gallery_wrap") as modal:
                    with gr.Row():
                        refresh_btn = gr.Button(T("refresh_list_btn"), elem_id="simple_tagger_refresh_list")
                        close_btn = gr.Button(T("close_btn"))
                    gallery = gr.Gallery(
                        value=[],
                        label=T("gallery_label"),
                        columns=6,
                        height=560,
                        allow_preview=False,
                        show_download_button=False,
                        object_fit="cover",
                    )
                model = gr.Dropdown(
                    choices=model_names, value=initial_model, label=T("model_label")
                )
                gen = gr.Slider(0.0, 1.0, value=_safe_float(s.get("gen_thresh"), 0.35), step=0.01, label=T("gen_label"))
                char = gr.Slider(0.0, 1.0, value=_safe_float(s.get("char_thresh"), 0.85), step=0.01, label=T("char_label"))
                excl = gr.Textbox(
                    label=T("exclude_label"),
                    placeholder=T("exclude_placeholder"),
                    lines=2,
                    value=s.get("exclude", "") or "",
                )
                save = gr.Checkbox(value=bool(s.get("save", True)), label=T("save_label"))
                with gr.Row():
                    btn = gr.Button(T("interrogate_btn"), variant="primary")
                    folder_btn = gr.Button(T("open_folder_btn"), elem_id="simple_tagger_open_folder")
                with gr.Accordion(T("settings_title"), open=False) as settings_acc:
                    lang_dd = gr.Dropdown(
                        choices=sorted(lang_names.keys()),
                        value=_LANG,
                        label=T("lang_label"),
                    )
                    custom_tb = gr.Textbox(
                        label=T("custom_dir_label"),
                        placeholder=T("custom_dir_placeholder"),
                        lines=1,
                        value=_CUSTOM_DIR,
                    )
                    with gr.Row():
                        rescan_btn = gr.Button(T("rescan_btn"))
                        download_btn = gr.Button(T("download_btn"))
                    settings_status = gr.Textbox(label=T("settings_status_label"), lines=2)
            with gr.Column():
                out = gr.Textbox(label=T("tags_label"), lines=10, show_copy_button=True)
                info = gr.Textbox(label=T("status_label"), lines=2)
                # Hidden autosave status: Gradio 3.x requires a real output,
                # otherwise an event with outputs=None silently never fires.
                save_status = gr.Textbox(visible=False, value="ok")
        # Opt out of Forge UiLoadsave (ui-config.json): the single source of truth
        # is the extension's settings.json. Otherwise Forge overwrites component
        # values on startup with stale defaults from ui-config.json.
        for _comp in (model, gen, char, excl, save, out, info, save_status,
                      lang_dd, custom_tb, settings_status):
            _comp.do_not_save_to_config = True
        # Components relabeled live when the language changes.
        _relabel = [title, inp, open_btn, refresh_btn, close_btn, gallery, model,
                    gen, char, excl, save, btn, folder_btn, out, info, settings_acc,
                    lang_dd, custom_tb, rescan_btn, download_btn, settings_status]

        def _apply_language(lang):
            status = set_language(lang)
            return (
                gr.update(value=T("title_md")),
                gr.update(label=T("image_label")),
                gr.update(value=T("saved_images_btn")),
                gr.update(value=T("refresh_list_btn")),
                gr.update(value=T("close_btn")),
                gr.update(label=T("gallery_label")),
                gr.update(label=T("model_label")),
                gr.update(label=T("gen_label")),
                gr.update(label=T("char_label")),
                gr.update(label=T("exclude_label"), placeholder=T("exclude_placeholder")),
                gr.update(label=T("save_label")),
                gr.update(value=T("interrogate_btn")),
                gr.update(value=T("open_folder_btn")),
                gr.update(label=T("tags_label")),
                gr.update(label=T("status_label")),
                gr.update(label=T("settings_title")),
                gr.update(label=T("lang_label")),
                gr.update(label=T("custom_dir_label"), placeholder=T("custom_dir_placeholder")),
                gr.update(value=T("rescan_btn")),
                gr.update(value=T("download_btn")),
                gr.update(label=T("settings_status_label"), value=status),
            )

        gallery.select(fn=on_gallery_select, outputs=[inp, modal, modal_state])
        open_btn.click(fn=open_gallery, inputs=[modal_state], outputs=[modal, gallery, modal_state])
        close_btn.click(fn=close_gallery, inputs=[], outputs=[modal, modal_state])
        refresh_btn.click(fn=refresh_gallery, inputs=[], outputs=[gallery])
        btn.click(fn=interrogate, inputs=[inp, model, gen, char, excl, save], outputs=[out, info, gallery])
        folder_btn.click(fn=open_save_dir, inputs=[], outputs=[info])
        lang_dd.change(fn=_apply_language, inputs=[lang_dd], outputs=_relabel)
        custom_tb.change(fn=set_custom_dir, inputs=[custom_tb, model], outputs=[model, settings_status])
        rescan_btn.click(fn=rescan_models_ui, inputs=[model], outputs=[model, settings_status])
        download_btn.click(fn=download_selected_model, inputs=[model], outputs=[settings_status, model])
        # Autosave on parameter change so values survive a UI restart
        # without a mandatory Interrogate click.
        _ui_inputs = [model, gen, char, excl, save]
        model.change(fn=save_ui_state, inputs=_ui_inputs, outputs=[save_status])
        excl.change(fn=save_ui_state, inputs=_ui_inputs, outputs=[save_status])
        save.change(fn=save_ui_state, inputs=_ui_inputs, outputs=[save_status])
        # Sliders: write once on release, not a storm of events while dragging;
        # change is kept for keyboard input.
        gen.change(fn=save_ui_state, inputs=_ui_inputs, outputs=[save_status])
        gen.release(fn=save_ui_state, inputs=_ui_inputs, outputs=[save_status])
        char.change(fn=save_ui_state, inputs=_ui_inputs, outputs=[save_status])
        char.release(fn=save_ui_state, inputs=_ui_inputs, outputs=[save_status])
    return [(tab, "Simple Tagger", "simple_tagger")]


script_callbacks.on_ui_tabs(on_ui_tabs)
