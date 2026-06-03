"""Lemonade Server image generation backend for Hermes ``image_generate``.

Uses Lemonade's OpenAI-compatible ``POST /images/generations`` endpoint
(stable-diffusion.cpp). Models are discovered from ``GET /models`` when the
server is reachable, with a static catalog fallback matching Lemonade's
StableDiffusion.cpp suggestions.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    resolve_aspect_ratio,
    save_b64_image,
    success_response,
)

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://127.0.0.1:13305/api/v1"
DEFAULT_MODEL = "SD-Turbo"
DEFAULT_TIMEOUT = 600.0


def _common_dir() -> Path:
    here = Path(__file__).resolve().parent
    bundled = here / "common"
    if bundled.is_dir():
        return bundled
    return here.parent / "common"


def _load_common(name: str):
    mod_name = f"hermes_image_gen_common_{name}"
    cached = sys.modules.get(mod_name)
    if cached is not None:
        return cached
    path = _common_dir() / f"{name}.py"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"missing common module: {name}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _ensure_server() -> Dict[str, Any]:
    runtime = _load_common("runtime")
    return runtime.ensure_lemonade_running(_lemonade_base_url(), wait_timeout=_request_timeout())

# Lemonade Model Manager StableDiffusion.cpp suggestions (plus common aliases).
_STATIC_MODELS: Dict[str, Dict[str, Any]] = {
    "Flux-2-Klein-4B": {
        "display": "Flux 2 Klein 4B",
        "speed": "~30-120s",
        "strengths": "Fast Flux, image editing capable",
        "size_gb": 16.1,
        "defaults": {"width": 512, "height": 512, "steps": 20, "cfg_scale": 4.0},
    },
    "Flux-2-Klein-9B-GGUF": {
        "display": "Flux 2 Klein 9B GGUF",
        "speed": "~45-180s",
        "strengths": "Higher-quality Flux Klein",
        "size_gb": 19.0,
        "defaults": {"width": 512, "height": 512, "steps": 20, "cfg_scale": 4.0},
    },
    "Qwen-Image-2512-GGUF": {
        "display": "Qwen Image 2512 GGUF",
        "speed": "~60-240s",
        "strengths": "Complex prompts and text rendering",
        "size_gb": 19.4,
        "defaults": {"width": 512, "height": 512, "steps": 20, "cfg_scale": 4.0},
    },
    "Qwen-Image-GGUF": {
        "display": "Qwen Image GGUF",
        "speed": "~60-240s",
        "strengths": "LLM-based image model",
        "size_gb": 18.2,
        "defaults": {"width": 512, "height": 512, "steps": 20, "cfg_scale": 4.0},
    },
    "SD-1.5": {
        "display": "Stable Diffusion 1.5",
        "speed": "~30-180s",
        "strengths": "Classic SD, lightweight",
        "size_gb": 7.7,
        "defaults": {"width": 512, "height": 512, "steps": 20, "cfg_scale": 7.5},
    },
    "SD-Turbo": {
        "display": "SD Turbo",
        "speed": "~5-60s",
        "strengths": "Very fast iteration (4 steps)",
        "size_gb": 5.2,
        "defaults": {"width": 512, "height": 512, "steps": 4, "cfg_scale": 1.0},
    },
    "SDXL-Base-1.0": {
        "display": "SDXL Base 1.0",
        "speed": "~60-300s",
        "strengths": "1024px photorealism",
        "size_gb": 6.9,
        "defaults": {"width": 1024, "height": 1024, "steps": 30, "cfg_scale": 7.5},
    },
    "SDXL-Turbo": {
        "display": "SDXL Turbo",
        "speed": "~15-90s",
        "strengths": "Fast SDXL",
        "size_gb": 6.9,
        "defaults": {"width": 512, "height": 512, "steps": 4, "cfg_scale": 1.0},
    },
    "Z-Image-Turbo": {
        "display": "Z-Image Turbo",
        "speed": "~30-180s",
        "strengths": "Bilingual EN/CN turbo model",
        "size_gb": 20.7,
        "defaults": {"width": 512, "height": 512, "steps": 8, "cfg_scale": 2.5},
    },
}

_ASPECT_SIZES = {
    "landscape": lambda w, h: (max(w, h), min(w, h)) if w != h else (int(w * 1.33), h),
    "square": lambda w, h: (w, h),
    "portrait": lambda w, h: (min(w, h), max(w, h)) if w != h else (w, int(h * 1.33)),
}

_models_cache: Dict[str, Any] = {"at": 0.0, "models": {}}


def _load_lemonade_config() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        if not isinstance(section, dict):
            return {}
        nested = section.get("lemonade")
        return nested if isinstance(nested, dict) else {}
    except Exception as exc:
        logger.debug("Could not load image_gen.lemonade config: %s", exc)
        return {}


def _lemonade_base_url() -> str:
    env = (os.environ.get("LEMONADE_BASE_URL") or "").strip()
    if env:
        return env.rstrip("/")
    cfg = _load_lemonade_config()
    url = str(cfg.get("base_url") or "").strip()
    if url:
        return url.rstrip("/")
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        providers = cfg.get("custom_providers") if isinstance(cfg, dict) else None
        if isinstance(providers, list):
            for entry in providers:
                if not isinstance(entry, dict):
                    continue
                if str(entry.get("name") or "").strip().lower() == "lemonade":
                    base = str(entry.get("base_url") or "").strip()
                    if base:
                        return base.rstrip("/")
    except Exception:
        pass
    return DEFAULT_BASE_URL


def _lemonade_api_key() -> str:
    env = (os.environ.get("LEMONADE_API_KEY") or "").strip()
    if env:
        return env
    cfg = _load_lemonade_config()
    return str(cfg.get("api_key") or "lemonade")


def _request_timeout() -> float:
    cfg = _load_lemonade_config()
    try:
        return float(cfg.get("timeout", DEFAULT_TIMEOUT))
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT


def _pull_timeout() -> float:
    cfg = _load_lemonade_config()
    try:
        return float(cfg.get("pull_timeout", 7200.0))
    except (TypeError, ValueError):
        return 7200.0


def _load_timeout() -> float:
    cfg = _load_lemonade_config()
    try:
        return float(cfg.get("load_timeout", 600.0))
    except (TypeError, ValueError):
        return 600.0


def _auto_pull_enabled() -> bool:
    cfg = _load_lemonade_config()
    return bool(cfg.get("auto_pull", True))


def _model_meta(model_id: str) -> Dict[str, Any]:
    live = _models_cache.get("models") or {}
    if model_id in live:
        return live[model_id]
    return _STATIC_MODELS.get(model_id, {})


def _catalog_entry(model_id: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    size_gb = meta.get("size_gb")
    price = f"local · {size_gb:.1f} GB" if isinstance(size_gb, (int, float)) else "local"
    return {
        "id": model_id,
        "display": meta.get("display", model_id),
        "speed": meta.get("speed", "local"),
        "strengths": meta.get("strengths", "Lemonade stable-diffusion.cpp"),
        "price": price,
    }


def _parse_remote_models(payload: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(payload, dict):
        return {}
    data = payload.get("data")
    if not isinstance(data, list):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("id") or "").strip()
        if not model_id:
            continue
        labels = item.get("labels") if isinstance(item.get("labels"), list) else []
        recipe = str(item.get("recipe") or "").strip().lower()
        is_image = "image" in labels or recipe == "sd-cpp"
        if not is_image:
            continue
        defaults = item.get("image_defaults") if isinstance(item.get("image_defaults"), dict) else {}
        meta = dict(_STATIC_MODELS.get(model_id, {}))
        if defaults:
            meta["defaults"] = {
                "width": int(defaults.get("width", meta.get("defaults", {}).get("width", 512))),
                "height": int(defaults.get("height", meta.get("defaults", {}).get("height", 512))),
                "steps": int(defaults.get("steps", meta.get("defaults", {}).get("steps", 20))),
                "cfg_scale": float(defaults.get("cfg_scale", meta.get("defaults", {}).get("cfg_scale", 7.5))),
            }
        size = item.get("size")
        if isinstance(size, (int, float)):
            meta["size_gb"] = float(size)
        meta.setdefault("display", model_id)
        out[model_id] = meta
    return out


def _refresh_models_cache(*, force: bool = False) -> Dict[str, Dict[str, Any]]:
    now = time.time()
    if not force and now - float(_models_cache.get("at") or 0.0) < 60.0:
        cached = _models_cache.get("models")
        if isinstance(cached, dict) and cached:
            return cached

    merged = {k: dict(v) for k, v in _STATIC_MODELS.items()}
    base = _lemonade_base_url()
    try:
        resp = requests.get(
            f"{base}/models",
            params={"show_all": "true"},
            headers={"Authorization": f"Bearer {_lemonade_api_key()}"},
            timeout=15,
        )
        if resp.ok:
            remote = _parse_remote_models(resp.json())
            merged.update(remote)
    except Exception as exc:
        logger.debug("Lemonade model list unavailable: %s", exc)

    _models_cache["at"] = now
    _models_cache["models"] = merged
    return merged


def _resolve_model(requested: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    env_override = (os.environ.get("LEMONADE_IMAGE_MODEL") or "").strip()
    candidates: List[str] = []
    if requested:
        candidates.append(requested.strip())
    if env_override:
        candidates.append(env_override)
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        if isinstance(section, dict):
            for key in ("model",):
                val = section.get(key)
                if isinstance(val, str) and val.strip():
                    candidates.append(val.strip())
            nested = section.get("lemonade")
            if isinstance(nested, dict):
                val = nested.get("model")
                if isinstance(val, str) and val.strip():
                    candidates.append(val.strip())
    except Exception:
        pass

    catalog = _refresh_models_cache()
    for candidate in candidates:
        if candidate in catalog:
            return candidate, catalog[candidate]
    if DEFAULT_MODEL in catalog:
        return DEFAULT_MODEL, catalog[DEFAULT_MODEL]
    if catalog:
        first = next(iter(catalog))
        return first, catalog[first]
    return DEFAULT_MODEL, _STATIC_MODELS[DEFAULT_MODEL]


def _size_for_aspect(model_id: str, aspect_ratio: str) -> str:
    meta = _model_meta(model_id)
    defaults = meta.get("defaults") if isinstance(meta.get("defaults"), dict) else {}
    base_w = int(defaults.get("width", 512))
    base_h = int(defaults.get("height", 512))
    fn = _ASPECT_SIZES.get(aspect_ratio, _ASPECT_SIZES["square"])
    width, height = fn(base_w, base_h)
    # Lemonade expects multiples of 8 for many SD backends.
    width = max(256, (width // 8) * 8)
    height = max(256, (height // 8) * 8)
    return f"{width}x{height}"


class LemonadeImageGenProvider(ImageGenProvider):
    @property
    def name(self) -> str:
        return "lemonade"

    @property
    def display_name(self) -> str:
        return "Lemonade (local)"

    def is_available(self) -> bool:
        base = _lemonade_base_url()
        runtime = _load_common("runtime")
        if runtime.lemonade_is_running(base):
            return True
        return bool(_ensure_server().get("ok"))

    def list_models(self) -> List[Dict[str, Any]]:
        catalog = _refresh_models_cache()
        return [_catalog_entry(mid, meta) for mid, meta in sorted(catalog.items())]

    def default_model(self) -> Optional[str]:
        model_id, _ = _resolve_model()
        return model_id

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Lemonade (local)",
            "badge": "free · local",
            "tag": "Stable Diffusion / Flux / Qwen / Z-Image via Lemonade Server",
            "env_vars": [
                {
                    "key": "LEMONADE_BASE_URL",
                    "prompt": "Lemonade API base URL",
                    "url": "https://lemonade-server.ai/docs/api/openai/",
                },
            ],
        }

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        aspect = resolve_aspect_ratio(aspect_ratio)
        model_id, meta = _resolve_model(
            str(kwargs.get("model")).strip() if kwargs.get("model") else None
        )
        defaults = meta.get("defaults") if isinstance(meta.get("defaults"), dict) else {}
        steps = kwargs.get("steps", defaults.get("steps"))
        cfg_scale = kwargs.get("cfg_scale", defaults.get("cfg_scale"))
        seed = kwargs.get("seed")

        if not prompt or not str(prompt).strip():
            return error_response(
                error="Prompt is required",
                error_type="validation_error",
                provider=self.name,
                model=model_id,
                prompt=prompt or "",
                aspect_ratio=aspect,
            )

        start = _ensure_server()
        if not start.get("ok"):
            return error_response(
                error=f"Lemonade server not reachable at {_lemonade_base_url()}: {start}",
                error_type="backend_unavailable",
                provider=self.name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        base = _lemonade_base_url()
        model_prep: Dict[str, Any] = {"skipped": True}
        if _auto_pull_enabled():
            lm = _load_common("lemonade_models")
            on_progress = _lemonade_progress_callback("image_generate", model_id)
            model_prep = lm.ensure_model_ready(
                base,
                model_id,
                api_key=_lemonade_api_key(),
                pull_timeout=_pull_timeout(),
                load_timeout=_load_timeout(),
                on_progress=on_progress,
            )
            if not model_prep.get("success"):
                return error_response(
                    error=(
                        f"{model_prep.get('error') or f'Failed to prepare model {model_id!r}'}"
                        f" (prep={json.dumps(model_prep)[:400]})"
                    ),
                    error_type="model_not_ready",
                    provider=self.name,
                    model=model_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )

        payload: Dict[str, Any] = {
            "model": model_id,
            "prompt": str(prompt).strip(),
            "size": _size_for_aspect(model_id, aspect),
            "n": 1,
            "response_format": "b64_json",
        }
        if steps is not None:
            payload["steps"] = int(steps)
        if cfg_scale is not None:
            payload["cfg_scale"] = float(cfg_scale)
        if seed is not None:
            payload["seed"] = int(seed)

        timeout = _request_timeout()
        resp = self._post_generation(base, payload, timeout=timeout)
        if resp is None:
            return error_response(
                error="Lemonade request failed (no response)",
                error_type="network_error",
                provider=self.name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        if resp.status_code >= 400:
            detail = resp.text[:500]
            try:
                detail = json.dumps(resp.json())[:500]
            except Exception:
                pass
            if (
                _auto_pull_enabled()
                and resp.status_code >= 500
                and "model_load" in detail.lower()
            ):
                lm = _load_common("lemonade_models")
                retry_prep = lm.ensure_model_ready(
                    base,
                    model_id,
                    api_key=_lemonade_api_key(),
                    pull_timeout=_pull_timeout(),
                    load_timeout=_load_timeout(),
                    on_progress=_lemonade_progress_callback("image_generate", model_id),
                )
                model_prep["retry_prep"] = retry_prep
                if retry_prep.get("success"):
                    resp = self._post_generation(base, payload, timeout=timeout)
                    if resp is not None and resp.status_code < 400:
                        return self._finish_generation(
                            resp,
                            model_id=model_id,
                            prompt=prompt,
                            aspect=aspect,
                            payload=payload,
                            start=start,
                            model_prep=model_prep,
                        )
            return error_response(
                error=f"Lemonade HTTP {resp.status_code}: {detail}",
                error_type="api_error",
                provider=self.name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        return self._finish_generation(
            resp,
            model_id=model_id,
            prompt=prompt,
            aspect=aspect,
            payload=payload,
            start=start,
            model_prep=model_prep,
        )

    def _post_generation(
        self,
        base: str,
        payload: Dict[str, Any],
        *,
        timeout: float,
    ) -> Optional[requests.Response]:
        try:
            return requests.post(
                f"{base}/images/generations",
                headers={
                    "Authorization": f"Bearer {_lemonade_api_key()}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=timeout,
            )
        except requests.RequestException as exc:
            logger.warning("Lemonade generation request failed: %s", exc)
            return None

    def _finish_generation(
        self,
        resp: requests.Response,
        *,
        model_id: str,
        prompt: str,
        aspect: str,
        payload: Dict[str, Any],
        start: Dict[str, Any],
        model_prep: Dict[str, Any],
    ) -> Dict[str, Any]:
        try:
            body = resp.json()
        except ValueError as exc:
            return error_response(
                error=f"Lemonade returned invalid JSON: {exc}",
                error_type="invalid_response",
                provider=self.name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, list) or not data:
            return error_response(
                error="Lemonade response contained no images",
                error_type="empty_response",
                provider=self.name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        first = data[0] if isinstance(data[0], dict) else {}
        b64_data = first.get("b64_json")
        if not isinstance(b64_data, str) or not b64_data.strip():
            return error_response(
                error="Lemonade response missing b64_json image data",
                error_type="empty_response",
                provider=self.name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        try:
            saved = save_b64_image(b64_data, prefix=f"lemonade_{model_id.replace('/', '_')}")
        except Exception as exc:
            return error_response(
                error=f"Failed to save Lemonade image: {exc}",
                error_type="io_error",
                provider=self.name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        return success_response(
            image=str(saved),
            model=model_id,
            prompt=prompt,
            aspect_ratio=aspect,
            provider=self.name,
            extra={
                "size": payload["size"],
                "steps": payload.get("steps"),
                "cfg_scale": payload.get("cfg_scale"),
                "backend": "lemonade-sd-cpp",
                "autostart": start,
                "model_prep": model_prep,
            },
        )


def _lemonade_progress_callback(tool_name: str, model_id: str):
    tp = _load_common("tui_progress")
    return tp.lemonade_pull_progress_callback(tool_name, model_id)


def register(ctx) -> None:
    _load_common("display_patch").ensure_display_patch()
    ctx.register_image_gen_provider(LemonadeImageGenProvider())
