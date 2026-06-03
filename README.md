# hermes-lemonade-llm-image-support

**Fast local image generation for Hermes** via [Lemonade Server](https://github.com/lemonade-sdk/lemonade) — SD, SDXL, Flux, Qwen, and more, exposed to the agent as `image_generate`.

**Author:** [SkyLaTry](https://github.com/SkyLaTry) · **Hermes:** 0.15.1+ · **Plugin ID:** `image_gen/lemonade-llm-image-support` · **Version:** 1.1.1 · **Repo:** [SkyLaTry/hermes-lemonade-llm-image-support](https://github.com/SkyLaTry/hermes-lemonade-llm-image-support)

Part of the [SkyLaTry Hermes plugin set](https://github.com/SkyLaTry/hermes-essentials/blob/main/PLUGINS.md).

For ComfyUI workflows, node management, and richer agent tooling, pair with **[hermes-image-local-tools](https://github.com/SkyLaTry/hermes-image-local-tools)**.

---

## What it does

This plugin registers a **Lemonade-backed** `image_gen` provider for Hermes. It allows both image generation and llm useage with the lemonade local api-endpint with auto detection and isntallation fo models. The agent can generate images locally through stable-diffusion.cpp models served by Lemonade — no cloud API keys, no external upload of prompts.

Install lands under `image_gen/lemonade-llm-image-support/` so Hermes discovers the backend correctly. Bundled `common/` helpers ship with the release repo.

---

## Use cases

| Scenario | Why Lemonade |
|----------|----------------|
| **Quick illustrations in chat** | “Draw a icon for this app idea” — agent calls `image_generate` and returns a file path. |
| **AMD / hybrid GPU setups** | Lemonade targets efficient local inference; good fit when ComfyUI overhead is more than you need. |
| **Privacy-sensitive prompts** | Images never leave your machine; no third-party image API. |
| **Iterative design loops** | Regenerate variants from the same TUI or Telegram session without switching tools. |
| **Lightweight stack** | One server process, simple HTTP API — less setup than a full ComfyUI graph for basic txt2img. |
| **Flux / SDXL / SD 1.5** | Swap models in Lemonade; Hermes uses whichever backend you configure as `image_gen.provider: lemonade`. |

**Pair with [hermes-image-local-tools](https://github.com/SkyLaTry/hermes-image-local-tools)** when you also want ComfyUI graphs, workflow files, or `comfyui_manage` / `lemonade_manage` agent tools.

---

## Quick start

```bash
# Hermes (once)
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash

# This plugin → ~/.hermes/plugins/image_gen/lemonade-llm-image-support/
curl -fsSL https://raw.githubusercontent.com/SkyLaTry/hermes-lemonade-llm-image-support/main/install.sh | bash
hermes gateway restart
```

> **Important:** Image-gen plugins must live under `image_gen/`. Do **not** use plain `hermes plugins install` for this repo — it clones to the wrong path.

**Manual install:**

```bash
mkdir -p ~/.hermes/plugins/image_gen
git clone https://github.com/SkyLaTry/hermes-lemonade-llm-image-support.git \
  ~/.hermes/plugins/image_gen/lemonade-llm-image-support
hermes plugins enable image_gen/lemonade-llm-image-support
hermes gateway restart
```

See [INSTALL.md](INSTALL.md) for a short copy-paste checklist.

---

## Enable in config

```yaml
plugins:
  enabled:
    - image_gen/lemonade-llm-image-support
    - image_gen/local-tools   # optional: agent tools + ComfyUI backend

image_gen:
  provider: lemonade
  lemonade:
    base_url: http://127.0.0.1:13305/api/v1
```

Environment: `LEMONADE_BASE_URL`, etc.

---

## Example agent flows

1. **Chat:** “Generate a 1024×1024 wallpaper, dark cyberpunk city” → agent uses `image_generate` → image saved locally, path returned in chat.
2. **Telegram + gateway:** Same flow from a mobile channel if `image_gen` is enabled on the gateway host.
3. **Switch backends:** Use Lemonade for speed; enable [local-tools](https://github.com/SkyLaTry/hermes-image-local-tools) and set `provider: comfyui` when you need custom workflows.

---

## Migration from `image_gen/lemonade`

Rename the install folder and update `~/.hermes/config.yaml`:

```yaml
plugins:
  enabled:
    - image_gen/lemonade-llm-image-support   # was image_gen/lemonade
```

---

## Related SkyLaTry plugins

See [PLUGINS.md](PLUGINS.md) for the full index.

| Plugin | Repository |
|--------|------------|
| Hermes Essentials | [hermes-essentials](https://github.com/SkyLaTry/hermes-essentials) |
| Screen Awareness | [hermes-screen-awareness](https://github.com/SkyLaTry/hermes-screen-awareness) |
| Sys Controll | [hermes-sys-controll](https://github.com/SkyLaTry/hermes-sys-controll) |
| **Lemonade LLM Image** *(this repo)* | [hermes-lemonade-llm-image-support](https://github.com/SkyLaTry/hermes-lemonade-llm-image-support) |
| Image Local Tools | [hermes-image-local-tools](https://github.com/SkyLaTry/hermes-image-local-tools) |

---

## License

SkyLaTry Shared Source License — see [LICENSE](LICENSE) and [LICENSING.md](LICENSING.md).
