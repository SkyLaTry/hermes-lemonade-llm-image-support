# Install

Requires [Hermes Agent](https://github.com/NousResearch/hermes-agent) 0.15.1+.

**One command:**

```bash
curl -fsSL https://raw.githubusercontent.com/SkyLaTry/hermes-lemonade-llm-image-support/main/install.sh | bash
hermes gateway restart
```

**Manual:**

```bash
mkdir -p ~/.hermes/plugins/image_gen
git clone https://github.com/SkyLaTry/hermes-lemonade-llm-image-support.git \
  ~/.hermes/plugins/image_gen/lemonade-llm-image-support
hermes plugins enable image_gen/lemonade-llm-image-support
hermes gateway restart
```

This repo bundles `common/` at release time. For monorepo dev, keep `plugins/image_gen/common/` as a sibling.

```yaml
plugins:
  enabled:
    - image_gen/lemonade-llm-image-support
    - image_gen/local-tools   # agent tools + optional ComfyUI backend

image_gen:
  provider: lemonade
```

See [README.md](README.md) for migration from `image_gen/lemonade`.
