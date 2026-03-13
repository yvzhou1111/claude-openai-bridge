# Claude OpenAI Bridge

Use OpenAI-compatible endpoints inside Claude Code through a lightweight local bridge.

This project does three things:

- Runs a local Anthropic-compatible proxy for Claude Code.
- Converts Claude-style requests into OpenAI-compatible `responses`, `chat_completions`, or a mixed `hybrid` mode.
- Provides a small config workflow that can update Claude Code settings, detect the best upstream API mode, verify tool use, and test whether image understanding is actually reliable.

## Why This Exists

Claude Code expects an Anthropic-style API. Many third-party providers expose OpenAI-compatible APIs instead. This project lets you:

- keep using Claude Code locally
- point it at OpenAI-compatible providers
- auto-pick a working model from `/models`
- detect whether the upstream really supports images instead of only claiming it does

## Features

- Anthropic-compatible local proxy for Claude Code
- Upstream modes:
  - `responses`
  - `chat_completions`
  - `hybrid`
- Tool-call mapping from Claude content blocks to OpenAI-compatible tool schemas
- Image block forwarding for upstreams that genuinely support vision
- Hard fail for image uploads when the current upstream is not reliably vision-capable
- Cross-platform config path handling for Windows, Linux, and macOS
- Small interactive GUI with `--gui`
- Background startup support:
  - Linux user `systemd`
  - macOS `LaunchAgent`
  - Windows Startup folder entry

## Install

### Option 1: pipx

```bash
pipx install .
```

### Option 2: pip

```bash
python -m pip install .
```

### Option 3: run from source

```bash
python -m pip install -e .
```

## Quick Start

1. Configure the bridge.

CLI:

```bash
claude-openai-bridge configure
```

GUI:

```bash
claude-openai-bridge configure --gui
```

2. Install the background service / startup item.

```bash
claude-openai-bridge install-service
```

3. Check status.

```bash
claude-openai-bridge status
```

4. Open Claude Code again.

The tool updates these files automatically:

- `~/.claude/settings.json`
- `~/.claude-code/settings.json`

Claude Code will point to the local bridge after configuration.

## Commands

### Configure

```bash
claude-openai-bridge configure \
  --base-url https://example.com/v1 \
  --api-key sk-... \
  --api-mode auto
```

Behavior:

- loads `/models` if available
- lets you pick a model
- tests text and tool-use
- tests image understanding with a generated validation image
- writes the final config
- restarts the installed background service if one already exists

### Run

```bash
claude-openai-bridge run
```

This starts the local bridge in the foreground.

### Verify

```bash
claude-openai-bridge verify
```

This validates the currently saved upstream.

### Status

```bash
claude-openai-bridge status
```

This prints:

- saved config path
- active upstream settings
- background service state
- local bridge health if running

## API Modes

### `responses`

Use this when the upstream supports the OpenAI Responses API well.

### `chat_completions`

Use this when the upstream is more stable with chat completions.

### `hybrid`

Use this when text / tool traffic should go through `responses`, but image requests are more reliable through `chat_completions`.

The `configure` command can pick this automatically.

## Image Support

This project does **not** assume that an upstream really supports vision just because `/models` says `"image"` is available.

The configure flow performs an image validation round. If that fails:

- the upstream can still be saved for text / tool use
- `upstream_image_support` is set to `false`
- future image uploads will be rejected with a clear error instead of returning a fake description

That is intentional.

## Config File

The bridge stores its own config outside the repo.

### Linux

`~/.config/claude-openai-bridge/config.json`

### macOS

`~/Library/Application Support/ClaudeOpenAIBridge/config.json`

### Windows

`%APPDATA%\\ClaudeOpenAIBridge\\config.json`

## Platform Guides

- [Linux guide](docs/linux.md)
- [macOS guide](docs/macos.md)
- [Windows guide](docs/windows.md)

## Troubleshooting

### Claude Code still uses the old upstream

- reopen Claude Code
- check `claude-openai-bridge status`
- confirm `~/.claude/settings.json` and `~/.claude-code/settings.json` point at the local bridge

### Text works but images fail

That means the upstream failed the real vision validation. The bridge will block image uploads on purpose.

### The upstream says it supports images, but results are wrong

That is exactly why this project validates image understanding with a generated test image instead of trusting the upstream metadata.

### Linux background service

```bash
systemctl --user status claude-openai-bridge.service
journalctl --user -u claude-openai-bridge.service -f
```

### macOS background service

```bash
launchctl list | grep claude-openai-bridge
```

### Windows startup item

Check the Startup folder entry created by `install-service`.

## Security Notes

- API keys are stored locally in the bridge config file.
- Do not commit your generated config.
- Treat third-party upstreams as untrusted until tested.

## License

MIT
