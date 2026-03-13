# macOS

## Install

```bash
python3 -m pip install .
```

## Configure

```bash
claude-openai-bridge configure --gui
```

## Install background service

```bash
claude-openai-bridge install-service
```

This writes a LaunchAgent under:

`~/Library/LaunchAgents/`

## Check service

```bash
launchctl list | grep claude-openai-bridge
```

## Remove service

```bash
claude-openai-bridge uninstall-service
```
