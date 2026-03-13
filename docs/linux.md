# Linux

## Install

```bash
python -m pip install .
```

## Configure

```bash
claude-openai-bridge configure --gui
```

or:

```bash
claude-openai-bridge configure
```

## Install background service

```bash
claude-openai-bridge install-service
```

This writes:

`~/.config/systemd/user/claude-openai-bridge.service`

## Check service

```bash
systemctl --user status claude-openai-bridge.service
journalctl --user -u claude-openai-bridge.service -f
```

## Remove service

```bash
claude-openai-bridge uninstall-service
```
