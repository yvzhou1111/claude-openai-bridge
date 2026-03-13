# Windows

## Install

```powershell
py -m pip install .
```

## Configure

```powershell
claude-openai-bridge configure --gui
```

If the console script is not on `PATH`, use:

```powershell
py -m claude_openai_bridge.cli configure --gui
```

## Install startup item

```powershell
claude-openai-bridge install-service
```

This writes a startup `.cmd` file into your Startup folder so the bridge launches after login.

## Check status

```powershell
claude-openai-bridge status
```

## Remove startup item

```powershell
claude-openai-bridge uninstall-service
```
