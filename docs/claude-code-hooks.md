# Optional: format-on-write hook

Once `uv run ruff format` is confirmed working, this hook keeps every Python
file Claude Code writes or edits formatted, so formatting never shows up in a
diff as noise.

Add to `.claude/settings.json` (merge into the existing object):

```json
"hooks": {
  "PostToolUse": [
    {
      "matcher": "Write|Edit",
      "hooks": [
        {
          "type": "command",
          "command": "uv run ruff format \"$CLAUDE_FILE_PATHS\" 2>nul"
        }
      ]
    }
  ]
}
```

## Windows notes

- Without Git Bash installed, hook commands run through PowerShell or CMD, so
  `2>/dev/null` will not work; use `2>nul`.
- If Git Bash is installed but not found automatically, point at it in
  `~/.claude/settings.json`:

  ```json
  { "env": { "CLAUDE_CODE_GIT_BASH_PATH": "C:\\Program Files\\Git\\bin\\bash.exe" } }
  ```

- Test the command manually in the same shell before adding it as a hook. A
  hook that fails silently on every write is worse than no hook.
- Add the hook **after** the first notebooks are working. Debugging a hook and
  debugging a pipeline at the same time is a bad trade.
