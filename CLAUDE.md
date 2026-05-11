# CLAUDE.md

Project-specific supplement for `/workspace/amazon-two-tower`.

This file is the top-level project requirements file for Claude Code, equivalent to AGENTS.md for Codex.
It is a direct copy of AGENTS.md. The top-level `/workspace/AGENTS.md` remains the primary workspace rule file.
This file only records repository-specific operational notes.

## GitHub SSH / Push Notes

This repository uses SSH remote:

```text
origin git@github.com:EddyHai57/amazon-two-tower.git
```

Expected SSH key path on the server:

```text
~/.ssh/id_ed25519_amazon_two_tower
```

Do not use the default `~/.ssh/id_ed25519` path for this project unless it actually exists.

To test GitHub SSH access:

```bash
chmod 700 ~/.ssh
chmod 600 ~/.ssh/id_ed25519_amazon_two_tower
chmod 644 ~/.ssh/id_ed25519_amazon_two_tower.pub
ssh -i ~/.ssh/id_ed25519_amazon_two_tower -o IdentitiesOnly=yes -T git@github.com
```

Reliable push command:

```bash
GIT_SSH_COMMAND='ssh -i ~/.ssh/id_ed25519_amazon_two_tower -o IdentitiesOnly=yes' git push
```

Reason:

In Codex/server sessions, `ssh-agent` environment variables may not persist across separate shell calls. Therefore, plain `git push` may fail with `Permission denied (publickey)` even if `ssh -T git@github.com` succeeded in a previous command. Use `GIT_SSH_COMMAND` to explicitly specify the project key.

Never commit:

- `~/.ssh/*`
- private keys
- tokens
- credentials
- `outputs/`
- `data/processed/`
- checkpoints
- `logs/`
- `.venv_backup*/`
