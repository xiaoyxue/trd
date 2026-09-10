# CLAUDE.md

The contributor guide for this repository is [`AGENTS.md`](AGENTS.md). There is
only one tracked copy of it (repo root); it is imported here rather than
duplicated, so Claude Code loads the same file every other agent reads.

@AGENTS.md

## Division of labour: Claude Code codes, Copilot runs the e2e

Claude Code's scope is the **code and the non-interactive gates** — design,
implementation, review, and the automated gates a test level asks for: **L1**
(`nix flake check`) and **L2** (goldens, `gpu_tests`, `gui_render`). It runs
those itself and reports them in the [verification
matrix](AGENTS.md#verification-matrix).

**It does not drive the §3/§4 end-to-end cases** — the native and browser
windows, the video editors, the large-file seek. Those need a display, a real
event loop and a human-paced UI, and a coding session that also drives them
spends most of its context on screenshots and click coordinates rather than on
the diff.

So a PR whose level reaches **L3** is handed off, not stalled. Claude Code:

1. runs and reports L1 + L2 on its own platform;
2. leaves every §3/§4 row **🤝**, never ✅ and never a quiet `n/a` — `n/a` stays
   reserved for gates the *level* excludes;
3. writes the handoff as a **ready-to-paste prompt** for GitHub Copilot: the
   exact commands with real paths, the acceptance cases, and what a pass looks
   like, so the run needs no re-derivation;
4. posts it in **both** the PR and the issue, per
   [Handoff](AGENTS.md#pr-workflow).

The declared level still says **L3**. A handed-off gate is an outstanding gate:
the PR is not done until Copilot posts the completed matrix with those rows
flipped to ✅.
