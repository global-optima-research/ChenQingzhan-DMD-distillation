# Artifacts

This directory stores active curated evidence: selected videos, thumbnails, prompt files, and readmes copied from server-side experiments.

Keep the default workflow light:

- Leave raw checkpoints and full output trees on the server.
- Add local artifacts only when they are needed for comparison, reports, or handoff.
- Put the exact server source path in each artifact README.
- Prefer a small curated set over full experiment dumps.

Future bulk outputs should go under `artifacts/raw/` or `artifacts/tmp/`, which are ignored by git.

Historical inference videos from 2026-04 and 2026-05 were moved to `archive/artifacts/inference-videos-2026-spring/`.
