# Full workspace snapshot

The private `World-controller/CEBRA` repository contains the source-code snapshot on `main`. A broader snapshot of the current project workspace is attached to the private GitHub Release `full-workspace-2026-11-03` as split Zstandard-compressed TAR volumes. It includes the project code, paper/patent materials, datasets and experiment outputs, model-cache files, figures, and local research notes.

## Restore

Download every `CEBRA-full-workspace-20261103.tar.zst.part-*` asset and `SHA256SUMS.txt` from the release. In an empty directory on macOS or Linux:

```bash
shasum -a 256 -c SHA256SUMS.txt
cat CEBRA-full-workspace-20261103.tar.zst.part-* > CEBRA-full-workspace-20261103.tar.zst
zstd -t CEBRA-full-workspace-20261103.tar.zst
zstd -d -c CEBRA-full-workspace-20261103.tar.zst | tar -xf -
```

The archive uses paths relative to the project root. Concatenate the numbered volumes in lexical order to reconstruct it. Check each part against `SHA256SUMS.txt` before concatenation.

## Exclusions

Git internals/history (`.git`), virtual environments, Python/build/package caches, `node_modules`, local assistant/app configuration and sessions, external symlinks, `.DS_Store`, environment/credential files, and private-key files are not included. The snapshot contains model files with the corresponding licenses and policy files present in the workspace. It is a point-in-time backup and does not replace an independent backup of the original workspace.
