# Full workspace snapshot

The private `World-controller/CEBRA` repository contains the source-code snapshot on `main`. The rest of the project working tree, including paper and patent materials, research data, experiment outputs, model files, and local project notes, is attached to the private GitHub Release `full-workspace-2026-11-03` as a split Zstandard-compressed TAR archive.

## Restore the complete working-tree snapshot

Download every `CEBRA-full-workspace-2026-11-03.tar.zst.part-*` asset and `SHA256SUMS.txt` from the release. In an empty directory on macOS or Linux:

```bash
shasum -a 256 -c SHA256SUMS.txt
cat CEBRA-full-workspace-2026-11-03.tar.zst.part-* > CEBRA-full-workspace-2026-11-03.tar.zst
zstd -t CEBRA-full-workspace-2026-11-03.tar.zst
zstd -d -c CEBRA-full-workspace-2026-11-03.tar.zst | tar -xf -
```

The release archive has paths relative to the project root, so extraction into an empty directory reconstructs the workspace tree. The checksum file lists each downloaded volume; concatenating them in lexical order produces the archive.

## Snapshot scope

The snapshot includes the current working-tree files and untracked project materials. This includes `src/`, `data/`, `dq-paper/`, `paper/`, `patent/`, `refine-logs/`, `model_cache/`, `figures/`, `results/`, and project notes/configuration. It preserves symbolic links as links and includes model/dataset license files present in the workspace.

Excluded: Git internals/history (`.git`), virtual environments (`.venv*`, `venv`, `env`), Python/build caches, `node_modules`, browser-session state, `.DS_Store`, local app settings, and credential files (`.env*`, private-key files). The archive does not replace independent backups of the original workspace.

Archive and volume checksums are listed in the release assets. Snapshot layout/count/size were inventoried before packaging; the release asset hashes provide end-to-end transfer verification.
