# 📦 Backup Plex - Release Notes (v6.0.0)

## ✨ New Features
- Interactive `--restore` and `--test-restore` modes
- Dynamic archive support: `.zst`, `.7z`, `.tar`
- Compression fallback logic based on system compatibility
- Progress bars with ETA for backup/restore
- Plex shutdown/start with user prompt with restoration
- Clean summaries and logging with runtime breakdown
- Auto-pruning of old backups based on retention limits
- Backup summary includes "Next Scheduled Full Backup"

## 🛠 Improvements
- Modular backup functions (`zst_backup`, `7z_backup`, `tar_backup`)
- All tmp/logs directories are automatically created
- Log rotation for the last 10 runs
- More robust error handling and output suppression.
- Config validation for webhook/channel/paths before execution

## 🔧 Developer Notes
- Requires `pv` for best progress display
- Requires `zstd` or `7z` depending on compression method
- Designed to be used in Unraid and general Linux environments