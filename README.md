# Extra Scripts

A collection of customizable Bash scripts designed to simplify and automate backup tasks for Plex Media Server, general appdata, and specific folders. These scripts are especially useful for Unraid or other home server environments.

## 📦 Included Scripts
[Release Notes](release_notes.md)

- `backup_plex.sh`  
  Backs up Plex Media Server data using a provided configuration file.


- `backup_appdata.sh`  
  Backs up specified application data folders with flexible config support.

- `backup_folder.sh`  
  Generic backup script for any folder you want to protect.

## 🧩 Config Files

- `backup-plex-example.conf`  
  Sample config file for `backup_plex.sh`.

- `backup_appdata.sample.conf`  
  Sample config file for `backup_appdata.sh`.

- `exclude-file.txt`  
  Example list of paths to exclude during backup (used by `rsync`).

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Drazzilb08/extra-scripts.git
cd extra-scripts
```

### 2. Make the scripts executable

```bash
chmod +x backup_plex.sh backup_appdata.sh backup_folder.sh
```

### 3. Configure the scripts
- Edit the sample config files (`backup-plex-example.conf`, `backup_appdata.sample.conf`) to suit your needs.
- Rename them to `backup-plex.conf` and `backup_appdata.conf` respectively.
- Update the paths and options in the config files as necessary.
- For `backup_folder.sh`, you can directly edit the script to set the source and destination folders.
- For `backup_appdata.sh`, you can directly edit the script to set the source and destination folders.
- For `backup_plex.sh`, you can directly edit the script to set the source and destination folders.

### 4. Run the scripts
- To back up Plex Media Server data:
```bash
./backup_plex.sh
```
- To back up application data:
```bash
./backup_appdata.sh
```
- To back up a specific folder:
```bash
./backup_folder.sh
```

⚙️ Features
•	Easy-to-edit config files
•	Exclusion support using exclude-file.txt
•	Clean, modular, and cron-job friendly
•	Works great with Unraid, Linux servers, or NAS setups

📝 License

This project is licensed under the [MIT License](https://github.com/Drazzilb08/extra-scripts/blob/main/LICENSE).

👤 Author

Created and maintained by [Drazzilb08](https://github.com/Drazzilb08)