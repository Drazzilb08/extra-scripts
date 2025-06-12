#!/usr/bin/env bash
#  _____ _             ____             _                   _____           _       _
# |  __ \ |           |  _ \           | |                 / ____|         (_)     | |
# | |__) | | _____  __| |_) | __ _  ___| | ___   _ _ __   | (___   ___ _ __ _ _ __ | |_
# |  ___/| |/ _ \ \/ /|  _ < / _` |/ __| |/ / | | | '_ \   \___ \ / __| '__| | '_ \| __|
# | |    | |  __/>  < | |_) | (_| | (__|   <| |_| | |_) |  ____) | (__| |  | | |_) | |_
# |_|    |_|\___/_/\_\|____/ \__,_|\___|_|\_\\__,_| .__/  |_____/ \___|_|  |_| .__/ \__|
#                                                  | |                        | |
#                                                  |_|                        |_|

# ==========================================================
# 🎬 backup-plex - A script to backup your Plex database
# 🧠 Author: Drazzilb | 💻 Version: 6.0.0 | 📄 License: MIT
# ==========================================================
# This script performs scheduled backups of a Plex Media Server instance.
# Supports full or essential backups, compression, verification, restore, and webhook notifications.

script_version="6.0.0"

## Display help message and usage instructions
display_help() {
    cat <<EOF
📦 Plex Backup Script - Version $script_version

Usage:
  \$0 [options]

Options:
  -s, --source <dir>          Source Plex data directory
  -d, --destination <dir>     Destination for backups
  -k, --keep-essential <num>  Number of Essential backups to retain
  -K, --keep-full <num>       Number of Full backups to retain
  -F, --force-full <days>     Run full backup every X days
  -f, --full-backup           Enable full backup mode
  -c, --compress              Compress backups using 7z
  -r, --dry-run               Simulate run without changes
  -S, --shutdown              Shut down Plex during backup
  -q, --quiet                 Suppress output
  -u, --unraid-notify         Use Unraid's notify system
  -w, --webhook <url>         Discord or Notifiarr webhook
  -C, --channel <id>          Notifiarr channel ID
  -b, --bar-color <hex>       Hex color for notifications
  -n, --bot-name <name>       Bot name (for Discord)
  -D, --debug                 Enable debug logging
  -x, --use-config <true|false> Use config file or not
  --restore                   Restore the most recent backup
  --test-restore              Simulate restore to a temporary test directory
  -h, --help                  Show this help message

Example:
  bash \$0 -s /mnt/appdata/plex -d /mnt/backups/plex -f -c -w https://discord.com/api/webhooks/...
EOF
}

## Handle CLI arguments and map to script variables
handle_options() {
    bar_color='e5a00d'
    use_config_file="true"
    shutdown_plex="false"
    dry_run="false"

    TEMP=$(getopt -o s:d:k:c:w:C:K:F:f:r:S:D:x:q:u:b:n:h --long help,restore,test-restore -- "$@")
    eval set -- "$TEMP"

    while true; do
        case "$1" in
            -s) source_dir="$2"; shift 2 ;;
            -d) destination_dir="$2"; shift 2 ;;
            -k) keep_essential="$2"; shift 2 ;;
            -c) compress="$2"; shift 2 ;;
            -w) webhook="$2"; shift 2 ;;
            -C) channel="$2"; shift 2 ;;
            -K) keep_full="$2"; shift 2 ;;
            -F) force_full_backup="$2"; shift 2 ;;
            -f) full_backup="$2"; shift 2 ;;
            -r) dry_run="$2"; shift 2 ;;
            -S) shutdown_plex="$2"; shift 2 ;;
            -D) debug="$2"; shift 2 ;;
            -x) use_config_file="$2"; shift 2 ;;
            -q) quiet="$2"; shift 2 ;;
            -u) unraid_notify="$2"; shift 2 ;;
            -b) bar_color="$2"; shift 2 ;;
            -n) bot_name="$2"; shift 2 ;;
            --restore) restore_flag="true"; shift ;;
            --test-restore) test_restore_flag="true"; shift ;;
            -h|--help) display_help; shift ;;
            --) shift; break ;;
            *) break ;;
        esac
    done
}

## Set up configuration file paths and script directory
config_dir_setup() {
    config_dir=$(dirname "$(readlink -f "$0")")
    script_path=$(dirname "$0")
    config_file="$script_path/backup_plex.conf"
}

## Load configuration from file if it exists
load_config_file() {
    if [ -f "$config_file" ]; then
        source "$config_file"
        # If debug is set  print this statement
        if [[ "${debug,,}" == "true" ]]; then
            echo "⚙️  Config file exists and is accessible."
        fi
    fi
}

## Validate directory paths and webhook if defined
check_config() {
    if [ ! -d "$source_dir" ]; then
        echo "❌ ERROR: Your source directory ($source_dir) does not exist"
        exit 0
    fi
    if [ -z "$source_dir" ]; then
        echo "❌ ERROR: Your source directory is not set"
        exit 0
    fi
    if [ ! -d "$destination_dir" ]; then
        echo "❌ ERROR: Your destination directory ($destination_dir) does not exist"
        exit 0
    fi
    if [ -z "$destination_dir" ]; then
        echo "❌ ERROR: Your destination directory is not set"
        exit 0
    fi
    if [[ "${compress,,}" == "true" ]]; then
        if ! command -v zstd >/dev/null && ! command -v zstdcat >/dev/null; then
            echo "❌ Compression enabled but neither zstd nor 7z is installed."
            exit 1
        fi
    fi
    if [ -n "$webhook" ]; then
        if [[ ! $webhook =~ ^https://discord\.com/api/webhooks/ ]] && [[ ! $webhook =~ ^https://notifiarr\.com/api/v1/notification/passthrough ]]; then
            echo "❌ ERROR: Invalid webhook format"
            exit 0
        fi
        if [[ $webhook =~ notifiarr ]] && [ -z "$channel" ]; then
            echo "❌ ERROR: Notifiarr webhook requires a channel ID (-C)"
            exit 0
        fi
        if [[ $webhook =~ notifiarr ]]; then
            apikey="${webhook##*/}"
            [ "${debug,,}" == "true" ] && echo "📡 Validating Notifiarr webhook: $webhook"
            response_code=$(curl --write-out "%{response_code}" --silent --output /dev/null \
                -H "x-api-key: $apikey" "https://notifiarr.com/api/v1/user/validate")
        else
            [ "${debug,,}" == "true" ] && echo "📡 Validating Discord webhook: $webhook"
            response_code=$(curl --write-out "%{response_code}" --silent --output /dev/null "$webhook")
        fi

        [ "${debug,,}" == "true" ] && echo "🔄 Response: $response_code"
        if [ "$response_code" -eq 200 ]; then
            if [[ "${debug,,}" == "true" ]]; then
                echo "✅ Webhook is valid"
            fi
        else
            echo "⚠️ Webhook is not valid. Continuing without notifications."
        fi
    fi
}

## Convert a hex color code to decimal for Discord/Notifiarr embeds
hex_to_decimal() {
    if [[ $bar_color =~ ^\#[0-9A-Fa-f]{6}$ ]]; then
        hex_bar_color=${bar_color:1}
        decimal_bar_color=$((0x${bar_color:1}))
    elif [[ $bar_color =~ ^[0-9A-Fa-f]{6}$ ]]; then
        hex_bar_color=$bar_color
        decimal_bar_color=$((0x$bar_color))
    else
        echo "Bar color: $bar_color"
        echo -e "❌ Invalid color format. Use 6-digit hex (e.g. ff0000)"
        exit 0
    fi
}

## Print messages if not in quiet mode
verbose_output() {
    if [[ "$(echo "$quiet" | tr '[:upper:]' '[:lower:]')" == "false" ]]; then
        echo -e "$1"
    fi
}

## Clean up old backups beyond the retention limit
cleanup_function() {
    verbose_output "💣 Cleaning up old backups..."
    destination_dir=$(realpath -s "$destination_dir")
    if [ -d "$destination_dir/Essential" ]; then
        verbose_output "🧹 Looking for old Essential backups (keeping last $keep_essential)..."
        find "$destination_dir/Essential" -mindepth 1 -maxdepth 1 -type d | sort -r | tail -n +$(( keep_essential + 1 )) | while read -r dir; do
            verbose_output "🗑  Removing: $dir"
            rm -rf "$dir"
        done
    fi
    if [ -d "$destination_dir"/Full ]; then
        verbose_output "🧹 Looking for old Full backups (keeping last $keep_full)..."
        find "$destination_dir/Full" -mindepth 1 -maxdepth 1 -type d | sort -r | tail -n +$(( keep_full + 1 )) | while read -r dir; do
            verbose_output "🗑  Removing: $dir"
            rm -rf "$dir"
        done
        verbose_output "✅ Done\n"
    fi
}

## Calculate total runtime of the backup process
calculate_runtime() {
    total_time=$((end - start))
    seconds=$((total_time % 60))
    minutes=$((total_time % 3600 / 60))
    hours=$((total_time / 3600))

    if ((minutes == 0 && hours == 0)); then
        run_output="⏱️ Backup completed in $seconds seconds"
    elif ((hours == 0)); then
        run_output="⏱️ Backup completed in $minutes minutes and $seconds seconds"
    else
        run_output="⏱️ Backup completed in $hours hours $minutes minutes and $seconds seconds"
    fi
}

## Send a system notification to Unraid (if enabled)
unraid_notification() {
    case "$backup_type" in
        essential) /usr/local/emhttp/webGui/scripts/notify -e "Unraid Server Notice" -s "Plex Backup" -d "Essential Plex data has been backed up" -i "normal" ;;
        full) /usr/local/emhttp/webGui/scripts/notify -e "Unraid Server Notice" -s "Plex Backup" -d "Full Plex data has been backed up" -i "normal" ;;
        both) /usr/local/emhttp/webGui/scripts/notify -e "Unraid Server Notice" -s "Plex Backup" -d "Essential & Full Plex data has been backed up" -i "normal" ;;
    esac
}

## Send a webhook notification to Discord or Notifiarr
send_notification() {
    if [[ -n "$webhook" ]]; then
        [ "${full_backup,,}" == "true" ] && echo -e "\ncurl -X POST -d '$payload' $webhook"
        curl -s -H "Content-Type: application/json" -X POST -d "$payload" "$webhook" --output /dev/null
    fi
}

## Format field entries for the webhook JSON payload
field_builder() {
    local title_text="$1"
    local text_value="$2"
    local reset="$3"
    if [ "$reset" == "true" ]; then
        fields=""
    fi
    local block='{
        "'"$title"'": "'"$title_text"'",
        "'"$text"'": "'"$text_value"'",
        "inline": false
    }'
    [ -n "$fields" ] && block=",$block"
    fields="${fields}${block}"
}

## Assemble the webhook JSON payload
build_payload() {
    get_ts=$(date -u -Iseconds)
    joke=$(curl -s https://raw.githubusercontent.com/Drazzilb08/daps/master/jokes.txt | shuf -n 1 | sed 's/"/\\"/g')
    if [[ $webhook =~ discord ]]; then
        bot_name="Notification Bot"
        title="name"
        text="value"
        common_fields='{
            "username": "'"$bot_name"'",
            "embeds": [{
                "title": "Plex Backup",'
        common_fields2='"footer": { "text": "Powered by: Drazzilb | '"$joke"'" },
            "color": '"$decimal_bar_color"',
            "timestamp": "'"$get_ts"'"
        }]
    }'
    elif [[ $webhook =~ notifiarr ]]; then
        title="title"
        text="text"
        common_fields='{
            "notification": {
                "update": false,
                "name": "Plex Backup",
                "event": ""
            },
            "discord": {
                "color": "'"$hex_bar_color"'",
                "text": {
                    "title": "Plex Backup",'
        common_fields2='
            "footer": "Powered by: Drazzilb | '"$joke"'"
        },
        "ids": {
            "channel": "'"$channel"'"
        }
    }
}'
    fi
}

## Wrap final payload into complete notification structure
payload() {
    local description="$1"
    payload="${common_fields}
        \"description\": \"$description\",
        \"fields\": [
            $fields
        ],
        $common_fields2"
}

## Perform a Zstandard compressed backup
zst_backup() {
    extension="zst"
    verbose_output "📦 Creating Zstandard archive..."
    dir_size=$(cd "$source_dir" && du -sb --exclude="${exclude[*]#--exclude=}" "${backup_source[@]}" 2>/dev/null | awk '{sum+=$1} END{print sum}')
    
    if [[ -z "$dir_size" || "$dir_size" -eq 0 ]]; then
        echo "❌ ERROR: Nothing to archive. The source may be empty or incorrectly specified."
        backup_failed=true
        return
    fi

    ( cd "$source_dir" &&
        tar --ignore-failed-read -cf - --transform "s|^|$base_dir_name/|" "${exclude[@]}" "${backup_source[@]}"
    ) | \
    pv --size "$dir_size" --progress --timer --rate --eta --bytes --name BACKUP --force --width "$(($(tput cols)-10))" 2> >(test -t 2 && cat >&2 || cat >/dev/null) | \
    zstd -q --threads=0 -19 -o "$backup_path/$folder_type-plex_backup.zst"
}

## Perform a 7z compressed backup
7z_backup() {
    extension="7z"
    verbose_output "📦 Creating 7z archive..."
    dir_size=$(cd "$source_dir" && du -sb --exclude="${exclude[*]#--exclude=}" "${backup_source[@]}" | awk '{sum+=$1} END{print sum}')
    ( cd "$source_dir" &&
        tar --ignore-failed-read -cf - --transform "s|^|$base_dir_name/|" "${exclude[@]}" "${backup_source[@]}"
    ) | \
    pv --size "$dir_size" --progress --timer --rate --eta --bytes --name BACKUP --force --width "$(($(tput cols)-10))" 2> >(test -t 2 && cat >&2 || cat >/dev/null) | \
    7z a -si -t7z -m0=lzma2 -mx=3 -md=16m -mfb=32 -mmt=on -ms=off "$backup_path/$folder_type-plex_backup.7z" >/dev/null
}

## Perform a tar archive backup with optional progress
tar_backup() {
    extension="tar"
    if [ "$dry_run" == "true" ]; then
        extension+=".dry_run"
        echo "🧪 Dry run: Creating dummy file at: $backup_path/$folder_type-plex_backup.$extension"
        touch "$backup_path/$folder_type-plex_backup.$extension"
    else
        if command -v pv >/dev/null; then
            verbose_output "📦 Creating archive..."
            dir_size=$(cd "$source_dir" && du -sb --exclude="${exclude[*]#--exclude=}" "${backup_source[@]}" | awk '{sum+=$1} END{print sum}')
            ( cd "$source_dir" &&
                tar --ignore-failed-read \
                --blocking-factor=128 \
                --no-check-device \
                -cf - \
                --transform "s|^|$base_dir_name/|" \
                "${exclude[@]}" "${backup_source[@]}"
            ) | \
            pv --size "$dir_size" --progress --timer --rate --eta --bytes --name BACKUP --force --width "$(($(tput cols) - 10))" 2> >(test -t 2 && cat >&2 || cat >/dev/null) > "$backup_path/$folder_type-plex_backup.$extension"
        else
            verbose_output "📦 Creating archive..."
            ( cd "$source_dir" &&
                tar --ignore-failed-read -cf - --transform "s|^|$base_dir_name/|" "${exclude[@]}" "${backup_source[@]}"
            ) > "$backup_path/$folder_type-plex_backup.$extension"
        fi

        if tar -tf "$backup_path/$folder_type-plex_backup.$extension" >/dev/null; then
            verbose_output -e "\n✅ Verified: tar archive is readable"
        else
            echo -e "\n❌ ERROR: tar archive verification failed!"
            backup_failed=true
        fi
    fi
}

## Primary function to create a backup (Essential or Full)
create_backup() {
    local folder_type=$1
    verbose_output "\n=============================== 📦 $folder_type Backup ==============================="
    [ "$dry_run" == "true" ] && verbose_output "🧪 Dry run mode enabled — no files will be created or modified."
    source_dir=${source_dir%/}
    base_dir_name=$(basename "$source_dir")
    start=$(date +%s)
    backup_failed=false
    dest=$(realpath -s "$destination_dir")
    now="$(date +"%H.%M")"
    backup_path="$dest/$folder_type/$(date +%F)@$now"
    mkdir -p "$backup_path"

    if [ "$folder_type" == "Essential" ]; then
        backup_source=(
            "Plug-in Support/Databases"
            "Plug-in Support/Preferences"
            "Preferences.xml"
        )
        exclude=(
            "--exclude=Plug-in Support/Databases/dbtmp"
        )
    else
        backup_source=(".")
        exclude=(
            "--exclude=Cache"
            "--exclude=Codecs"
            "--exclude=Crash Reports"
            "--exclude=Diagnostics"
            "--exclude=Drivers"
            "--exclude=Logs"
        )
    fi

    if [ "$compress" == "true" ]; then
        if command -v zstd >/dev/null; then
            zst_backup
        else
            7z_backup
        fi
    else
        tar_backup
    fi

    if [ "$folder_type" == "Essential" ]; then
        essential_backup_size=$(du -sh "$backup_path/$folder_type-plex_backup.$extension" | awk '{print $1}')
    else
        full_backup_size=$(du -sh "$backup_path/$folder_type-plex_backup.$extension" | awk '{print $1}')
    fi

    full_backup_total_size=$( [ -d "$dest/Full/" ] && du -sh "$dest/Full/" | awk '{print $1}' || echo "0B" )
    essential_backup_total_size=$( [ -d "$dest/Essential/" ] && du -sh "$dest/Essential/" | awk '{print $1}' || echo "0B" )
    end=$(date +%s)

    if [ "$backup_failed" == "true" ]; then
        build_payload
        field_builder "Backup Status" "❌ Archive verification failed" "true"
        payload "Backup Failure"
        send_notification
        verbose_output "❌ Backup verification failed. Notification sent."
        return
    fi
    verbose_output "\n✅ Backup complete."
    calculate_runtime

    if [ "$dry_run" == "true" ]; then
        essential_backup_size="1.0G"
        full_backup_size="1.0G"
        full_backup_total_size="2.0G"
        essential_backup_total_size="2.0G"
        run_output="🧪 Dry Run: Simulated runtime"
    fi
}

## Detect whether Plex is running as Docker or systemd
get_plex_type() {

    if docker ps --format '{{.Names}}' 2>/dev/null | grep -qw plex; then
        plex_type="docker"
    elif [[ $(systemctl is-active plexmediaserver 2>/dev/null) == "active" ]]; then
        plex_type="systemctl"
    else
        echo "❌ ERROR: Plex is not running in Docker or systemctl"
        exit 1
    fi

}

## Stop Plex before backup if enabled
stop_plex() {
    if [ "$shutdown_plex" == "true" ] && [ "$dry_run" != "true" ]; then
        case "$backup_type" in
            essential|essential_no_full) backup_notification="Essential Backup" ;;
            full) backup_notification="Full Backup" ;;
            both) backup_notification="Essential & Full Backup" ;;
        esac

        get_plex_tyupe

        [[ "${full_backup,,}" == "true" ]] && echo "🔻 Plex detected as: $plex_type"

        if [ "$plex_type" == "docker" ]; then
            verbose_output "🛑 Stopping Docker Plex..."
            docker stop plex
        elif [ "$plex_type" == "systemctl" ]; then
            verbose_output "🛑 Stopping systemd Plex..."
            systemctl stop plexmediaserver.service
        fi

        build_payload
        field_builder "Plex is being shut down for a backup" "true"
        payload "Plex Status"
        send_notification
    fi
}

## Restart Plex after backup if enabled
start_plex() {
    if [ "$shutdown_plex" == "true" ]; then
        case "$backup_type" in
            essential|essential_no_full) backup_notification="Essential Backup" ;;
            full) backup_notification="Full Backup" ;;
            both) backup_notification="Essential & Full Backup" ;;
        esac

        [[ "${full_backup,,}" == "true" ]] && echo "🔼 Starting Plex ($plex_type)..."

        if [ "$plex_type" == "docker" ]; then
            verbose_output "🚀 Starting Docker Plex..."
            docker start plex
        elif [ "$plex_type" == "systemctl" ]; then
            verbose_output "🚀 Starting systemd Plex..."
            systemctl start plexmediaserver.service
        else
            echo "❌ ERROR: Plex type unknown. Cannot start."
            exit 1
        fi

        build_payload
        field_builder "Plex is being started after" "$backup_notification" "true"
        payload "Plex Status"
        send_notification
    fi
}

## Display debug output of current variables and runtime state
debug_output_function() {
    echo -e "\n===================== DEBUG INFO ====================="
    printf "%-25s %s\n" "Debug:" "$debug"
    printf "%-25s %s\n" "Source:" "$source_dir"
    printf "%-25s %s\n" "Destination:" "$destination_dir"
    printf "%-25s %s\n" "Keep essential:" "$keep_essential"
    printf "%-25s %s\n" "Keep full:" "$keep_full"
    printf "%-25s %s\n" "Full backup:" "$full_backup"
    printf "%-25s %s\n" "Force full backup:" "$force_full_backup"
    printf "%-25s %s\n" "Unraid notify:" "$unraid_notify"
    printf "%-25s %s\n" "Compress:" "$compress"
    printf "%-25s %s\n" "Dry run:" "$dry_run"
    printf "%-25s %s\n" "Quiet:" "$quiet"
    printf "%-25s %s\n" "Webhook:" "$webhook"
    printf "%-25s %s\n" "Bot name:" "$bot_name"
    printf "%-25s %s\n" "Channel:" "$channel"
    printf "%-25s %s\n" "Essential size:" "$essential_backup_size"
    printf "%-25s %s\n" "Essential total size:" "$essential_backup_total_size"
    printf "%-25s %s\n" "Full size:" "$full_backup_size"
    printf "%-25s %s\n" "Full total size:" "$full_backup_total_size"
    printf "%-25s %s\n" "Days since last full:" "$days"
    printf "%-25s %s\n" "Bar color (hex):" "$hex_bar_color"
    printf "%-25s %s\n" "Bar color (decimal):" "$decimal_bar_color"
    printf "%-25s %s\n" "Timestamp:" "$get_ts"
    printf "%-25s %s\n" "Last backup recorded:" "$lastbackup"
    printf "%-25s %s\n" "Backup type:" "$backup_type"
    printf "%-25s %s\n" "Shutdown Plex:" "$shutdown_plex"
    printf "%-25s %s\n" "Runtime:" "$run_output"
    printf "%-25s %s\n" "Config directory:" "$config_dir"
    printf "%-25s %s\n" "Plex Type:" "$plex_type"
    echo -e "======================================================\n"
}


## Restore backup from archive(s), optionally to a test location
run_restore() {
    get_plex_type
    echo -e "\n🛠️  Restore Mode Activated"
    echo "Which type of backup would you like to restore?"
    select restore_type in "Essential" "Full" "Both" "Cancel"; do
        case $restore_type in
            Essential|Full|Both) break ;;
            Cancel) echo "❌ Restore cancelled."; exit 0 ;;
            *) echo "Please choose a valid option." ;;
        esac
    done
    declare -A selected_backups

    if [ "$restore_type" == "Both" ]; then
        restore_types=("Full" "Essential")
    else
        restore_types=("$restore_type")
    fi

    for sub_type in "${restore_types[@]}"; do
        search_dir="$destination_dir/$sub_type"
        if [ ! -d "$search_dir" ]; then
            echo "❌ No backups found in $search_dir"
            exit 1
        fi

        echo -e "\n📦 Available $sub_type backups:"
        mapfile -t backup_paths < <(find "$search_dir" -mindepth 1 -maxdepth 1 -type d | sort -r)
        backups_list=()
        for path in "${backup_paths[@]}"; do
            label=$(basename "$path" | sed 's/@/ @ /')
            backups_list+=("$label")
        done

        if [ ${#backups_list[@]} -eq 0 ]; then
            echo "❌ No $sub_type backup directories found."
            exit 1
        fi

        select selected_label in "${backups_list[@]}" "Cancel"; do
            if [[ "$selected_label" == "Cancel" ]]; then
                echo "❌ Restore cancelled."
                exit 0
            fi
            for i in "${!backups_list[@]}"; do
                if [[ "${backups_list[$i]}" == "$selected_label" ]]; then
                    selected_backups["$sub_type"]="${backup_paths[$i]}"
                    break 2
                fi
            done
            echo "Please choose a valid option."
        done
    done

    restore_target="$source_dir"
    [ "$test_restore_flag" == "true" ] && restore_target="$(dirname "$0")/tmp_restore"
    echo -e "\n🚚 Restoring from:"
    for type in "${!selected_backups[@]}"; do
        echo " - $type: ${selected_backups[$type]}"
    done
    echo "📁 Target location: $restore_target"

    mkdir -p "$restore_target"

    if [ -d "$restore_target" ] && [ "$(ls "$restore_target")" ]; then
        echo "⚠️  WARNING: $restore_target is not empty and may be overwritten."
        read -rp "Are you sure you want to continue? (yes/no): " confirm
        [[ ! "$confirm" =~ ^[Yy](es)?$ ]] && echo "❌ Restore cancelled." && exit 1
    fi

    # Detect and shut down Plex if it's running - commented out for now for testing
    if [ "$test_restore_flag" != "true" ]; then
        if docker ps --format '{{.Names}}' | grep -qw plex; then
            plex_type="docker"
            echo "🛑 Stopping Plex (Docker)..."
            docker stop plex
        elif systemctl is-active --quiet plexmediaserver; then
            plex_type="systemctl"
            echo "🛑 Stopping Plex (systemd)..."
            systemctl stop plexmediaserver
        else
            plex_type=""
            echo "ℹ️  Plex does not appear to be running."
        fi
    else
        echo "🧪 Test restore: skipping Plex shutdown"
    fi

    for type in "${restore_types[@]}"; do
        archive_dir="${selected_backups[$type]}"
        archive=$(find "$archive_dir" -type f -name '*plex_backup.*' | head -n 1)
 
        if [ ! -f "$archive" ]; then
            echo "❌ No $type backup archive found in $archive_dir"
            exit 1
        fi
 
        echo "📦 Restoring $type backup from: $archive"
 
        if [[ "$archive" == *.7z ]]; then
            echo "📦 Extracting .7z archive..."
            if command -v pv >/dev/null; then
            7z x -so "$archive" | pv | tar --strip-components=1 -xf - -C "$restore_target"
            else
                7z x -o"$restore_target" "$archive"
            fi
        elif [[ "$archive" == *.zst ]]; then
            echo "📦 Extracting .zst archive..."
            if command -v pv >/dev/null; then
                zstd -dc "$archive" | pv | tar --strip-components=1 -xf - -C "$restore_target"
            else
                zstd -dc "$archive" | tar -xf - -C "$restore_target"
            fi
        else
            echo "📦 Extracting .tar archive..."
            if command -v pv >/dev/null; then
                tar --strip-components=1 -xf - -C "$restore_target" < <(pv "$archive" 2> >(test -t 2 && cat >&2 || cat >/dev/null))
            else
                tar -xf "$archive" -C "$restore_target"
            fi
        fi
    done

    echo -e "\n✅ Restore complete"
    # Restart Plex if it was stopped
    if [ "$test_restore_flag" != "true" ]; then
        read -rp "Would you like to start Plex now? (yes/no): " start_plex_confirm
        if [[ "$start_plex_confirm" =~ ^[Yy](es)?$ ]]; then
            if [ "$plex_type" == "docker" ]; then
                echo "🚀 Restarting Plex (Docker)..."
                docker start plex >> /dev/null
            elif [ "$plex_type" == "systemctl" ]; then
                echo "🚀 Restarting Plex (systemd)..."
                systemctl start plexmediaserver
            fi
        else
            echo "⏸️ Plex restart skipped."
        fi
    else
        echo "🧪 Test restore: skipping Plex restart"
    fi
    [ "$test_restore_flag" == "true" ] && echo "🧪 Files restored to test directory: $restore_target"
}

## Main logic controller for backup operations
main() {
    # Handle --help flag early and exit if needed
    for arg in "$@"; do
        if [[ "$arg" == "-h" || "$arg" == "--help" ]]; then
            display_help
            exit 0
        fi
    done
    # Parse CLI arguments
    handle_options "$@"
    # Load config file if enabled
    config_dir_setup
    log_dir="$script_path/logs"
    mkdir -p "$log_dir"

    # Rotate old logs (keep only the 10 most recent)
    ls -tp "$log_dir"/plex_backup_*.log 2>/dev/null | grep -v '/$' | tail -n +11 | xargs -r rm --

    log_file="$log_dir/plex_backup_$(date +%F@%H.%M).log"
    exec > >(sed 's/\x1b\[[0-9;]*m//g' | tee -a "$log_file") 2>&1
    [ "$use_config_file" == "true" ] && load_config_file
    if [ "$restore_flag" == "true" ] || [ "$test_restore_flag" == "true" ]; then    
        run_restore
        [[ "${debug,,}" == "true" ]] && debug_output_function
        exit 0
    fi
    # If dry run enabled, print a notice
    [ "$dry_run" == "true" ] && verbose_output "🧪 Dry run mode enabled — simulation only"
    # Convert hex color to decimal
    hex_to_decimal "$bar_color"
    # Validate directories and webhook
    check_config "$@"
    # Determine last backup date
    last_plex_backup="$config_dir/.last_plex_backup.tmp"

    [ -f "$last_plex_backup" ] && lastbackup=$(cat "$last_plex_backup") || lastbackup=0
    current_date=$(date +"%m/%d/%y")
    days=$((($(date --date="$current_date" +%s) - $(date --date="$lastbackup" +%s)) / (60 * 60 * 24)))

    start=$(date +%s)
    # Stop Plex if configured
    get_plex_type
    stop_plex

    # Run Essential backup (and conditionally Full)
    if [[ "${full_backup,,}" == "false" ]]; then
        backup_type="essential"
        create_backup "Essential"
        build_payload
        field_builder "Runtime" "$run_output" "true"
        field_builder "This Essential backup size" "$essential_backup_size" "false"
        field_builder "Total size of all Essential backups" "$essential_backup_total_size" "false"
        payload "Essential Backup"
        send_notification

        if [ "$force_full_backup" != 0 ] && { [ "$days" -ge "$force_full_backup" ] || [ "$lastbackup" == 0 ]; }; then
            backup_type="both"
            create_backup "Full"
            build_payload
            field_builder "Runtime" "$run_output" "true"
            field_builder "This Full backup size" "$full_backup_size" "false"
            field_builder "Total size of all Full backups" "$full_backup_total_size" "false"
            payload "Full Backup"
            send_notification
            days="0"
            echo "$current_date" > "$last_plex_backup"
        else
            backup_type="essential_no_full"
            verbose_output "📆 Skipping full backup (only $days days since last)"
        fi
    else
        # Or just run Full backup
        backup_type="full"
        create_backup "Full"
        [ "$backup_failed" == "true" ] && return
        build_payload
        field_builder "Runtime" "$run_output" "true"
        field_builder "This Essential backup size" "$essential_backup_size" "false"
        field_builder "This Full backup size" "$full_backup_size" "false"
        field_builder "Total size of all Essential backups" "$essential_backup_total_size" "false"
        field_builder "Total size of all Full backups" "$full_backup_total_size" "false"
        field_builder "Days since last Full backup" "$days" "false"
        payload "Full and Essential Backup"
        send_notification
        echo "$current_date" > "$last_plex_backup"
        days="0"
    fi

    # Start Plex back up if needed
    start_plex
    # Clean up old backups
    cleanup_function
    # Print summary and optionally notify or debug
    [[ "${unraid_notify,,}" == "true" ]] && unraid_notification
    [[ "${debug,,}" == "true" ]] && debug_output_function
    printf "\n==================== ✅ Backup Summary ====================\n"
    printf "🔁  \e[1;34m%-30s\e[0m %s\n" "Backup Type:" "$backup_type"
    printf "⏱️   \e[1;34m%-30s\e[0m %s\n" "Runtime:" "$run_output"
    printf "📁  \e[1;34m%-30s\e[0m %s\n" "Source Directory:" "$source_dir"
    printf "💾  \e[1;34m%-30s\e[0m %s\n" "Destination Directory:" "$destination_dir"

    if [[ "$backup_type" =~ essential|both|essential_no_full ]]; then
        printf "🧩  \e[1;34m%-30s\e[0m %s\n" "Essential Size:" "$essential_backup_size"
        printf "📚  \e[1;34m%-30s\e[0m %s\n" "Total Essential Backups:" "$essential_backup_total_size"
    fi

    if [[ "$backup_type" =~ full|both ]]; then
        printf "🗂️   \e[1;34m%-30s\e[0m %s\n" "Full Size:" "$full_backup_size"
        printf "📦  \e[1;34m%-30s\e[0m %s\n" "Total Full Backups:" "$full_backup_total_size"
    fi

    printf "🗓️   \e[1;34m%-30s\e[0m %s\n" "Days Since Last Full:" "$days"

    if [[ "${full_backup,,}" == "false" && "$force_full_backup" -ne 0 ]]; then
        next_full=$(( force_full_backup - days ))
        if (( next_full > 0 )); then
            printf "📅  \e[1;34m%-30s\e[0m in %s day(s)\n" "Next Full Backup:" "$next_full"
        else
            printf "📅  \e[1;34m%-30s\e[0m %s\n" "Next Full Backup:" "Today (forced by schedule)"
        fi
    fi

    printf "🧪  \e[1;34m%-30s\e[0m %s\n" "Dry Run Mode:" "$dry_run"

    [[ "$backup_failed" == "true" ]] && printf "\n\e[1;31m⚠️  Backup verification failed. Please review the logs.\e[0m\n"
    printf "===========================================================\n"
    verbose_output "✅ All Done!"
}

# Kick off the script with all passed arguments
main "$@"