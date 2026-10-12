#!/bin/bash

BACKUP_DIR="backups"
WATCH_IP_FILE=".watch_ip"
DEFAULT_WATCH_IP="10.158.237.77"

connect_watch() {
    local current_ip="$DEFAULT_WATCH_IP"
    if [ -f "$WATCH_IP_FILE" ]; then
        local saved_ip
        saved_ip=$(cat "$WATCH_IP_FILE" 2>/dev/null | tr -d '[:space:]')
        [ -n "$saved_ip" ] && current_ip="$saved_ip"
    fi

    read -p "Enter watch IPv4 address [default: $current_ip]: " input_ip
    local watch_ip="${input_ip:-$current_ip}"
    echo "$watch_ip" > "$WATCH_IP_FILE"

    read -p "Enter dynamic port for watch: " dyn_port
    if [ -z "$dyn_port" ]; then
        echo "Error: Dynamic port cannot be empty."
        return 1
    fi

    echo "Connecting to ${watch_ip}:${dyn_port}..."
    adb connect "${watch_ip}:${dyn_port}"
    adb -s "${watch_ip}:${dyn_port}" reverse tcp:8765 tcp:8765
}

run_in_new_tab() {
    local title="$1"
    local cmd="$2"

    if command -v alacritty &>/dev/null; then
        alacritty -t "$title" -e bash -c "$cmd; exec bash" &
    elif command -v konsole &>/dev/null; then
        konsole --new-tab -p tabtitle="$title" -e bash -c "$cmd; exec bash" &
    elif command -v kitty &>/dev/null; then
        kitty --title "$title" bash -c "$cmd; exec bash" &
    elif command -v gnome-terminal &>/dev/null; then
        gnome-terminal --tab --title="$title" -- bash -c "$cmd; exec bash"
    elif command -v x-terminal-emulator &>/dev/null; then
        x-terminal-emulator -e bash -c "$cmd; exec bash" &
    else
        bash -c "$cmd"
    fi
}

count_file_lines() {
    local file="$1"
    if [ -f "$file" ]; then
        grep -cve '^\s*$' "$file" 2>/dev/null || wc -l < "$file"
    else
        echo 0
    fi
}

create_backup() {
    local note="$1"
    if [ ! -f "training_data.txt" ] && [ ! -f "hr_data.txt" ]; then
        echo "No training data files (training_data.txt or hr_data.txt) found to back up."
        return 1
    fi

    mkdir -p "$BACKUP_DIR"
    local timestamp
    timestamp=$(date +"%Y%m%d_%H%M%S")
    local dest="$BACKUP_DIR/backup_${timestamp}"
    mkdir -p "$dest"

    local n_csi
    local n_hr
    n_csi=$(count_file_lines "training_data.txt")
    n_hr=$(count_file_lines "hr_data.txt")

    [ -f "training_data.txt" ] && cp "training_data.txt" "$dest/"
    [ -f "hr_data.txt" ] && cp "hr_data.txt" "$dest/"
    [ -f "watch_hr_log.csv" ] && cp "watch_hr_log.csv" "$dest/"
    [ -f "csi_data.csv" ] && cp "csi_data.csv" "$dest/"

    cat <<EOF > "$dest/info.txt"
Date: $(date "+%Y-%m-%d %H:%M:%S")
Note: ${note:-Manual backup}
CSI Rows: $n_csi
HR Rows: $n_hr
EOF

    echo "Backup saved to: $dest"
    echo "  -> CSI: $n_csi rows, HR: $n_hr rows [${note:-Manual backup}]"
    return 0
}

restore_backup() {
    if [ ! -d "$BACKUP_DIR" ]; then
        echo "No backups directory found ($BACKUP_DIR)."
        return 1
    fi

    local backup_dirs=()
    while IFS= read -r dir; do
        [ -n "$dir" ] && backup_dirs+=("$dir")
    done < <(find "$BACKUP_DIR" -maxdepth 1 -mindepth 1 -type d -name "backup_*" 2>/dev/null | sort -r)

    if [ ${#backup_dirs[@]} -eq 0 ]; then
        echo "No backups found in $BACKUP_DIR/."
        return 1
    fi

    echo ""
    echo "Available Backups:"
    local i=1
    for dir in "${backup_dirs[@]}"; do
        local b_name
        b_name=$(basename "$dir")
        local b_info=""
        if [ -f "$dir/info.txt" ]; then
            local csi_r hr_r b_note
            csi_r=$(grep "CSI Rows:" "$dir/info.txt" | cut -d':' -f2- | xargs)
            hr_r=$(grep "HR Rows:" "$dir/info.txt" | cut -d':' -f2- | xargs)
            b_note=$(grep "Note:" "$dir/info.txt" | cut -d':' -f2- | xargs)
            b_info=" (CSI: ${csi_r:-0}, HR: ${hr_r:-0} | $b_note)"
        fi
        echo "$i) $b_name$b_info"
        ((i++))
    done
    echo ""

    read -p "Select backup number to restore (or Enter to cancel): " choice_idx
    if [[ "$choice_idx" =~ ^[0-9]+$ ]] && [ "$choice_idx" -ge 1 ] && [ "$choice_idx" -le "${#backup_dirs[@]}" ]; then
        local target="${backup_dirs[$((choice_idx-1))]}"
        read -p "Are you sure you want to restore $(basename "$target")? Current files will be replaced! [y/N]: " confirm
        if [[ "$confirm" =~ ^[Yy]$ ]]; then
            if [ -f "training_data.txt" ] || [ -f "hr_data.txt" ]; then
                create_backup "Auto-backup before restore" >/dev/null 2>&1
            fi
            [ -f "$target/training_data.txt" ] && cp "$target/training_data.txt" .
            [ -f "$target/hr_data.txt" ] && cp "$target/hr_data.txt" .
            [ -f "$target/watch_hr_log.csv" ] && cp "$target/watch_hr_log.csv" .
            [ -f "$target/csi_data.csv" ] && cp "$target/csi_data.csv" .
            echo "Successfully restored from $(basename "$target")!"
        else
            echo "Restore cancelled."
        fi
    else
        echo "Cancelled."
    fi
}

revert_to_latest_backup() {
    if [ ! -d "$BACKUP_DIR" ]; then
        echo "No backups directory found."
        return 1
    fi

    local latest
    latest=$(find "$BACKUP_DIR" -maxdepth 1 -mindepth 1 -type d -name "backup_*" 2>/dev/null | sort -r | head -n 1)

    if [ -z "$latest" ]; then
        echo "No backups found in $BACKUP_DIR/."
        return 1
    fi

    local b_name
    b_name=$(basename "$latest")
    echo "Latest backup found: $b_name"
    if [ -f "$latest/info.txt" ]; then
        cat "$latest/info.txt" | sed 's/^/  /'
    fi

    read -p "Revert current training data to this backup? [y/N]: " confirm
    if [[ "$confirm" =~ ^[Yy]$ ]]; then
        if [ -f "training_data.txt" ] || [ -f "hr_data.txt" ]; then
            create_backup "Auto-backup before revert" >/dev/null 2>&1
        fi
        [ -f "$latest/training_data.txt" ] && cp "$latest/training_data.txt" .
        [ -f "$latest/hr_data.txt" ] && cp "$latest/hr_data.txt" .
        [ -f "$latest/watch_hr_log.csv" ] && cp "$latest/watch_hr_log.csv" .
        [ -f "$latest/csi_data.csv" ] && cp "$latest/csi_data.csv" .
        echo "Successfully reverted to $b_name!"
    else
        echo "Revert cancelled."
    fi
}

view_backup_status() {
    echo ""
    echo "=== CURRENT DATA STATUS ==="
    echo "  training_data.txt: $(count_file_lines "training_data.txt") rows"
    echo "  hr_data.txt:       $(count_file_lines "hr_data.txt") rows"
    if [ -f "watch_hr_log.csv" ]; then
        echo "  watch_hr_log.csv:  $(wc -l < "watch_hr_log.csv") rows"
    fi
    echo ""
    echo "=== SAVED BACKUPS ==="
    if [ -d "$BACKUP_DIR" ]; then
        local count=0
        while IFS= read -r dir; do
            if [ -n "$dir" ]; then
                local b_name
                b_name=$(basename "$dir")
                local b_info=""
                if [ -f "$dir/info.txt" ]; then
                    local csi_r hr_r b_note
                    csi_r=$(grep "CSI Rows:" "$dir/info.txt" | cut -d':' -f2- | xargs)
                    hr_r=$(grep "HR Rows:" "$dir/info.txt" | cut -d':' -f2- | xargs)
                    b_note=$(grep "Note:" "$dir/info.txt" | cut -d':' -f2- | xargs)
                    b_info=" -> CSI: ${csi_r:-0}, HR: ${hr_r:-0} ($b_note)"
                fi
                echo "  $b_name$b_info"
                ((count++))
            fi
        done < <(find "$BACKUP_DIR" -maxdepth 1 -mindepth 1 -type d -name "backup_*" 2>/dev/null | sort -r)
        if [ $count -eq 0 ]; then
            echo "  (No backups yet)"
        fi
    else
        echo "  (No backups directory)"
    fi
}

nuke_training_data() {
    if [ ! -f "training_data.txt" ] && [ ! -f "hr_data.txt" ]; then
        echo "No training data files found (already clean)."
        return 0
    fi

    echo ""
    echo "====================================================="
    echo "  ⚠️   WARNING: NUKE TRAINING DATASET  ⚠️"
    echo "====================================================="
    echo "Current dataset to be deleted:"
    echo "  - training_data.txt ($(count_file_lines "training_data.txt") rows)"
    echo "  - hr_data.txt       ($(count_file_lines "hr_data.txt") rows)"
    echo ""

    read -p "Create an automatic safety backup before nuking? [Y/n]: " backup_first
    if [[ ! "$backup_first" =~ ^[Nn]$ ]]; then
        create_backup "Pre-nuke-archive"
    fi

    echo ""
    read -p "Are you sure you want to permanently delete current data? Type 'yes' or 'nuke' to confirm: " confirm
    if [[ "$confirm" == "yes" || "$confirm" == "nuke" ]]; then
        rm -f "training_data.txt" "hr_data.txt"
        echo "✓ Removed training_data.txt and hr_data.txt."

        if [ -f "watch_hr_log.csv" ] || [ -f "csi_data.csv" ] || [ -f "csi_data_log.txt" ]; then
            read -p "Also clear raw logs (watch_hr_log.csv, csi_data.csv)? [y/N]: " clear_logs
            if [[ "$clear_logs" =~ ^[Yy]$ ]]; then
                rm -f "watch_hr_log.csv" "csi_data.csv" "csi_data_log.txt"
                echo "✓ Cleared raw log files."
            fi
        fi
        echo ""
        echo "💥 Training data successfully nuked! You are ready to start fresh."
    else
        echo "Nuke operation cancelled. No files were removed."
    fi
}

manage_backups() {
    while true; do
        echo ""
        echo "===== DATA MANAGEMENT & BACKUPS ====="
        echo "1. Create a new backup now"
        echo "2. Revert to latest backup"
        echo "3. Choose a backup to restore"
        echo "4. View data & backup status"
        echo "5. Nuke training data (start fresh)"
        echo "6. Return to main menu"
        echo "====================================="
        read -p "Enter choice [1-6]: " b_choice
        case $b_choice in
            1)
                read -p "Enter a label/note for this backup (optional): " b_note
                create_backup "$b_note"
                read -p "Press Enter to continue..."
                ;;
            2)
                revert_to_latest_backup
                read -p "Press Enter to continue..."
                ;;
            3)
                restore_backup
                read -p "Press Enter to continue..."
                ;;
            4)
                view_backup_status
                read -p "Press Enter to continue..."
                ;;
            5)
                nuke_training_data
                read -p "Press Enter to continue..."
                ;;
            6)
                break
                ;;
            *)
                echo "Invalid choice."
                ;;
        esac
    done
}

while true; do
    clear
    echo "===== SETUP SCRIPT ====="
    echo "1. Watch connection"
    echo "2. Run watch script"
    echo "3. Data collection"
    echo "4. Manage data (Backup / Restore / Nuke)"
    echo "5. Model training"
    echo "6. Run actual program"
    echo "7. Exit"
    echo "======================="
    read -p "Enter choice [1-7]: " choice
    case $choice in
        1)
            connect_watch
            read -p "Press Enter to continue..."
            ;;
        2)
            run_in_new_tab "Watch HR" "python3 watch/watch_hr.py -p 8765"
            read -p "Press Enter to continue..."
            ;;
        3)
            if [ -f "training_data.txt" ] || [ -f "hr_data.txt" ]; then
                read -p "Existing training data detected. Create a backup first? [Y/n]: " AUTO_BACKUP
                if [[ ! "$AUTO_BACKUP" =~ ^[Nn]$ ]]; then
                    create_backup "Auto-backup before collection"
                fi
            fi
            read -p "Enter USB port for receiver (e.g., /dev/ttyUSB0): " USB_PORT
            read -p "Append to existing training data? [y/N]: " APPEND_TRAINING
            if [[ "$APPEND_TRAINING" =~ ^[Yy]$ ]]; then
                APPEND_FLAG="--training-append"
            else
                APPEND_FLAG=""
            fi
            read -p "Max samples (-1 for unlimited): " MAX_SAMPLES
            if [[ -z "$MAX_SAMPLES" ]]; then
                MAX_SAMPLES="-1"
            fi
            run_in_new_tab "Data Collection" "python3 read_and_process_csi.py -p \"$USB_PORT\" --collect-training-data $APPEND_FLAG --max-samples $MAX_SAMPLES"
            read -p "Press Enter to continue..."
            ;;
        4)
            manage_backups
            ;;
        5)
            run_in_new_tab "Model Training" "python3 train.py"
            read -p "Press Enter to continue..."
            ;;
        6)
            read -p "Enter USB port for receiver (e.g., /dev/ttyUSB0): " USB_PORT
            run_in_new_tab "Actual Program" "python3 read_and_process_csi.py -p $USB_PORT"
            read -p "Press Enter to continue..."
            ;;
        7)
            exit 0
            ;;
        *)
            echo "Invalid option. Press Enter to try again."
            read -p ""
            ;;
    esac
done
