#!/bin/bash

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

while true; do
    clear
    echo "===== SETUP SCRIPT ====="
    echo "1. Watch connection"
    echo "2. Run watch script"
    echo "3. Data collection"
    echo "4. Model training"
    echo "5. Run actual program"
    echo "6. Exit"
    echo "======================="
    read -p "Enter choice [1-6]: " choice
    case $choice in
        1)
            read -p "Enter dynamic port for watch: " DYN_PORT
            adb connect 10.235.245.9:$DYN_PORT
            adb -s 10.235.245.9:$DYN_PORT reverse tcp:8765 tcp:8765
            read -p "Press Enter to continue..."
            ;;
        2)
            run_in_new_tab "Watch HR" "python3 watch/watch_hr.py -p 8765"
            read -p "Press Enter to continue..."
            ;;
        3)
            read -p "Enter USB port for receiver (e.g., /dev/ttyUSB0): " USB_PORT
            run_in_new_tab "Data Collection" "python3 read_and_process_csi.py -p $USB_PORT"
            read -p "Press Enter to continue..."
            ;;
        4)
            run_in_new_tab "Model Training" "python3 train.py"
            read -p "Press Enter to continue..."
            ;;
        5)
            read -p "Enter USB port for receiver (e.g., /dev/ttyUSB0): " USB_PORT
            run_in_new_tab "Actual Program" "python3 read_and_process_csi.py -p $USB_PORT"
            read -p "Press Enter to continue..."
            ;;
        6)
            exit 0
            ;;
        *)
            echo "Invalid option. Press Enter to try again."
            read -p ""
            ;;
    esac
done
