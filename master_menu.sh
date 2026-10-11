#!/bin/bash

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
            python3 watch/watch_hr.py -p 8765
            read -p "Press Enter to continue..."
            ;;
        3)
            read -p "Enter USB port for receiver (e.g., /dev/ttyUSB0): " USB_PORT
            python3 read_and_process_csi.py --collect_training_data --training_append -p $USB_PORT
            read -p "Press Enter to continue..."
            ;;
        4)
            python3 train.py
            read -p "Press Enter to continue..."
            ;;
        5)
            read -p "Enter USB port for receiver (e.g., /dev/ttyUSB0): " USB_PORT
            python3 read_and_process_csi.py -p $USB_PORT
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
