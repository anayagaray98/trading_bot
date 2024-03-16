#!/bin/bash

file_path="output.txt"
max_size=5242880  # 5 MB in bytes

while true; do
    file_size=$(stat -c %s "output.txt")
    if [ "$file_size" -gt "104857600" ]; then
        truncate -s 0 "output.txt"  # Truncate the file to 0 bytes
    fi
    sleep 3600  # Check file size every hour
done
