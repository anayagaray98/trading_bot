#!/bin/bash

file_path="output.txt"

# Make the cleanup script executable
chmod +x cleanup_output.sh

# Start the cleanup script in the background
./cleanup_output.sh &

bot_script="bot.py"
bot_pid=0

while true; do
    # Kill all previous bot script processes
    pkill -9 -f "$bot_script"

    # Run your Python script in the background and capture its PID
    python3 "$bot_script" >> "$file_path" &
    bot_pid=$!

    # Wait for the bot script to complete
    wait $bot_pid
done
