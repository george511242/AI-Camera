#!/bin/bash

# Create audio directory if it doesn't exist
mkdir -p /app/audio

# Check if no-phone.wav exists in audio directory
if [ ! -f /app/audio/no-phone.wav ]; then
    # Check if source file exists
    if [ -f /app/no-phone.wav ]; then
        echo "Copying default no-phone.wav to audio directory..."
        cp /app/no-phone.wav /app/audio/no-phone.wav
        echo "Default audio file copied successfully."
    else
        echo "Warning: Default audio file /app/no-phone.wav not found."
    fi
else
    echo "Audio file /app/audio/no-phone.wav already exists."
fi
./scaling_frequency.sh -c rk3566

# Start the application
exec python -m server
