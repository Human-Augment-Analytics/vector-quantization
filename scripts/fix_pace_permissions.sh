#!/bin/bash
# Fix permissions for PACE ICE shared storage
# Run this after deploying code or if you encounter permission errors

echo "Fixing permissions for PACE shared storage..."

# Fix logs directory and database
if [ -d "logs" ]; then
    echo "Setting permissions on logs/"
    chmod -R 777 logs/
    if [ -f "logs/benchmark_runs.db" ]; then
        chmod 666 logs/benchmark_runs.db
        echo "✓ Fixed logs/benchmark_runs.db"
    fi
fi

# Fix codebooks directory
if [ -d "codebooks" ]; then
    echo "Setting permissions on codebooks/"
    chmod -R 777 codebooks/
    echo "✓ Fixed codebooks/"
fi

# Fix any existing plots
if [ -d "plots" ]; then
    echo "Setting permissions on plots/"
    chmod -R 777 plots/
    echo "✓ Fixed plots/"
fi

echo "✓ Done! Permissions fixed for shared storage."
