#!/bin/bash
# run_and_package.sh - Runs the experiments and packages the entire code + results onto the Desktop.
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "=============================================================="
echo "1. Running experiments (quick test mode)..."
echo "=============================================================="
./run_all.sh --test

echo -e "\n=============================================================="
echo "2. Packaging complete project to Desktop..."
echo "=============================================================="
PACKAGE_DIR="$HOME/Desktop/grtp_kerr_complete"
rm -rf "$PACKAGE_DIR"
mkdir -p "$PACKAGE_DIR"

# Copy source, experiments, and tests folders
cp -R grtp "$PACKAGE_DIR/"
cp -R experiments "$PACKAGE_DIR/"
cp -R tests "$PACKAGE_DIR/"

# Copy config and documentation files
cp README.md pyproject.toml summary_report.txt run_all.sh "$PACKAGE_DIR/"

# Copy generated data and figures
cp -R data "$PACKAGE_DIR/"
cp -R figures "$PACKAGE_DIR/"

# Remove Python cache directories from the package for cleanliness
find "$PACKAGE_DIR" -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$PACKAGE_DIR" -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true

echo -e "\n=============================================================="
echo "Success! The complete package (code + figures + datasets) is at:"
echo "📂 $PACKAGE_DIR"
echo "=============================================================="
