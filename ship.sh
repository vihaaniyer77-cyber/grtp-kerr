#!/bin/bash
# ship.sh - Automates pushing the clean code to GitHub

# Set terminal to exit on error
set -e

# Prompt for details
read -p "Enter your GitHub username: " USERNAME
read -p "Enter repository name [grtp-kerr]: " REPONAME
REPONAME=${REPONAME:-grtp-kerr}

echo -e "\n---> Initializing Git repository..."
git init

echo -e "\n---> Adding files..."
# Ignore the script itself so it doesn't get committed to github
if [ -f .gitignore ]; then
    echo "ship.sh" >> .gitignore
else
    echo "ship.sh" > .gitignore
    echo ".DS_Store" >> .gitignore
fi
git add .

echo -e "\n---> Creating initial commit..."
git commit -m "Initial commit: General Relativistic Test-Particle Tracking"

echo -e "\n---> Setting default branch to main..."
git branch -M main

# Configure remote origin
REMOTE_URL="https://github.com/$USERNAME/$REPONAME.git"
git remote add origin "$REMOTE_URL" 2>/dev/null || git remote set-url origin "$REMOTE_URL"

echo -e "\n=============================================================="
echo "IMPORTANT:"
echo "Please make sure you have created a blank repository named"
echo "\"$REPONAME\" on your GitHub account (https://github.com/new)"
echo "DO NOT initialize it with a README, license, or .gitignore."
echo "=============================================================="
read -p "Press [Enter] when ready to push to GitHub..."

echo -e "\n---> Pushing code to GitHub..."
git push -u origin main

echo -e "\n=============================================================="
echo "Success! Your repository is live at:"
echo "🔗 https://github.com/$USERNAME/$REPONAME"
echo "=============================================================="
