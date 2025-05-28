#!/bin/bash

# Script to fetch only new folders from upstream
# Usage: ./sync_new_folders.sh [folder_name]

set -e

echo "🔄 Fetching latest changes from upstream..."
git fetch upstream

if [ $# -eq 1 ]; then
    # If a specific folder is provided
    FOLDER=$1
    echo "📁 Checking if folder '$FOLDER' exists in upstream..."
    
    if git ls-tree -r --name-only upstream/main | grep -q "^$FOLDER/"; then
        echo "✅ Folder '$FOLDER' found in upstream"
        
        # Check if the folder already exists locally
        if [ -d "$FOLDER" ]; then
            echo "⚠️  Folder '$FOLDER' already exists locally"
            read -p "Do you want to replace it? (y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                rm -rf "$FOLDER"
                echo "🗑️  Local folder deleted"
            else
                echo "❌ Operation cancelled"
                exit 1
            fi
        fi
        
        echo "📥 Fetching folder '$FOLDER'..."
        git checkout upstream/main -- "$FOLDER/"
        echo "✅ Folder '$FOLDER' successfully fetched!"
        
        echo "📋 Folder contents:"
        ls -la "$FOLDER/"
        
    else
        echo "❌ Folder '$FOLDER' not found in upstream"
        exit 1
    fi
else
    # List all new folders available
    echo "📋 Folders available in upstream but not locally:"
    
    # Get the list of folders in upstream
    upstream_folders=$(git ls-tree -d --name-only upstream/main | grep -E '^[0-9]+_' | sort)
    
    # Get the list of local folders
    local_folders=$(find . -maxdepth 1 -type d -name '[0-9]*_*' | sed 's|^\./||' | sort)
    
    # Find the differences
    new_folders=$(comm -23 <(echo "$upstream_folders") <(echo "$local_folders"))
    
    if [ -z "$new_folders" ]; then
        echo "✅ No new folders available"
    else
        echo "$new_folders"
        echo ""
        echo "💡 To fetch a specific folder, use:"
        echo "   ./sync_new_folders.sh FOLDER_NAME"
    fi
fi

echo ""
echo "🎯 Current git status:"
git status --short 