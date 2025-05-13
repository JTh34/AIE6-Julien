#!/bin/bash

# Liste des dossiers à protéger
PROTECTED_DIRS=(
  "11_App_Challenge/puppycompanion-app"
  "04_Production_RAG/DataRepository"
)

# Sauvegarde des dossiers protégés
echo "Sauvegarde des dossiers protégés..."
BACKUP_DIR=$(mktemp -d)
for dir in "${PROTECTED_DIRS[@]}"; do
  if [ -d "$dir" ]; then
    echo "Sauvegarde de $dir..."
    mkdir -p "$(dirname "$BACKUP_DIR/$dir")"
    cp -r "$dir" "$(dirname "$BACKUP_DIR/$dir")"
  fi
done

# Récupération des fichiers depuis upstream
echo "Récupération des fichiers depuis upstream..."
git fetch upstream

# Création de la liste de tous les fichiers dans upstream/main
git ls-tree -r --name-only upstream/main > /tmp/upstream_files.txt

# Pour chaque fichier dans upstream/main
while read file; do
  # Vérifier si le fichier est dans un répertoire protégé
  protected=false
  for dir in "${PROTECTED_DIRS[@]}"; do
    if [[ "$file" == "$dir"/* ]]; then
      protected=true
      break
    fi
  done
  
  # Si non protégé, récupérer depuis upstream
  if [ "$protected" = false ]; then
    echo "Récupération de $file..."
    git checkout upstream/main -- "$file"
  fi
done < /tmp/upstream_files.txt

# Restauration des dossiers protégés
echo "Restauration des dossiers protégés..."
for dir in "${PROTECTED_DIRS[@]}"; do
  if [ -d "$BACKUP_DIR/$dir" ]; then
    echo "Restauration de $dir..."
    rm -rf "$dir"
    mkdir -p "$(dirname "$dir")"
    cp -r "$BACKUP_DIR/$dir" "$(dirname "$dir")"
  fi
done

echo "Synchronisation terminée!" 