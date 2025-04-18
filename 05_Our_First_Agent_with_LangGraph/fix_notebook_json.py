#!/usr/bin/env python3
# coding: utf-8

import json
import nbformat

try:
    # Charger le notebook en tant que JSON brut
    with open('Introduction_to_LangGraph_for_Agents_Assignment_Version.ipynb', 'r', encoding='utf-8') as file:
        notebook_json = json.load(file)
    
    # Modifier la structure des widgets pour ajouter la clé 'state'
    if 'widgets' in notebook_json.get('metadata', {}):
        # Sauvegarde des données widget existantes
        widget_data = notebook_json['metadata']['widgets'].get('application/vnd.jupyter.widget-state+json', {})
        
        # Réinitialiser la structure des widgets avec 'state'
        notebook_json['metadata']['widgets'] = {
            'state': {},  # Ajout de la clé 'state' manquante
            'application/vnd.jupyter.widget-state+json': widget_data
        }

    # Convertir en notebook nbformat pour garantir un format valide
    notebook = nbformat.from_dict(notebook_json)
    
    # Écrire le notebook avec nbformat pour garantir un format valide
    nbformat.write(notebook, 'Introduction_to_LangGraph_for_Agents_Assignment_Version.ipynb')
    
    print("Le notebook a été corrigé avec nbformat. Vous pouvez maintenant le pousser sur GitHub.")
    
except Exception as e:
    print(f"Erreur lors de la correction du notebook: {e}")
    
    # Alternative: approche de correction en traitant le fichier comme texte brut
    print("Tentative de récupération du notebook en créant une copie simplifiée...")
    
    try:
        with open('Introduction_to_LangGraph_for_Agents_Assignment_Version.ipynb', 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Créer une version simplifiée du notebook
        with open('Introduction_to_LangGraph_for_Agents_Assignment_Version_backup.ipynb', 'w', encoding='utf-8') as file:
            file.write(content)
            
        print("Une copie de sauvegarde a été créée. Si les corrections automatiques échouent, essayez d'utiliser cette copie.")
    except Exception as e:
        print(f"Erreur lors de la création de la copie simplifiée: {e}") 