#!/usr/bin/env python3
# coding: utf-8

import json

# Charger le notebook
with open('Introduction_to_LangGraph_for_Agents_Assignment_Version.ipynb', 'r') as file:
    notebook = json.load(file)

# Modifier la structure des widgets pour ajouter la clé 'state'
if 'widgets' in notebook['metadata']:
    # Sauvegarde des données widget existantes
    widget_data = notebook['metadata']['widgets'].get('application/vnd.jupyter.widget-state+json', {})
    
    # Réinitialiser la structure des widgets avec 'state'
    notebook['metadata']['widgets'] = {
        'state': {},  # Ajout de la clé 'state' manquante
        'application/vnd.jupyter.widget-state+json': widget_data
    }

# Enregistrer le notebook corrigé
with open('Introduction_to_LangGraph_for_Agents_Assignment_Version.ipynb', 'w') as file:
    json.dump(notebook, file, indent=2)

print("Le notebook a été corrigé. Vous pouvez maintenant le pousser sur GitHub.") 