#!/usr/bin/env python3
# coding: utf-8

import json
import os
import shutil
from datetime import datetime

# Nom du fichier notebook
notebook_file = 'Introduction_to_LangGraph_for_Agents_Assignment_Version.ipynb'

# Créer une copie de sauvegarde avant de modifier
backup_file = f"{notebook_file}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
try:
    shutil.copy2(notebook_file, backup_file)
    print(f"Copie de sauvegarde créée: {backup_file}")
except Exception as e:
    print(f"Erreur lors de la création de la sauvegarde: {e}")
    exit(1)

try:
    # Tenter de charger le JSON du notebook
    with open(notebook_file, 'r', encoding='utf-8') as file:
        try:
            notebook = json.load(file)
            print("Le notebook a été chargé avec succès")
            
            # Correction de la structure des widgets
            if 'metadata' in notebook and 'widgets' in notebook['metadata']:
                widget_data = notebook['metadata']['widgets'].get('application/vnd.jupyter.widget-state+json', {})
                notebook['metadata']['widgets'] = {
                    'state': {},
                    'application/vnd.jupyter.widget-state+json': widget_data
                }
                print("Structure de widgets corrigée")
            
            # Écrire le notebook corrigé
            with open(notebook_file, 'w', encoding='utf-8') as out_file:
                json.dump(notebook, out_file, indent=2, ensure_ascii=False)
                
            print(f"Le notebook a été réparé et enregistré dans {notebook_file}")
            print("Vous pouvez maintenant le pousser sur GitHub.")
            
        except json.JSONDecodeError as json_err:
            print(f"Le notebook n'est pas un JSON valide: {json_err}")
            print("Tentative de réparation basique du JSON...")
            
            # Lecture du fichier en tant que texte
            with open(notebook_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Créer un fichier de sortie minimal
            minimal_notebook = {
                "cells": [],
                "metadata": {
                    "kernelspec": {
                        "display_name": "Python 3",
                        "language": "python",
                        "name": "python3"
                    },
                    "language_info": {
                        "codemirror_mode": {
                            "name": "ipython",
                            "version": 3
                        },
                        "file_extension": ".py",
                        "mimetype": "text/x-python",
                        "name": "python",
                        "nbconvert_exporter": "python",
                        "pygments_lexer": "ipython3",
                        "version": "3.9.0"
                    },
                    "widgets": {
                        "state": {},
                        "application/vnd.jupyter.widget-state+json": {}
                    }
                },
                "nbformat": 4,
                "nbformat_minor": 4
            }
            
            # Sauvegarder le notebook minimal
            with open(f"{notebook_file}.minimal", 'w', encoding='utf-8') as out_file:
                json.dump(minimal_notebook, out_file, indent=2, ensure_ascii=False)
                
            print(f"Un notebook minimal a été créé dans {notebook_file}.minimal")
            print("Vous pouvez renommer ce fichier pour remplacer votre notebook original.")
            
except Exception as e:
    print(f"Une erreur inattendue s'est produite: {e}")