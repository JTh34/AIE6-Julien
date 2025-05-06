#!/usr/bin/env python3
import json
import os

# Liste des notebooks à traiter
notebooks = [
    'AIE6_Julien_UnslothGRPOTraining.ipynb',
    'Inference_Bonus_AIE6_Julien_UnslothGRPOTraining_improved.ipynb',
    'Train_Bonus_AIE6_Julien_UnslothGRPOTraining_improved.ipynb'
]

for notebook_path in notebooks:
    if not os.path.exists(notebook_path):
        print(f"Le fichier {notebook_path} n'existe pas, passage au suivant")
        continue
        
    print(f"\nLecture du notebook: {notebook_path}")
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)

        # Supprimer la clé 'widgets' des métadonnées
        if 'metadata' in notebook and 'widgets' in notebook['metadata']:
            print(f"Suppression de la clé 'widgets' des métadonnées de {notebook_path}")
            del notebook['metadata']['widgets']
            
            # Sauvegarder le notebook modifié
            with open(notebook_path, 'w', encoding='utf-8') as f:
                json.dump(notebook, f, ensure_ascii=False, indent=2)
            
            print(f"Notebook sauvegardé: {notebook_path}")
        else:
            print(f"Pas de clé 'widgets' trouvée dans les métadonnées de {notebook_path}")
    except Exception as e:
        print(f"Erreur lors du traitement de {notebook_path}: {e}")

print("\nTraitement terminé pour tous les notebooks.") 