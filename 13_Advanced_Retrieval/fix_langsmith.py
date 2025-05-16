import uuid
from langsmith import Client

def create_langsmith_project():
    """
    Crée un projet LangSmith avec un nom unique et le retourne pour être utilisé dans le notebook.
    """
    # Créer un nom de projet unique
    project_name = f"john-wick-retrieval-{uuid.uuid4().hex[:8]}"
    
    # Initialiser le client LangSmith
    client = Client()
    
    # Créer explicitement le projet
    try:
        client.create_project(project_name=project_name)
        print(f"✅ Projet '{project_name}' créé avec succès dans LangSmith")
        
        # Définir la variable d'environnement
        print(f"Pour utiliser ce projet, exécutez la commande suivante:")
        print(f"export LANGCHAIN_PROJECT={project_name}")
        
        return project_name
    except Exception as e:
        print(f"❌ Erreur lors de la création du projet: {e}")
        return None

if __name__ == "__main__":
    project_name = create_langsmith_project()
    if project_name:
        print("\nÀ partir de maintenant, vous pouvez modifier votre code pour:")
        print("1. Importer et utiliser ce projet directement")
        print("2. Ajouter le code suivant dans votre notebook pour créer le projet explicitement:\n")
        print("```python")
        print("from langsmith import Client")
        print("client = Client()")
        print(f"project_name = \"{project_name}\"")
        print("try:")
        print("    client.create_project(project_name=project_name)")
        print("    print(f\"Projet '{project_name}' créé avec succès\")")
        print("except Exception as e:")
        print("    print(f\"Erreur: {e}\")")
        print("```") 