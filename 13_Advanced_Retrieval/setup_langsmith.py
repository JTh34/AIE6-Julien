import os
import uuid
from getpass import getpass

# Définir les variables d'environnement nécessaires pour LangSmith
if not os.environ.get("LANGSMITH_API_KEY"):
    api_key = getpass("Entrez votre clé API LangSmith: ")
    os.environ["LANGSMITH_API_KEY"] = api_key

# Activer le traçage LangChain
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Créer un nom de projet unique et le définir
project_name = f"john-wick-retriever-{uuid.uuid4().hex[:8]}"
os.environ["LANGCHAIN_PROJECT"] = project_name

print(f"\n✅ Configuration LangSmith complétée:")
print(f"   - LANGCHAIN_TRACING_V2: {os.environ.get('LANGCHAIN_TRACING_V2')}")
print(f"   - LANGCHAIN_PROJECT: {project_name}")
print(f"   - LANGSMITH_API_KEY: {'définie' if os.environ.get('LANGSMITH_API_KEY') else 'non définie'}")
print("\nImportez ce script dans votre notebook avec:")
print("from setup_langsmith import project_name") 