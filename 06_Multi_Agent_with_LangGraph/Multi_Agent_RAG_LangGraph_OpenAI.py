from typing import TypedDict, Annotated, List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.schema import Document
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.tools.tavily_search import TavilySearchResults
import tiktoken

from langgraph.graph import StateGraph, END

# State pour notre graphe
class RAGState(TypedDict):
    subject: str
    question: str
    documents: List[Document]
    answer: str

# Fonction pour rechercher des articles
def search_papers(state):
    subject = state["subject"]
    
    print(f"\n===== RECHERCHE D'ARTICLES SUR: {subject} =====\n")
    
    # Utilisation de l'outil ArxivQueryRun intégré
    arxiv_tool = ArxivQueryRun()
    
    try:
        # Utiliser l'outil ArxivQueryRun pour rechercher des articles
        results = arxiv_tool.invoke({"query": subject})
        
        # Analyser les résultats pour créer des objets Document
        documents = []
        
        # Traitement des résultats
        if results and isinstance(results, str):
            entries = results.split("\n\n")
            
            for entry in entries:
                if not entry.strip():
                    continue
                
                # Extraire les informations de chaque article
                lines = entry.split("\n")
                title = ""
                authors = ""
                summary = ""
                
                for line in lines:
                    if line.startswith("Title:"):
                        title = line.replace("Title:", "").strip()
                    elif line.startswith("Authors:"):
                        authors = line.replace("Authors:", "").strip()
                    elif line.startswith("Summary:"):
                        summary = line.replace("Summary:", "").strip()
                
                if title:
                    doc = Document(
                        page_content=f"Title: {title}\nSummary: {summary}",
                        metadata={"title": title, "authors": authors}
                    )
                    documents.append(doc)
        
        if not documents:
            print("Aucun article trouvé.")
            return {
                "subject": subject,
                "question": state["question"],
                "documents": [],
                "answer": "Je n'ai pas pu trouver d'articles pertinents sur ce sujet."
            }
        
        # Affichage détaillé des articles trouvés
        print(f"Articles trouvés: {len(documents)}\n")
        for i, doc in enumerate(documents, 1):
            print(f"Article {i}:")
            print(f"Titre: {doc.metadata['title']}")
            print(f"Auteurs: {doc.metadata['authors']}")
            print(f"Résumé: {doc.page_content.split('Summary: ', 1)[1][:200]}...\n")
        
        # Liste des titres pour affichage
        titles = [f"- {doc.metadata['title']} par {doc.metadata['authors']}" for doc in documents]
        
        return {
            "subject": subject,
            "question": state["question"],
            "documents": documents,
            "answer": f"J'ai trouvé {len(documents)} articles sur {subject}:\n" + "\n".join(titles)
        }
    
    except Exception as e:
        print(f"Erreur lors de la recherche d'articles: {e}")
        return {
            "subject": subject,
            "question": state["question"],
            "documents": [],
            "answer": f"Une erreur s'est produite lors de la recherche d'articles: {str(e)}"
        }

# Fonction pour effectuer le RAG
def perform_rag(state):
    documents = state["documents"]
    question = state["question"]
    
    if not documents:
        return state
    
    print(f"\n===== TRAITEMENT RAG POUR LA QUESTION: {question} =====\n")
    
    # Découpage en chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30,
        length_function=lambda x: len(tiktoken.encoding_for_model("gpt-4o-mini").encode(x)),
    )
    chunks = text_splitter.split_documents(documents)
    
    print(f"Documents découpés en {len(chunks)} chunks pour l'indexation vectorielle\n")
    
    # Création du vectorstore
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Qdrant.from_documents(
        chunks,
        embedding_model,
        location=":memory:",
        collection_name="arxiv_chunks",
    )
    
    # Recherche par similarité
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    relevant_chunks = retriever.invoke(question)
    
    print(f"Chunks les plus pertinents pour la question:\n")
    for i, chunk in enumerate(relevant_chunks, 1):
        print(f"Chunk {i} (de l'article '{chunk.metadata.get('title', 'inconnu')}'):")
        print(f"{chunk.page_content[:150]}...\n")
    
    # Construction du contexte et collecte des sources
    sources = set()
    context = ""
    for chunk in relevant_chunks:
        context += chunk.page_content + "\n\n"
        if "title" in chunk.metadata:
            sources.add(chunk.metadata["title"])
    
    # Génération de la réponse
    rag_prompt = f"""
Contenu de l'article:
{context}

Question: 
{question}

Réponds à la question en te basant uniquement sur le contenu des articles.
"""
    rag_model = ChatOpenAI(model="gpt-4o-mini")
    response = rag_model.invoke([HumanMessage(content=rag_prompt)])
    
    # Citation des sources
    sources_text = "Sources utilisées: " + ", ".join(sources)
    
    return {
        "subject": state["subject"],
        "question": question,
        "documents": documents,
        "answer": f"{response.content}\n\n{sources_text}"
    }

# Démonstration de l'usage des outils comme demandé
def demo_tools():
    print("Démonstration de l'usage des outils comme demandé par l'utilisateur:")
    
    # Création des outils comme demandé dans l'exemple
    tavily_tool = TavilySearchResults(max_results=5)
    
    tool_belt = [
        tavily_tool,
        ArxivQueryRun(),
    ]
    
    # Affichage des outils disponibles
    print("\nOutils disponibles dans tool_belt:")
    for i, tool in enumerate(tool_belt, 1):
        print(f"{i}. {tool.__class__.__name__}")
    
    return tool_belt

# Construction et exécution du graphe
def run_arxiv_rag(subject, question):
    # Création du graphe
    graph = StateGraph(RAGState)
    
    # Ajout des nœuds
    graph.add_node("search", search_papers)
    graph.add_node("rag", perform_rag)
    
    # Définition du point d'entrée
    graph.set_entry_point("search")
    
    # Ajout des arêtes
    graph.add_edge("search", "rag")
    graph.add_edge("rag", END)
    
    # Compilation du graphe
    compiled_graph = graph.compile()
    
    print(f"\n===== DÉMARRAGE DE LA RECHERCHE =====")
    print(f"Sujet: {subject}")
    print(f"Question: {question}\n")
    
    # Exécution du graphe
    result = compiled_graph.invoke({
        "subject": subject,
        "question": question,
        "documents": [],
        "answer": ""
    })
    
    print("\n===== RÉSULTAT FINAL =====\n")
    print(result["answer"])
    
    return result

# Exemple d'utilisation
if __name__ == "__main__":
    # Démonstration des outils
    tools = demo_tools()
    
    # Exécution du RAG
    subject = "correction d'erreur quantique"
    question = "Quelles sont les dernières avancées en matière de correction d'erreur quantique?"
    
    result = run_arxiv_rag(subject, question)
    
    print("\n===== RÉSULTAT =====\n")
    print(result["answer"]) 