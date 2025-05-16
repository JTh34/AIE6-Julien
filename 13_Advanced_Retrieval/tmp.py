
from langchain_community.document_loaders.csv_loader import CSVLoader
from datetime import datetime, timedelta

documents = []

for i in range(1, 5):
  loader = CSVLoader(
      file_path=f"john_wick_{i}.csv",
      metadata_columns=["Review_Date", "Review_Title", "Review_Url", "Author", "Rating"]
  )

  movie_docs = loader.load()
  for doc in movie_docs:

    # Add the "Movie Title" (John Wick 1, 2, ...)
    doc.metadata["Movie_Title"] = f"John Wick {i}"

    # convert "Rating" to an `int`, if no rating is provided - assume 0 rating
    doc.metadata["Rating"] = int(doc.metadata["Rating"]) if doc.metadata["Rating"] else 0

    # newer movies have a more recent "last_accessed_at"
    doc.metadata["last_accessed_at"] = datetime.now() - timedelta(days=4-i)

  documents.extend(movie_docs)

from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Qdrant.from_documents(
    documents,
    embeddings,
    location=":memory:",
    collection_name="JohnWick"
)
.prompts import ChatPromptTemplate

RAG_TEMPLATE = """\
You are a helpful and kind assistant. Use the context provided below to answer the question.

If you do not know the answer, or are unsure, say you don't know.

Query:
{question}

Context:
{context}
"""

rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

from langchain_openai import ChatOpenAI

chat_model = ChatOpenAI(model="gpt-4.1-nano")

naive_retriever = vectorstore.as_retriever(search_kwargs={"k" : 10})

from langchain_community.retrievers import BM25Retriever

bm25_retriever = BM25Retriever.from_documents(documents)


from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

compressor = CohereRerank(model="rerank-english-v3.0")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=naive_retriever
)
from langchain.retrievers.multi_query import MultiQueryRetriever

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=naive_retriever, llm=chat_model
)



from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models

parent_docs = documents
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)

client = QdrantClient(location=":memory:")

client.create_collection(
    collection_name="full_documents",
    vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
)

parent_document_vectorstore = Qdrant(
    collection_name="full_documents", embeddings=OpenAIEmbeddings(model="text-embedding-3-small"), client=client
)

store = InMemoryStore()

parent_document_retriever = ParentDocumentRetriever(
    vectorstore = parent_document_vectorstore,
    docstore=store,
    child_splitter=child_splitter,
)

parent_document_retriever.add_documents(parent_docs, ids=None)

parent_document_retrieval_chain = (
    {"context": itemgetter("question") | parent_document_retriever, "question": itemgetter("question")}
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
)

from langchain.retrievers import EnsembleRetriever

retriever_list = [bm25_retriever, naive_retriever, parent_document_retriever, compression_retriever, multi_query_retriever]
equal_weighting = [1/len(retriever_list)] * len(retriever_list)

ensemble_retriever = EnsembleRetriever(
    retrievers=retriever_list, weights=equal_weighting
)




from langchain_experimental.text_splitter import SemanticChunker

semantic_chunker = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile"
)

semantic_documents = semantic_chunker.split_documents(documents)

semantic_vectorstore = Qdrant.from_documents(
    semantic_documents,
    embeddings,
    location=":memory:",
    collection_name="JohnWickSemantic"
)

semantic_retriever = semantic_vectorstore.as_retriever(search_kwargs={"k" : 10})

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from ragas.testset import TestsetGenerator
from ragas import EvaluationDataset
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness, ResponseRelevancy, ContextEntityRecall, NoiseSensitivity
from ragas import evaluate, RunConfig
from ragas import EvaluationDataset
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

# Imports LangChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.callbacks.tracers import LangChainTracer
from langchain.schema.runnable import RunnableConfig

# Imports LangSmith
from langsmith import Client, traceable
from langsmith.run_helpers import get_current_run_tree

# Imports RAGAS
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator
from ragas import EvaluationDataset
from ragas.metrics import (
    LLMContextRecall,
    Faithfulness,
    FactualCorrectness,
    ResponseRelevancy,
    ContextEntityRecall,
    NoiseSensitivity
)
from ragas import evaluate, RunConfig
# Import required libraries
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from langsmith import Client
from langchain_core.tracers import LangChainTracer
from langchain_core.runnables.config import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from operator import itemgetter
from copy import deepcopy
import uuid
from ragas.metrics import (
    LLMContextRecall,
    Faithfulness,
    FactualCorrectness,
    ResponseRelevancy,
    ContextEntityRecall
)
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig
from ragas import EvaluationDataset
from langchain_openai import ChatOpenAI


# Generate SDG
generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1-mini"))
generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
testset_dataset = generator.generate_with_langchain_docs(documents, testset_size=10)



testset_df = testset_dataset.to_pandas()
testset_df


from getpass import getpass
import uuid
api_key = getpass("Entrez votre clé API LangSmith: ")
os.environ["LANGSMITH_API_KEY"] = api_key
os.environ["LANGCHAIN_TRACING_V2"] = "true"

os.environ["LANGCHAIN_PROJECT"] = f"john-wick-retriever-{uuid.uuid4().hex[:8]}"
print(f"Projet LangSmith défini: {os.environ['LANGCHAIN_PROJECT']}")


retrievers = [
    {"name": "Naive Retriever", "retriever": naive_retriever},
    {"name": "BM25 Retriever", "retriever": bm25_retriever},
    {"name": "Contextual Compression", "retriever": compression_retriever},
    {"name": "Multi-Query Retriever", "retriever": multi_query_retriever},
    {"name": "Parent Document Retriever", "retriever": parent_document_retriever},
    {"name": "Ensemble Retriever", "retriever": ensemble_retriever},
    {"name": "Semantic Chunking Retriever", "retriever": semantic_retriever}
]

eval_llm = ChatOpenAI(model="gpt-4.1-mini")


# from langsmith import Client

# client = Client()
# project_name = f"john-wick-retrieval-{uuid.uuid4().hex[:8]}"
# print(f"LangSmith project name: {project_name}")



def evaluate_retrievers_with_langsmith(retrievers, testset_dataset, llm):
    print("Starting retriever evaluation with LangSmith tracking...")
    
    # Setup LangSmith client
    client = Client()
    
     
    # Create a unique project name for this evaluation run
    project_name = f"john-wick-retrieval-{uuid.uuid4().hex[:8]}"
    print(f"LangSmith project name: {project_name}")
    
    # Créer explicitement le projet dans LangSmith
    try:
        client.create_project(project_name=project_name)
        print(f"Projet '{project_name}' créé avec succès dans LangSmith")
    except Exception as e:
        print(f"Erreur lors de la création du projet: {e}")
    # Results storage
    results = []
    
    rag_prompt = ChatPromptTemplate.from_template("""
    You are a movie expert specialized in John Wick films.
    Use only the information from the provided context to answer the question.
    If you cannot answer based on the context, clearly indicate so.
    
    Question: {question}
    
    Context:
    {context}
    
    Answer:
    """)
    
    # Storage for updated testsets for each retriever
    updated_testsets = {}
    
    # Evaluate each retriever
    for retriever_info in retrievers:
        name = retriever_info["name"]
        retriever = retriever_info["retriever"]
        
        print(f"\nEvaluating: {name}")
        
        # Make a deep copy of the testset for this retriever
        retriever_testset = deepcopy(testset_dataset)

        document_counts = []
        
        # Build RAG chain
        rag_chain = (
            {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
            | RunnablePassthrough.assign(context=itemgetter("context"))
            | {"response": rag_prompt | llm, "context": itemgetter("context")}
        )
        
        # Test on each question in the testset
        for i, test_row in enumerate(tqdm(retriever_testset, desc=f"Testing {name}")):
            try:
                # Extract the question from testset
                question = test_row.eval_sample.user_input
                              
                # Configure LangSmith tracing
                run_config = RunnableConfig(
                    tags=["retriever_evaluation", name],
                    project_name=project_name,
                    metadata={"retriever": name, "question_id": i}
                )
                
                # Run chain with LangSmith tracing
                response = rag_chain.invoke(
                    {"question": question},
                    config=run_config
                )
                
                doc_count = len(response["context"])
                
              
                document_counts.append(doc_count)
                
                # Update testset with response for later evaluation
                test_row.eval_sample.response = response["response"].content
                test_row.eval_sample.retrieved_contexts = [
                    context.page_content for context in response["context"]
                ]
                
                # Small pause to avoid API rate limits
                time.sleep(0.2)
                
            except Exception as e:
                print(f"Error on question {i}: {e}")
        
        # Calculate averages
       
        avg_docs = np.mean(document_counts)
        
        # Store the updated testset for this retriever
        updated_testsets[name] = retriever_testset
        
        # Query LangSmith API to get cost data
        print(f"Retrieving cost data for {name} from LangSmith...")
        
        # Get runs from the project
        # CORRECTION: Utiliser run_filter pour remplacer filter
        try:
            runs = client.list_runs(
                project_name=project_name,
                run_filter={"tags": ["retriever_evaluation", name]}
            )
            
            # Calculate total tokens and cost
            total_prompt_tokens = 0
            total_completion_tokens = 0
            total_cost = 0
            retrieval_times = []
            
            for run in runs:
                if hasattr(run, "metrics") and run.metrics:
                    metrics = run.metrics
                    total_prompt_tokens += metrics.get("prompt_tokens", 0)
                    total_completion_tokens += metrics.get("completion_tokens", 0)
                    total_cost += metrics.get("cost", 0)

                if hasattr(run, "metrics") and "run_time" in run.metrics:
                    retrieval_times.append(run.metrics["run_time"])
                elif hasattr(run, "start_time") and hasattr(run, "end_time"):
                    run_time = (run.end_time - run.start_time).total_seconds()
                    retrieval_times.append(run_time)
                    
            # Average per question
            avg_prompt_tokens = total_prompt_tokens / len(testset_dataset) if testset_dataset else 0
            avg_completion_tokens = total_completion_tokens / len(testset_dataset) if testset_dataset else 0
            avg_cost = total_cost / len(testset_dataset) if testset_dataset else 0
            avg_time = np.mean(retrieval_times) if retrieval_times else 0

            # Store results
            results.append({
                "Retriever": name,
                "Avg Time (s)": avg_time  ,
                "Avg Docs": avg_docs,
                "Avg Prompt Tokens": avg_prompt_tokens,
                "Avg Completion Tokens": avg_completion_tokens,
                "Avg Cost ($)": avg_cost,
                "Quality Score": 0.0,  # Will be updated later with RAGAS
            })
            
            print(f"✓ {name}: {avg_time:.2f}s, {avg_docs:.1f} docs, ${avg_cost:.5f}/query")
        
        except Exception as e:
            print(f"Erreur lors de la récupération des runs pour {name}: {e}")
            # Continue avec des valeurs par défaut
            results.append({
                "Retriever": name,
                "Avg Time (s)": 0.0,
                "Avg Docs": avg_docs,
                "Avg Prompt Tokens": 0.0,
                "Avg Completion Tokens": 0.0,
                "Avg Cost ($)": 0.0,
                "Quality Score": 0.0,
            })
    
    # Return results DataFrame and updated testsets
    return pd.DataFrame(results), updated_testsets, project_name


def run_ragas_evaluation(updated_testsets, results_df, llm):

    print("\nRunning RAGAS evaluation...")
    
    # Create a wrapper for the LLM for RAGAS
    evaluator_llm = LangchainLLMWrapper(llm)
    
    # Configure RAGAS run parameters
    run_config = RunConfig(
        timeout=600,  
        max_workers=2  
    )
    
    # Define RAGAS metrics to evaluate
    metrics = [
        LLMContextRecall(),
        Faithfulness(),
        FactualCorrectness(),
        ResponseRelevancy(),
        ContextEntityRecall()
    ]
    
    # Store RAGAS results for each retriever
    ragas_results = {}
    import os
import uuid
import time
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from copy import deepcopy
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig, RunnablePassthrough
from langsmith import Client

def fixed_evaluate_retrievers_with_langsmith(retrievers, testset_dataset, llm):
    """
    Version corrigée de evaluate_retrievers_with_langsmith qui utilise 
    la variable d'environnement LANGCHAIN_PROJECT plutôt que de créer un nouveau projet.
    """
    print("Starting retriever evaluation with LangSmith tracking...")
    
    # Setup LangSmith client
    client = Client()
    
    # Utiliser le projet défini dans les variables d'environnement ou en créer un nouveau
    if "LANGCHAIN_PROJECT" in os.environ and os.environ["LANGCHAIN_PROJECT"]:
        project_name = os.environ["LANGCHAIN_PROJECT"]
        print(f"Utilisation du projet LangSmith existant: {project_name}")
    else:
        # Create a unique project name for this evaluation run
        project_name = f"john-wick-retrieval-{uuid.uuid4().hex[:8]}"
        os.environ["LANGCHAIN_PROJECT"] = project_name
        print(f"Nouveau projet LangSmith créé: {project_name}")
    
    # Créer explicitement le projet dans LangSmith (si nécessaire)
    try:
        client.create_project(project_name=project_name)
        print(f"Projet '{project_name}' créé avec succès dans LangSmith")
    except Exception as e:
        print(f"Note: {e}")
        
    # Results storage
    results = []
    
    rag_prompt = ChatPromptTemplate.from_template("""
    You are a movie expert specialized in John Wick films.
    Use only the information from the provided context to answer the question.
    If you cannot answer based on the context, clearly indicate so.
    
    Question: {question}
    
    Context:
    {context}
    
    Answer:
    """)
    
    # Storage for updated testsets for each retriever
    updated_testsets = {}
    
    # Evaluate each retriever
    for retriever_info in retrievers:
        name = retriever_info["name"]
        retriever = retriever_info["retriever"]
        
        print(f"\nEvaluating: {name}")
        
        # Make a deep copy of the testset for this retriever
        retriever_testset = deepcopy(testset_dataset)
        document_counts = []
        
        # Build RAG chain
        rag_chain = (
            {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
            | RunnablePassthrough.assign(context=itemgetter("context"))
            | {"response": rag_prompt | llm, "context": itemgetter("context")}
        )
        
        # Test on each question in the testset
        for i, test_row in enumerate(tqdm(retriever_testset, desc=f"Testing {name}")):
            try:
                # Extract the question from testset
                question = test_row.eval_sample.user_input
                
                # IMPORTANT: Définir un nom de run unique pour ce test
                run_name = f"{name}-q{i}-{uuid.uuid4().hex[:4]}"
                
                # Configure LangSmith tracing avec plus de métadonnées
                run_config = RunnableConfig(
                    tags=["retriever_evaluation", name],
                    project_name=project_name,
                    name=run_name,
                    metadata={
                        "retriever": name, 
                        "question_id": i,
                        "timestamp": time.time()
                    }
                )
                
                # Capture le temps de début
                start_time = time.time()
                
                # Run chain with LangSmith tracing
                response = rag_chain.invoke(
                    {"question": question},
                    config=run_config
                )
                
                # Capture le temps d'exécution
                run_time = time.time() - start_time
                
                doc_count = len(response["context"])
                document_counts.append(doc_count)
                
                # Update testset with response for later evaluation
                test_row.eval_sample.response = response["response"].content
                test_row.eval_sample.retrieved_contexts = [
                    context.page_content for context in response["context"]
                ]
                
                # Tenter d'ajouter manuellement les métriques de temps
                try:
                    runs = client.list_runs(
                        project_name=project_name,
                        run_filter={"run_name": run_name}
                    )
                    for run in runs:
                        client.update_run(
                            run.id,
                            metrics={"run_time": run_time}
                        )
                except Exception as e:
                    print(f"Impossible de mettre à jour les métriques: {e}")
                
                # Small pause to avoid API rate limits
                time.sleep(0.2)
                
            except Exception as e:
                print(f"Error on question {i}: {e}")
        
        # Calculate averages
        avg_docs = np.mean(document_counts)
        
        # Store the updated testset for this retriever
        updated_testsets[name] = retriever_testset
        
        # Query LangSmith API to get cost data
        print(f"Retrieving cost data for {name} from LangSmith...")
        
        # Get runs from the project with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Pause pour donner du temps à LangSmith pour traiter les runs
                time.sleep(2)
                
                runs = client.list_runs(
                    project_name=project_name,
                    run_filter={"tags": ["retriever_evaluation", name]}
                )
                
                # Convertir l'itérateur en liste pour pouvoir le compter
                runs_list = list(runs)
                
                if not runs_list:
                    print(f"Tentative {attempt+1}/{max_retries}: Aucun run trouvé. Attente de 2 secondes...")
                    continue
                
                # Calculate total tokens and cost
                total_prompt_tokens = 0
                total_completion_tokens = 0
                total_cost = 0
                retrieval_times = []
                
                for run in runs_list:
                    if hasattr(run, "metrics") and run.metrics:
                        metrics = run.metrics
                        total_prompt_tokens += metrics.get("prompt_tokens", 0)
                        total_completion_tokens += metrics.get("completion_tokens", 0)
                        total_cost += metrics.get("cost", 0)
                    
                    # Vérifier différentes façons de récupérer le temps
                    if hasattr(run, "metrics") and "run_time" in run.metrics:
                        retrieval_times.append(run.metrics["run_time"])
                    elif hasattr(run, "start_time") and hasattr(run, "end_time") and run.start_time and run.end_time:
                        try:
                            run_time = (run.end_time - run.start_time).total_seconds()
                            retrieval_times.append(run_time)
                        except Exception as e:
                            print(f"Erreur de calcul du temps: {e}")
                
                # Si nous avons des données de temps, nous pouvons sortir de la boucle
                if retrieval_times:
                    break
                
                print(f"Tentative {attempt+1}/{max_retries}: Pas de données de temps trouvées. Attente...")
                
            except Exception as e:
                print(f"Erreur lors de la tentative {attempt+1}: {e}")
                time.sleep(2)  # Pause avant nouvelle tentative
        
        # Average per question
        num_questions = len(testset_dataset)
        avg_prompt_tokens = total_prompt_tokens / num_questions if num_questions else 0
        avg_completion_tokens = total_completion_tokens / num_questions if num_questions else 0
        avg_cost = total_cost / num_questions if num_questions else 0
        avg_time = np.mean(retrieval_times) if retrieval_times else 0
        
        # Si nous n'avons pas de données de temps depuis LangSmith, utilisez nos mesures manuelles
        if avg_time == 0 and document_counts:
            print("Aucun temps de LangSmith trouvé, calcul direct utilisé")
            avg_time = run_time  # Utiliser le dernier temps mesuré comme approximation
        
        # Store results
        results.append({
            "Retriever": name,
            "Avg Time (s)": avg_time,
            "Avg Docs": avg_docs,
            "Avg Prompt Tokens": avg_prompt_tokens,
            "Avg Completion Tokens": avg_completion_tokens,
            "Avg Cost ($)": avg_cost,
            "Quality Score": 0.0,  # Will be updated later with RAGAS
        })
        
        print(f"✓ {name}: {avg_time:.2f}s, {avg_docs:.1f} docs, ${avg_cost:.5f}/query")
    
    # Return results DataFrame and updated testsets
    return pd.DataFrame(results), updated_testsets, project_name



def run_complete_evaluation(testset_dataset, retrievers, llm):
    print("Starting retriever evaluation pipeline...")
    
    # Step 1: Evaluate retrievers and get costs from LangSmith
    results_df, updated_testsets, project_name = evaluate_retrievers_with_langsmith(
        retrievers, testset_dataset, llm
    )
    
    # Step 2: Run RAGAS evaluation for each retriever
    results_df, ragas_results = run_ragas_evaluation(updated_testsets, results_df, llm)
    
 
    
    print(f"\nEvaluation complete! Results available in LangSmith project: {project_name}")
    print("Evaluation visuals saved in 'evaluation_plots' directory")
    print("Full report saved as 'retriever_evaluation_report.md'")
    
    return {
        "results_df": results_df,
        "report": report,
        "ragas_results": ragas_results,
        "langsmith_project": project_name
    }

evaluation_results = run_complete_evaluation(testset_dataset, retrievers, eval_llm)