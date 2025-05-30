import json

from typing_extensions import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.graph import START, END, StateGraph

from assistant.configuration import Configuration, SearchAPI
from assistant.utils import deduplicate_and_format_sources, tavily_search, format_sources, perplexity_search, duckduckgo_search
from assistant.state import SummaryState, SummaryStateInput, SummaryStateOutput
from assistant.prompts import query_writer_instructions, summarizer_instructions, reflection_instructions

from assistant.utils import setup_qdrant_vectorstore,format_rag_results, search_local_knowledge

# Nodes
def generate_query(state: SummaryState, config: RunnableConfig):
    """ Generate a query for web search """

    # Format the prompt
    query_writer_instructions_formatted = query_writer_instructions.format(research_topic=state.research_topic)

    # Generate a query
    configurable = Configuration.from_runnable_config(config)
    llm_json_mode = ChatOllama(base_url=configurable.ollama_base_url, model=configurable.local_llm, temperature=0, format="json")
    result = llm_json_mode.invoke(
        [SystemMessage(content=query_writer_instructions_formatted),
        HumanMessage(content=f"Generate a query for web search:")]
    )
    query = json.loads(result.content)

    return {"search_query": query['query']}

def web_research(state: SummaryState, config: RunnableConfig):
    """ Gather information from the web """

    # Configure
    configurable = Configuration.from_runnable_config(config)

    # Handle both cases for search_api:
    # 1. When selected in Studio UI -> returns a string (e.g. "tavily")
    # 2. When using default -> returns an Enum (e.g. SearchAPI.TAVILY)
    if isinstance(configurable.search_api, str):
        search_api = configurable.search_api
    else:
        search_api = configurable.search_api.value

    # Search the web
    if search_api == "tavily":
        search_results = tavily_search(state.search_query, include_raw_content=True, max_results=1)
        search_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=1000, include_raw_content=True)
    elif search_api == "perplexity":
        search_results = perplexity_search(state.search_query, state.research_loop_count)
        search_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=1000, include_raw_content=False)
    elif search_api == "duckduckgo":
        search_results = duckduckgo_search(state.search_query, max_results=3, fetch_full_page=configurable.fetch_full_page)
        search_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=1000, include_raw_content=True)
    else:
        raise ValueError(f"Unsupported search API: {configurable.search_api}")

    return {"sources_gathered": [format_sources(search_results)], "research_loop_count": state.research_loop_count + 1, "web_research_results": [search_str]}

def summarize_sources(state: SummaryState, config: RunnableConfig):
    """ Summarize the gathered sources """

    # Existing summary
    existing_summary = state.running_summary

    # Most recent web research
    most_recent_web_research = state.web_research_results[-1]

    # Build the human message
    if existing_summary:
        human_message_content = (
            f"<User Input> \n {state.research_topic} \n <User Input>\n\n"
            f"<Existing Summary> \n {existing_summary} \n <Existing Summary>\n\n"
            f"<New Search Results> \n {most_recent_web_research} \n <New Search Results>"
        )
    else:
        human_message_content = (
            f"<User Input> \n {state.research_topic} \n <User Input>\n\n"
            f"<Search Results> \n {most_recent_web_research} \n <Search Results>"
        )

    # Run the LLM
    configurable = Configuration.from_runnable_config(config)
    llm = ChatOllama(base_url=configurable.ollama_base_url, model=configurable.local_llm, temperature=0)
    result = llm.invoke(
        [SystemMessage(content=summarizer_instructions),
        HumanMessage(content=human_message_content)]
    )

    running_summary = result.content

    # TODO: This is a hack to remove the <think> tags w/ Deepseek models
    # It appears very challenging to prompt them out of the responses
    while "<think>" in running_summary and "</think>" in running_summary:
        start = running_summary.find("<think>")
        end = running_summary.find("</think>") + len("</think>")
        running_summary = running_summary[:start] + running_summary[end:]

    return {"running_summary": running_summary}

def reflect_on_summary(state: SummaryState, config: RunnableConfig):
    """ Reflect on the summary and generate a follow-up query """

    # Generate a query
    configurable = Configuration.from_runnable_config(config)
    llm_json_mode = ChatOllama(base_url=configurable.ollama_base_url, model=configurable.local_llm, temperature=0, format="json")
    result = llm_json_mode.invoke(
        [SystemMessage(content=reflection_instructions.format(research_topic=state.research_topic)),
        HumanMessage(content=f"Identify a knowledge gap and generate a follow-up web search query based on our existing knowledge: {state.running_summary}")]
    )
    follow_up_query = json.loads(result.content)

    # Get the follow-up query
    query = follow_up_query.get('follow_up_query')

    # JSON mode can fail in some cases
    if not query:

        # Fallback to a placeholder query
        return {"search_query": f"Tell me more about {state.research_topic}"}

    # Update search query with follow-up query
    return {"search_query": follow_up_query['follow_up_query']}

# def finalize_summary(state: SummaryState):
#     """ Finalize the summary """

#     # Format all accumulated sources into a single bulleted list
#     all_sources = "\n".join(source for source in state.sources_gathered)
#     state.running_summary = f"## Summary\n\n{state.running_summary}\n\n ### Sources:\n{all_sources}"
#     return {"running_summary": state.running_summary}

def enhanced_summarize_sources(state: SummaryState, config: RunnableConfig):
    """Enhanced version that combines RAG + Web Search."""
    
    # Existing summary
    existing_summary = state.running_summary
    
    # RAG results
    rag_content = state.rag_results if state.rag_found else ""
    
    # Web results (most recent)
    web_content = state.web_research_results[-1] if state.web_research_results else ""
    
    # Build the message according to what is available
    if existing_summary:
        human_message_content = (
            f"<User Input>\n{state.research_topic}\n</User Input>\n\n"
            f"<Existing Summary>\n{existing_summary}\n</Existing Summary>\n\n"
        )
    else:
        human_message_content = (
            f"<User Input>\n{state.research_topic}\n</User Input>\n\n"
        )
    
    # Add RAG content if available
    if rag_content and state.rag_found:
        human_message_content += f"<Local Knowledge Base>\n{rag_content}\n</Local Knowledge Base>\n\n"
    
    # Add web content if available
    if web_content:
        human_message_content += f"<Web Search Results>\n{web_content}\n</Web Search Results>"
    
    # Special instructions for the LLM
    enhanced_instructions = """
    <GOAL>
    Generate a comprehensive summary that intelligently combines information from multiple sources:
    1. Local knowledge base (authoritative, curated information)
    2. Web search results (current, real-time information)
    3. Existing summary (accumulated knowledge)
    </GOAL>

    <PRIORITY>
    - Prioritize information from the local knowledge base when available
    - Use web results to complement or update local knowledge
    - Clearly indicate when information comes from different sources
    - Resolve any conflicts between sources by noting the discrepancy
    </PRIORITY>

    <FORMATTING>
    Start directly with the updated summary, without preamble. Do not use XML tags in the output.
    </FORMATTING>"""
    
    # Run the LLM
    configurable = Configuration.from_runnable_config(config)
    llm = ChatOllama(
        base_url=configurable.ollama_base_url, 
        model=configurable.local_llm, 
        temperature=0
    )
    
    result = llm.invoke([
        SystemMessage(content=enhanced_instructions),
        HumanMessage(content=human_message_content)
    ])
    
    running_summary = result.content
    
    # Cleanup for DeepSeek
    while "<think>" in running_summary and "</think>" in running_summary:
        start = running_summary.find("<think>")
        end = running_summary.find("</think>") + len("</think>")
        running_summary = running_summary[:start] + running_summary[end:]
    
    return {"running_summary": running_summary}

# def route_research(state: SummaryState, config: RunnableConfig) -> Literal["finalize_summary", "web_research"]:
#     """ Route the research based on the follow-up query """

#     configurable = Configuration.from_runnable_config(config)
#     if state.research_loop_count <= int(configurable.max_web_research_loops):
#         return "web_research"
#     else:
#         return "finalize_summary"
    
# Modify route_research to point to the new node
def route_research(state: SummaryState, config: RunnableConfig) -> Literal["enhanced_finalize_summary", "web_research"]:
    """Route the research based on the follow-up query"""
    configurable = Configuration.from_runnable_config(config)
    if state.research_loop_count <= int(configurable.max_web_research_loops):
        return "web_research"
    else:
        return "enhanced_finalize_summary"  

def local_knowledge_search(state: SummaryState, config: RunnableConfig):
    """Search in the local knowledge base with QDrant."""
    
    configurable = Configuration.from_runnable_config(config)
    
    # Initialize QDrant
    vectorstore = setup_qdrant_vectorstore(configurable)
    
    if not vectorstore:
        print("QDrant not available, switching directly to web search")
        return {
            "rag_results": "QDrant not available",
            "rag_found": False,
            "rag_sources": []
        }
    
    # Search in the local base
    search_results = search_local_knowledge(
        vectorstore, 
        state.search_query, 
        configurable
    )
    
    # Format the results
    formatted_results = format_rag_results(search_results)
    
    # Sources for the final bibliography
    sources = []
    if search_results["found"]:
        for result in search_results["results"]:
            sources.append(f"* {result['source']} (Score: {result['score']:.2f})")
    
    return {
        "rag_results": formatted_results,
        "rag_found": search_results["found"],
        "rag_sources": sources
    }



def route_after_rag(state: SummaryState, config: RunnableConfig) -> Literal["web_research", "enhanced_summarize_sources"]:
    """Decide if a web search is needed after RAG."""
    
    # If RAG found relevant results, sometimes we can be satisfied with that
    if state.rag_found:
        # Here you can add more sophisticated logic
        # For example, always do a web search to get the most up-to-date info
        return "web_research"
    else:
        # No RAG results, go directly to the web
        return "web_research"

def enhanced_finalize_summary(state: SummaryState):
    """Enhanced version that includes RAG and web sources."""
    
    # Web sources
    web_sources = "\n".join(source for source in state.sources_gathered)
    
    # RAG sources
    rag_sources = "\n".join(source for source in state.rag_sources)
    
    # Combine all sources
    all_sources = ""
    if rag_sources:
        all_sources += "### Local knowledge base sources:\n" + rag_sources + "\n\n"
    if web_sources:
        all_sources += "### Web sources:\n" + web_sources
    
    # Final summary
    final_summary = f"## Summary\n\n{state.running_summary}\n\n{all_sources}"
    
    return {"running_summary": final_summary}

# Add nodes and edges
builder = StateGraph(SummaryState, input=SummaryStateInput, output=SummaryStateOutput, config_schema=Configuration)
builder.add_node("generate_query", generate_query)
builder.add_node("local_knowledge_search", local_knowledge_search)  # NOUVEAU
builder.add_node("web_research", web_research)
builder.add_node("enhanced_summarize_sources", enhanced_summarize_sources)  # MODIFIÉ
builder.add_node("reflect_on_summary", reflect_on_summary)
builder.add_node("enhanced_finalize_summary", enhanced_finalize_summary)  # MODIFIÉ

# Add edges 
builder.add_edge(START, "generate_query")
builder.add_edge("generate_query", "local_knowledge_search")  
builder.add_conditional_edges("local_knowledge_search", route_after_rag)  
builder.add_edge("web_research", "enhanced_summarize_sources") 
builder.add_edge("enhanced_summarize_sources", "reflect_on_summary")
builder.add_conditional_edges("reflect_on_summary", route_research)
builder.add_edge("enhanced_finalize_summary", END)


graph = builder.compile()