
    # Retriever Evaluation Report

    ## Results Summary

    | Retriever | Total Time (s) | Retrieval (s) | LLM (s) | Docs | Quality | Cost ($) | Efficiency |
    |-----------|---------------|--------------|---------|------|---------|----------|------------|
    | Naive Retriever | 0.00 | 0.00 | 0.00 | 0.0 | 0.00 | $0.00000 | 0.0 |
| BM25 Retriever | 0.00 | 0.00 | 0.00 | 0.0 | 0.00 | $0.00000 | 0.0 |
| Contextual Compression | 0.00 | 0.00 | 0.00 | 0.0 | 0.00 | $0.00000 | 0.0 |
| Multi-Query Retriever | 0.00 | 0.00 | 0.00 | 0.0 | 0.00 | $0.00000 | 0.0 |
| Parent Document Retriever | 0.00 | 0.00 | 0.00 | 0.0 | 0.00 | $0.00000 | 0.0 |
| Ensemble Retriever | 0.00 | 0.00 | 0.00 | 0.0 | 0.00 | $0.00000 | 0.0 |
| Semantic Chunking Retriever | 0.00 | 0.00 | 0.00 | 0.0 | 0.00 | $0.00000 | 0.0 |

        ## RAGAS Metrics Details

        | Retriever | LLM Context Recall | Faithfulness | Factual Correctness | Response Relevancy | Context Entity Recall |
        |-----------|-------------------|--------------|---------------------|-------------------|----------------------|
        | Naive Retriever | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| BM25 Retriever | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Contextual Compression | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Multi-Query Retriever | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Parent Document Retriever | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Ensemble Retriever | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Semantic Chunking Retriever | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

    ## Latency Analysis

    | Retriever | Total Time (s) | Retrieval % | LLM % | Processing % |
    |-----------|---------------|------------|-------|--------------|
    | Naive Retriever | 0.00 | 0.0% | 0.0% | 0.0% |
| BM25 Retriever | 0.00 | 0.0% | 0.0% | 0.0% |
| Contextual Compression | 0.00 | 0.0% | 0.0% | 0.0% |
| Multi-Query Retriever | 0.00 | 0.0% | 0.0% | 0.0% |
| Parent Document Retriever | 0.00 | 0.0% | 0.0% | 0.0% |
| Ensemble Retriever | 0.00 | 0.0% | 0.0% | 0.0% |
| Semantic Chunking Retriever | 0.00 | 0.0% | 0.0% | 0.0% |

    ## Best Retrievers by Category

    | Category | Best Retriever |
    |----------|---------------|
    | Overall Quality | Naive Retriever |
    | Cost Efficiency | Naive Retriever |
    | Time Efficiency | Naive Retriever |
    | Lowest Cost | Naive Retriever |
    | Fastest Response | Naive Retriever |

    ## Analysis

    Based on our evaluation of retrievers using the John Wick movie reviews dataset:

    1. **Naive Retriever** achieved the highest quality results with a score of 0.00. This retriever is ideal when result accuracy is the top priority.

    2. **Naive Retriever** offered the best balance of quality and cost, achieving an efficiency score of 0.0. This retriever is recommended for production systems where both performance and cost must be optimized.

    3. **Naive Retriever** had the lowest overall latency at 0.00s, making it suitable for applications where response time is critical.

    4. **Naive Retriever** had the lowest cost per query at $0.00000, making it suitable for high-volume applications with tight budget constraints.

    ## Tradeoffs and Recommendations

    For this John Wick dataset, we recommend:
    - **For production use**: Naive Retriever
    - **For development/testing**: Naive Retriever
    - **For real-time applications**: Naive Retriever
    - **For high-stakes applications**: Naive Retriever
    