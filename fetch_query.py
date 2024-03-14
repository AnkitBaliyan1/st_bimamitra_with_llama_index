import os
os.environ["HUGGINGFACE_API_KEY"] = "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"
os.environ["OPENAI_API_KEY"] = "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"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from llama_index.core import (
    ServiceContext,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.node_parser import get_leaf_nodes
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.indices.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine


def get_automerging_index(
    # documents,
    llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    save_dir="merging_index",
    chunk_sizes=None,
):
    # chunk_sizes = chunk_sizes or [2048, 512, 128]
    # node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)
    # nodes = node_parser.get_nodes_from_documents(documents)
    # leaf_nodes = get_leaf_nodes(nodes)
    merging_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
    )
    # storage_context = StorageContext.from_defaults()
    # storage_context.docstore.add_documents(nodes)

    if not os.path.exists(save_dir):
        pass
        # automerging_index = VectorStoreIndex(
        #     leaf_nodes, storage_context=storage_context, service_context=merging_context
        # )
        # automerging_index.storage_context.persist(persist_dir=save_dir)
    else:
        pass
    
    automerging_index = load_index_from_storage(
        StorageContext.from_defaults(persist_dir=save_dir),
        service_context=merging_context,
    )
    return automerging_index


def get_automerging_query_engine(
    automerging_index,
    similarity_top_k=12,
    rerank_top_n=6,
):
    base_retriever = automerging_index.as_retriever(similarity_top_k=similarity_top_k)
    retriever = AutoMergingRetriever(
        base_retriever, automerging_index.storage_context, verbose=True
    )
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="BAAI/bge-reranker-base"
    )
    auto_merging_engine = RetrieverQueryEngine.from_args(
        retriever, node_postprocessors=[rerank]
    )
    return auto_merging_engine


from llama_index.llms.openai import OpenAI
import time

def prepare_engine():
    index = get_automerging_index(
        # [document],
        llm=OpenAI(model="gpt-3.5-turbo", temperature=0.1),
        # save_dir="./st_buidling_with_llama/auto_merging_retrieval/merging_index",
        save_dir = './merging_index'
        
    )
    query_engine = get_automerging_query_engine(index, similarity_top_k=12)

    return query_engine



def fetch_query_main(query_engine, user_query = "What is health insurance?"):

    # user_query = "What is health insurance?"
    start_time = time.time()
    response = query_engine.query(user_query)
    # response = eval_response(user_query, query_engine, app_id = 'app')
    end_time = time.time()

    elapsed_time = end_time - start_time
    answer = response.response

    print(f"Time taken to fetch response: {elapsed_time} seconds")
    return answer, elapsed_time
    


if __name__ == "__main__":
    fetch_query_main()