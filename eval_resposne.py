
from trulens_eval import Tru
from trulens_eval.feedback import Groundedness
import nest_asyncio
import numpy as np
from trulens_eval import (
    Feedback,
    TruLlama,
    OpenAI
)
import time

def define_eval_parameters(llm=None):
    if llm is None:
        provider = OpenAI()
    else:
        provider = llm

    qa_relevance = (
        Feedback(provider.relevance_with_cot_reasons, name="Answer Relevance")
        .on_input_output()
    )

    qs_relevance = (
        Feedback(provider.relevance_with_cot_reasons, name = "Context Relevance")
        .on_input()
        .on(TruLlama.select_source_nodes().node.text)
        .aggregate(np.mean)
    )

    #grounded = Groundedness(groundedness_provider=openai, summarize_provider=openai)
    grounded = Groundedness(groundedness_provider=provider)

    groundedness = (
        Feedback(grounded.groundedness_measure_with_cot_reasons, name="Groundedness")
            .on(TruLlama.select_source_nodes().node.text)
            .on_output()
            .aggregate(grounded.grounded_statements_aggregator)
    )

    feedbacks = [qa_relevance, qs_relevance, groundedness]

    return feedbacks


def eval_response(user_query, query_engine, app_id = 'app'):
    start_time = time.time()
    
    Tru().reset_database()

    feedbacks = define_eval_parameters()

    tru_recorder = TruLlama(
        query_engine,
        app_id=app_id,
        feedbacks=feedbacks
        )
    
    with tru_recorder as recording:
        response = query_engine.query(user_query)
    
    print(Tru().get_leaderboard(app_ids=[]))
    records, feedback = Tru().get_records_and_feedback(app_ids=[])

    print(records[["input", "output"] + feedback])
    end_time = time.time()

    elapsed_time = end_time - start_time

    return response, elapsed_time, records

