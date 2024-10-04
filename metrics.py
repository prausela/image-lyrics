from langchain.embeddings import HuggingFaceEmbeddings
from numpy import dot
from numpy.linalg import norm
from langchain_community.llms import HuggingFaceHub

def calculate_faithfulness(answer, retrieved_chunks):
    embeddings = HuggingFaceEmbeddings( 
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'} 
    )
    
    sentences = [sentence for sentence in answer.split('\n') if sentence != ""]
    claims = [0] * len(sentences)

    for i in range(len(retrieved_chunks)):
        chunk_sentences = [sentence for sentence in retrieved_chunks[i].split('\n') if sentence != ""]
        for chunk_sentence in chunk_sentences:
            for i in range(len(sentences)):
                embedding_vector_sentence = embeddings.embed_query(sentences[i]) 
                embeddings_vector_chunk = embeddings.embed_query(chunk_sentence)
                cos_sim = dot(embedding_vector_sentence, embeddings_vector_chunk)/(norm(embedding_vector_sentence)*norm(embeddings_vector_chunk))
                if cos_sim > 0.4:
                    claims[i] = 1

    faithfulness = sum(claims) / len(sentences)
    return faithfulness

from rag import postprocess_response

def calculate_answer_relevance(answer, caps):
    embeddings = HuggingFaceEmbeddings( 
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'} 
    )
    
    hf_llm = HuggingFaceHub(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation",
        model_kwargs={
            "max_new_tokens": 512,
            "top_k": 30,
            "temperature": 0.3,
            "repetition_penalty": 1.2,
        },
    )

    question_1 = hf_llm.predict(f"Given the following context\n\n{answer}\n\nQuestion: What inspired these lyrics?\nHelpful Answer:")
    question_2 = hf_llm.predict(f"Given the following context\n\n{answer}\n\nQuestion: What are its main themes?\nHelpful Answer:")
    question_3 = hf_llm.predict(f"Given the following context\n\n{answer}\n\nQuestion: What does it evoke?\nHelpful Answer:")
    question_4 = hf_llm.predict(f"Given the following context\n\n{answer}\n\nQuestion: What elements are present?\nHelpful Answer:")

    question_1, _ = postprocess_response(question_1)
    question_2, _ = postprocess_response(question_2)
    question_3, _ = postprocess_response(question_3)
    question_4, _ = postprocess_response(question_4)

    caps_embedding = embeddings.embed_query(caps) 
    question_1_embedding = embeddings.embed_query(question_1) 
    question_2_embedding = embeddings.embed_query(question_2)
    question_3_embedding = embeddings.embed_query(question_3)
    question_4_embedding = embeddings.embed_query(question_4)

    cos_sim_1 = dot(caps_embedding, question_1_embedding)/(norm(caps_embedding)*norm(question_1_embedding))
    cos_sim_2 = dot(caps_embedding, question_2_embedding)/(norm(caps_embedding)*norm(question_2_embedding))
    cos_sim_3 = dot(caps_embedding, question_3_embedding)/(norm(caps_embedding)*norm(question_3_embedding))
    cos_sim_4 = dot(caps_embedding, question_4_embedding)/(norm(caps_embedding)*norm(question_4_embedding))

    answer_relevance = (cos_sim_1 + cos_sim_2 + cos_sim_3 + cos_sim_4) / 4
    return answer_relevance
    