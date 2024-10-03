from langchain.embeddings import HuggingFaceEmbeddings
from numpy import dot
from numpy.linalg import norm

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