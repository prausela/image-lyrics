import os
from PIL import Image
import requests
from transformers import BlipProcessor, BlipForConditionalGeneration

def generate_caption(image_name):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

    # Path to the image in the images directory
    img_path = os.path.join('docs/images', image_name)
    raw_image = Image.open(img_path).convert('RGB')

    # unconditional image captioning
    inputs = processor(raw_image, return_tensors="pt")

    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ------------- Retrieval-Augmented Generation  ------------- #

def get_docs():
    """
    Loads each file into one document, like knowledge base
    :return: docs
    """

    loader = DirectoryLoader("docs/lyrics", "*.txt", loader_cls=TextLoader)  # Reads custom data from local files

    docs = loader.load()
    return docs

def split_text(docs, config: dict):
    """
    Get chunks from docs. Our loaded doc may be too long for most models, and even if it fits is can struggle to find relevant context. So we generate chunks
    :param docs: docs to be split
    :return: chunks
    """

    text_splitter = RecursiveCharacterTextSplitter( # recommended splitter for generic text
        chunk_size=config["splitter"]["chunk_size"],
        chunk_overlap=config["splitter"]["chunk_overlap"],
        add_start_index=True
    )
    chunks = text_splitter.split_documents(docs)

    return chunks

def get_data_store(chunks, batch_size=1000):
    """
    Store chunks into a db. ChromaDB uses vector embeddings as the key, creates a new DB from the documents
    :param docs:
    :param chunks:
    :return: database
    """
    embeddings = HuggingFaceEmbeddings( #  embedding=OpenAIEmbeddings() rate limit
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'} # TODO gpu
    )

    db = None
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        if db is None:
            db = Chroma.from_documents(documents=batch, embedding=embeddings)
        else:
            db.add_documents(batch)

    return db

from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA

def generate_response(db, prompt, config: dict):
    """
    Generate a response with a LLM based on previous custom context
    :return: chatbot response
    """

    hf_llm = HuggingFaceHub(
        repo_id="HuggingFaceH4/zephyr-7b-beta",  # Model id
        task="text-generation",                  # Specific task the model is intended to perform
        model_kwargs={
            "max_new_tokens": config["llm"]["max_new_tokens"], # The maximum number of tokens to generate in the response.  Limits the length of the generated text to ensure responses are concise or fit within certain constraints.
            "top_k": config["llm"]["top_k"],                   # Limits the sampling pool to the top k tokens, increasing focus on more likely tokens
            "temperature": config["llm"]["temperature"],       # Controls the randomness of predictions, with lower values making the output more deterministic. : Produces more focused and less random text by making the model more confident in its choices.
            "repetition_penalty": config["llm"]["repetition_penalty"],                       # Penalizes repeated tokens to avoid repetitive output.  Discourages the model from repeating the same token sequences, resulting in more varied and natural text.
        },
    )

    chain = RetrievalQA.from_chain_type( # Generate chat model based on previous llm
        llm=hf_llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_type=config["retriever"]["search_type"], search_kwargs={"k": config["retriever"]["k"]}),
        verbose=False
    )

    response = chain.run(prompt)
    return response

def postprocess_response(response, n_retrieved_chunks = None):
    retrived_chunks = None
    if n_retrieved_chunks is not None:
        retrived_chunks = get_retrieved_chunks(response, n_retrieved_chunks)
    answer_start = response.find("Helpful Answer:")
    if answer_start != -1:
        answer = response[answer_start + len("Helpful Answer:"):].strip()
    else:
        answer = response.strip()

    return answer, retrived_chunks

def query_rewriting(response):
    answer_start = response.find("araf")
    if answer_start != -1:
        answer = response[answer_start + len("araf"):].strip()
        word_end = answer.find(" ")
        if word_end != -1:
            answer = answer[word_end + len(" "):].strip()
    else:
        answer = response.strip()

    return answer

def get_retrieved_chunks(response, retriever_k):
    chosen_chunks = response.split('\n\n')
    return chosen_chunks[1:retriever_k]

def predict(inputs: list[str], db, config: dict) -> tuple[str, list[str], str, str]:
    caps = ""
    for user_input in inputs:
        cap = generate_caption(user_input)
        caps += query_rewriting(cap) + " "

    caps = caps[:-1]
    prompt = f"Write a song lyrics about \"{caps}\" having intro, verse 1, chorus 1, verse 2, chorus 2, bridge, chorus 3, outro. Ignore the words intro, verse, chorus and outro."
    response = generate_response(db, prompt, config)
    answer, retrieved_chunks = postprocess_response(response, config["retriever"]["k"])
    return answer, retrieved_chunks, prompt, caps