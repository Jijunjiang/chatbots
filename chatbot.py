

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForCausalLM

import argparse
import gradio as gr
import os

model_id = "deepseek-ai/deepseek-llm-7b-chat"  # You can use a smaller DeepSeek if limited on RAM/VRAM

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    load_in_8bit=True
)
#    offload_folder="/tmp/offload"  # or any writable folder
#)



def load_txt_documents(folder_path):
    docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                docs.append(f.read())
    return docs

def split_documents(docs, chunk_size=512, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.create_documents(docs)


def create_vector_store(chunks, persist_dir="./chroma_store"):
    # Retrival and inference are isolated steps, so use a different model here is fine.
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(chunks, embedder, persist_directory=persist_dir)
    vectordb.persist()
    return vectordb

def retrieve_context(vectordb, query, k=4):
    return vectordb.similarity_search(query, k=k)


def parse_args():
    parser = argparse.ArgumentParser(description="Run DeepSeek RAG Chatbot")
    parser.add_argument(
        "--data_dir", 
        type=str, 
        required=True,
        help="Path to the folder containing .txt files"
    )
    return parser.parse_args()

args = parse_args()
folder_path = args.data_dir

documents = load_txt_documents(folder_path)
chunks = split_documents(documents)
vectordb = create_vector_store(chunks)




def chat(message, history):

    # Retrive related context
    retrieved_docs = vectordb.similarity_search(message, k=5)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    print('history ' + str(history) + ' message ' + str(message))
    prompt = f"""Answer the question based on the context below.\n\nContext:\n{context}\n\n and you (Assistant) can user chat history: """

    for turn in history:
        if len(turn) == 2:
            prompt += f"User: {turn[0]}\nAssistant: {turn[1]}\n"
    prompt += f"User: {message}\nAssistant:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=True,
        top_p=0.98,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = response[len(prompt):].strip()
    return answer

gr.ChatInterface(chat).launch(share=False)