import streamlit as st
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts import PromptTemplate
from llama_index.core import Settings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core.node_parser import SentenceSplitter
import torch
from transformers import BitsAndBytesConfig
from huggingface_hub import login
login(token='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')

def main():
    torch.cuda.empty_cache()
    st.title("Question and Answering Assistant")

    # Embedding and LLM Setup
    def setup_models():
        embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
        llm = HuggingFaceLLM(
            model_name="meta-llama/Llama-2-7b-chat-hf",
            tokenizer_name="meta-llama/Llama-2-7b-chat-hf",
            query_wrapper_prompt=PromptTemplate("<s> [INST] {query_str} [/INST]"),
            context_window=3900,
            model_kwargs={'quantization_config': quantization_config},
            device_map="auto",
        )
        Settings.llm = llm
        Settings.embed_model = embed_model
        Settings.text_splitter = SentenceSplitter(chunk_size=1024)
        Settings.chunk_size = 1024
        Settings.chunk_overlap = 20
        Settings.transformations = [SentenceSplitter(chunk_size=1024)]
        return embed_model, llm

    # Document Processing
    def process_documents(files):
        documents_dir = "uploaded_documents"
        os.makedirs(documents_dir, exist_ok=True)
        for i, file in enumerate(files):
            with open(os.path.join(documents_dir, f"document_{i}.pdf"), "wb") as f:
                f.write(file.getbuffer())
        return SimpleDirectoryReader(documents_dir).load_data()

    # Streamlit UI
    uploaded_files = st.file_uploader("Upload one or more PDF documents", accept_multiple_files=True)
    if uploaded_files:
        documents = process_documents(uploaded_files)
        st.write("Files uploaded successfully!")
        embed_model, llm = setup_models()
        index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
        query_engine = index.as_query_engine(llm=llm)
        user_input = st.text_input("Enter your question:")
        if user_input:
            response = query_engine.query(user_input)
            st.text_area("Answer", value=str(response), height=300)
    if torch.cuda.is_available():
        st.write(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9} GB")
        st.write(f"Current Memory Allocated: {torch.cuda.memory_allocated() / 1e6} MB")
        st.write(f"Current Memory Cached: {torch.cuda.memory_reserved() / 1e6} MB")
if __name__ == "__main__":
    main()
