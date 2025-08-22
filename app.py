import re
import pickle
import numpy as np
import faiss
import torch
import streamlit as st
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import minmax_scale
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import base64


# ------------------ LOAD RESOURCES ------------------ #
@st.cache_resource
def load_indexes():
    """Load FAISS, BM25, and chunks from disk (cached for Streamlit)."""
    faiss_index = faiss.read_index("faiss_index.idx")
    with open("bm25_index.pkl", "rb") as f:
        bm25 = pickle.load(f)
    with open("chunks.pkl", "rb") as f:
        all_chunks = pickle.load(f)
    return faiss_index, bm25, all_chunks


@st.cache_resource
def load_models():
    """Load pretrained + finetuned models and embeddings (cached for Streamlit)."""
    pretrain_tokenizer = AutoTokenizer.from_pretrained("t5-small", legacy=False)
    pretrain_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

    finetune_tokenizer = AutoTokenizer.from_pretrained(
        "gauravvivek8/finetune-t5-small-financial-statement", legacy=False
    )
    finetune_model = AutoModelForSeq2SeqLM.from_pretrained(
        "gauravvivek8/finetune-t5-small-financial-statement"
    )

    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    return pretrain_tokenizer, pretrain_model, finetune_tokenizer, finetune_model, embedding_model


# Load everything once
faiss_index, bm25, all_chunks = load_indexes()
pretrain_tokenizer, pretrain_model, finetune_tokenizer, finetune_model, embedding_model = load_models()


# ------------------ HYBRID SEARCH ------------------ #
def hybrid_search(query: str, top_n: int = 5, alpha: float = 0.5, use_rrf=False):
    """
    Hybrid retrieval combining FAISS (dense) and BM25 (sparse).
    alpha: weight for dense retrieval (0 to 1) if not using RRF.
    """
    query_clean = re.sub(r"[^a-zA-Z0-9\s]", "", query.lower())

    # Dense retrieval
    query_embedding = embedding_model.encode([query_clean], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)
    dense_scores, dense_indices = faiss_index.search(query_embedding, top_n)
    dense_results = [(int(idx), float(score)) for idx, score in zip(dense_indices[0], dense_scores[0])]

    # Sparse retrieval
    sparse_scores = bm25.get_scores(word_tokenize(query_clean))
    sparse_indices = np.argsort(sparse_scores)[::-1][:top_n]
    sparse_results = [(int(idx), float(sparse_scores[idx])) for idx in sparse_indices]

    if use_rrf:
        fused_scores = dense_results + sparse_results
    else:
        dense_dict = {idx: s for idx, s in dense_results}
        sparse_dict = {idx: s for idx, s in sparse_results}
        dense_norm = minmax_scale(list(dense_dict.values())) if dense_dict else []
        sparse_norm = minmax_scale(list(sparse_dict.values())) if sparse_dict else []

        for i, (idx, _) in enumerate(dense_results):
            dense_dict[idx] = dense_norm[i]
        for i, (idx, _) in enumerate(sparse_results):
            sparse_dict[idx] = sparse_norm[i]

        scores_dict = {}
        for idx, score in dense_dict.items():
            scores_dict[idx] = scores_dict.get(idx, 0) + alpha * score
        for idx, score in sparse_dict.items():
            scores_dict[idx] = scores_dict.get(idx, 0) + (1 - alpha) * score

        fused_scores = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]

    return [(all_chunks[idx], score) for idx, score in fused_scores]


# ------------------ GENERATION ------------------ #
def generate_ans(question, searchtype):
    results = hybrid_search(question, top_n=5, alpha=0.6)
    context = "\n".join(text.lower() for text, _ in results)

    input_text = f"question: {question} context: {context}"
    tokenizer = finetune_tokenizer
    model = finetune_model
    if searchtype == "rag":
        tokenizer = pretrain_tokenizer
        model = pretrain_model

    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_length=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# ------------------ PDF VIEWER ------------------ #
def show_pdf(file_path, width=350, height=500):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="{width}" height="{height}" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


# ------------------ STREAMLIT MAIN ------------------ #
def streamlit_main():
    st.set_page_config(page_title="QnA App", page_icon="🤖", layout="centered")

    st.title("🤖 Question Answering App")
    st.write("Ask any question and choose a method (RAG / Finetune).")

    question = st.text_area("❓ Enter your question here:", height=100)
    q_type = st.radio("⚙️ Choose type:", ["rag", "finetune"])

    if st.button("Get Answer"):
        if question.strip():
            with st.spinner("Generating answer... 🧠"):
                answer = generate_ans(question, q_type)
            st.success(answer)
        else:
            st.warning("Please enter a question.")

    st.subheader("📄 Training Documents")

    # col1, col2 = st.columns(2)
    # with col1:
    #     st.write("**TCS 2023-24**")
    #     show_pdf("TCS_2023-24.pdf", width=350, height=500)

    # with col2:
    #     st.write("**TCS 2024-25**")
    #     show_pdf("TCS_2024-25.pdf", width=350, height=500)

    col1= st.columns(1)[0]

    with col1:
        st.write("**TCS 2024-25**")
        show_pdf("TCS_2024-25.pdf", width=700, height=600)



# ------------------ CLI MAIN ------------------ #
def cli_main():
    print("Welcome to QnA CLI! Type 'exit' to quit.\n")
    while True:
        question = input("\nEnter your question (or type 'exit' to quit): ")
        if question.lower() == "exit":
            print("Goodbye!")
            break

        q_type = input("Enter type (rag/finetune): ").strip().lower()
        if q_type in ["rag", "finetune"]:
            print("🧠", generate_ans(question, q_type))
        else:
            print("❌ Invalid type. Please enter 'rag' or 'finetune'.\n")


# ------------------ ENTRY POINT ------------------ #
if __name__ == "__main__":
    try:
        import streamlit.runtime
        streamlit_main()
    except ImportError:
        cli_main()
