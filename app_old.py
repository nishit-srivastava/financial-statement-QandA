
import faiss
import pickle
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration
)
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi
import numpy as np
import re
import numpy as np
from sklearn.preprocessing import minmax_scale
import faiss
from nltk.tokenize import word_tokenize
import re
import numpy as np
from sklearn.preprocessing import minmax_scale
import faiss
from nltk.tokenize import word_tokenize
# load fasiss
faiss_index = faiss.read_index("faiss_index.idx")

# load pickel 
with open("bm25_index.pkl", "rb") as f:
    bm25 = pickle.load(f)


with open("chunks.pkl", "rb") as f:
    all_chunks = pickle.load(f)



pretrain_tokenizer = T5Tokenizer.from_pretrained("t5-small")
pretrain_model = T5ForConditionalGeneration.from_pretrained("t5-small")



model_path = "./t5-small-qa-model"

# Load model + tokenizer

#finetune_model = T5ForConditionalGeneration.from_pretrained(model_path)
#finetune_tokenizer = T5Tokenizer.from_pretrained(model_path)
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

finetune_tokenizer = AutoTokenizer.from_pretrained("gauravvivek8/finetune-t5-small-financial-statement")
finetune_model = AutoModelForSeq2SeqLM.from_pretrained("gauravvivek8/finetune-t5-small-financial-statement")
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")



def hybrid_search(query: str, top_n: int = 5, alpha: float = 0.5, use_rrf=False):
    """
    Hybrid retrieval combining FAISS (dense) and BM25 (sparse).
    alpha: weight for dense retrieval (0 to 1) if not using RRF.
    """
    # Preprocess query
    query_clean = re.sub(r"[^a-zA-Z0-9\s]", "", query.lower())
    
    # Dense retrieval (cosine similarity if normalized)
    query_embedding = embedding_model.encode([query_clean], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)
    dense_scores, dense_indices = faiss_index.search(query_embedding, top_n)
    dense_results = [(int(idx), float(score)) for idx, score in zip(dense_indices[0], dense_scores[0])]
    
    # Sparse retrieval
    sparse_scores = bm25.get_scores(word_tokenize(query_clean))
    sparse_indices = np.argsort(sparse_scores)[::-1][:top_n]
    sparse_results = [(int(idx), float(sparse_scores[idx])) for idx in sparse_indices]
    
    if use_rrf:
        # Reciprocal Rank Fusion
        fused_scores = reciprocal_rank_fusion(dense_results, sparse_results)
    else:
        # Normalize both score sets
        dense_dict = {idx: s for idx, s in dense_results}
        sparse_dict = {idx: s for idx, s in sparse_results}
        dense_norm = minmax_scale(list(dense_dict.values())) if dense_dict else []
        sparse_norm = minmax_scale(list(sparse_dict.values())) if sparse_dict else []
        
        for i, (idx, _) in enumerate(dense_results):
            dense_dict[idx] = dense_norm[i]
        for i, (idx, _) in enumerate(sparse_results):
            sparse_dict[idx] = sparse_norm[i]
        
        # Weighted fusion
        scores_dict = {}
        for idx, score in dense_dict.items():
            scores_dict[idx] = scores_dict.get(idx, 0) + alpha * score
        for idx, score in sparse_dict.items():
            scores_dict[idx] = scores_dict.get(idx, 0) + (1 - alpha) * score
        
        fused_scores = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    return [(all_chunks[idx], score) for idx, score in fused_scores]

def count_tokens(text,tokenizer):
    #print(text)
    return len(tokenizer(text, return_tensors="pt"))

# Build prompt safely within context window
def build_messages(context, question,tokenizer, max_ctx=4096, reserve=512,):
    """
    max_ctx: model's context siz
    reserve: keep space for system prompt, question, and answer
    """
    system_msg = "Answer STRICTLY and CONCISELY using ONLY the provided context.Provide  answer  only. If the answer is not present, reply exactly: Not found in context."
    
    system_tokens = count_tokens(system_msg,tokenizer)
    question_tokens = count_tokens(question,tokenizer)

    # budget for context
    budget = max_ctx - (system_tokens + question_tokens + reserve)
    context_tokens = tokenizer(context,return_tensors="pt")

    if len(context_tokens) > budget:
        context_tokens = context_tokens[:budget]  # truncate
        context = tokenizer.decode(context_tokens)

    input_text = f"question: {question} context: {context}"
    return input_text

# Example usage

def generate_ans(question ,searchtype):
    results=hybrid_search(question, top_n=5, alpha=0.6)
    context=""
    for text, score in results:
        context+= "\n " +text.lower()
    #print(context)
    input_text = f"question: {question} context: {context}"
    tokenizer=finetune_tokenizer
    model=finetune_model
    if searchtype=='rag':
        tokenizer=pretrain_tokenizer
        model=pretrain_model



    
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_length=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    while True:
        question = input("\nEnter your question (or type 'exit' to quit): ")
        if question.lower() == "exit":
            print("Goodbye!")
            break

        q_type = input("Enter type (rag/finetune): ").strip().lower()

        if q_type == "rag" or  q_type == "finetune":
            print(generate_ans(question,q_type))
        else:
            print("❌ Invalid type. Please enter 'rag' or 'finetune'.")
            continue


if __name__ == "__main__":
    main()
