 
import streamlit as st
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
import re
from sentence_transformers import CrossEncoder

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline


st.title("📚 Simple and Smart PDF RAG Application")


# Load models
@st.cache_resource
def load_models():

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )
    reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

    generate = pipeline(
        "text-generation",
        model="HuggingFaceTB/SmolLM2-360M-Instruct",
        device=-1
    )

    return embedding_model, reranker, generate


embedding_model, reranker, generate = load_models()


# Upload PDF
uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


if uploaded_file is not None:

    # -----------------------------
    # Extract text
    # -----------------------------
    reader = PdfReader(uploaded_file)
    #reader = PdfReader(uploaded_file)
    pages = []
    
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()

        if page_text:
            pages.append({
            "text": page_text,
            "page": page_number
             })

    # -----------------------------
    # Chunking
    # -----------------------------
    
    document = []
    chunk_size = 500
    temp_chunk = ""

    for page in pages:
        page_text = page["text"]
        page_number=page["page"]
        sentences = re.split(
        r'(?<=[.!?])\s+',
        page_text)
   
        for sentence in sentences:
            if len(sentence) + len(temp_chunk) <= chunk_size:

               temp_chunk += sentence + " "

            else:

                 if temp_chunk:
                  
                    document.append({
                    "text":temp_chunk.strip(),  
                    "page": page_number})

                    temp_chunk = sentence + " "


        if temp_chunk:
           document.append({
                  "text":temp_chunk.strip(),
                     "page": page_number
           })
           temp_chunk=""


    st.success(
        f"Created {len(document)} chunks"
    )

    st.write("Number of pages:", len(pages))
    st.write("Number of chunks:", len(document))

 
    # -----------------------------
# Create BM25 keyword index
# -----------------------------

    tokenized_documents = [
    re.findall(r"\b\w+\b", doc["text"].lower())
    for doc in document
    ]

    if not document:
         st.warning("No extractable text found in this PDF.")
         st.stop()

    bm25 = BM25Okapi(tokenized_documents)

 
    # -----------------------------
    # Create embeddings
    # -----------------------------

    document_emd = embedding_model.encode(
     [doc["text"] for doc in document]
        
    )

    document_emd = np.array(
        document_emd
    ).astype("float32")


    # -----------------------------
    # Normalize
    # -----------------------------

    faiss.normalize_L2(
        document_emd
    )


    # -----------------------------
    # FAISS
    # -----------------------------

    index = faiss.IndexFlatL2(
        document_emd.shape[1]
    )

    index.add(document_emd)


    # -----------------------------
    # Question
    # -----------------------------

    question = st.text_input(
        "Ask a question about the PDF:"
    )


    if question:

        # Question embedding

        question_emd = embedding_model.encode(
            question
        )

        question_emd = np.array(
            question_emd
        ).astype("float32")

        question_emd = question_emd.reshape(
            1, -1
        )

        faiss.normalize_L2(
            question_emd
        )


        

        
        # -----------------------------
        # Hybrid Search: FAISS + BM25
        # -----------------------------

        k = min(10, len(document))

        # 1. Dense retrieval using FAISS
        distances, dense_indices = index.search(
             question_emd,
             k
           )

        # 2. Keyword retrieval using BM25
        query_tokens = re.findall(
          r"\b\w+\b",
         question.lower()
           )

        bm25_scores = bm25.get_scores(query_tokens)

        keyword_indices = np.argsort(bm25_scores)[::-1][:k]

        # 3. Combine rankings using Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        rrf_constant = 60

        for rank, idx in enumerate(dense_indices[0], start=1):
             idx = int(idx)
             rrf_scores[idx] = rrf_scores.get(idx, 0) + (
                 1 / (rrf_constant + rank)
             )

        for rank, idx in enumerate(keyword_indices, start=1):
             idx = int(idx)
             rrf_scores[idx] = rrf_scores.get(idx, 0) + (
               1 / (rrf_constant + rank)
             )

        # 4. Select the combined candidate set
        candidate_indices = sorted(
        rrf_scores,
        key=rrf_scores.get,
        reverse=True
        )[:10]

        candidate_docs = [
        {
        "text": document[i]["text"],
        "page": document[i]["page"],
        "chunk_id": i
        }
        for i in candidate_indices
         ]

       # 5. Rerank candidates using CrossEncoder
        pairs = [
        [question, doc["text"]]
        for doc in candidate_docs
         ]

        rerank_scores = reranker.predict(pairs)

        ranked_docs = sorted(
        zip(rerank_scores, candidate_docs),
        key=lambda x: float(x[0]),
        reverse=True
         )

        # 6. Select the top three chunks
        top_docs = [
        doc for score, doc in ranked_docs[:3]
         ]

    # Debug information
        st.write("FAISS distances:", distances[0])
        st.write("BM25 top scores:", bm25_scores[keyword_indices])
        st.write("Reranker scores:", [
        float(score) for score, doc in ranked_docs
        ])

        st.write(
        "Retrieved chunk IDs:",
          [doc["chunk_id"] for doc in top_docs]
          )
 
         
        # -----------------------------
        # Retrieve documents
        # -----------------------------

      
        ret_doc = [
        doc["text"]
        for doc in top_docs
          ]

        content = "\n\n".join(ret_doc)

        retrieved_pages = sorted(
        set(doc["page"] for doc in top_docs)
        )

        st.write("Source pages:", retrieved_pages)

        with st.expander("Debug: Retrieved context"):
            st.write(content)
     
        # -----------------------------
        # Prompt
        # -----------------------------

        prompt = f"""Answer the question using the document context.

        Instructions:
          - Include the important details that directly answer the question.
          - Use only information supported by the context.
          - Explain the answer clearly in simple language.
          - Do not repeat the question or copy unrelated text.
          - If the context does not contain enough information, say:
          "The information is not available in the provided document."

          Document context:
          {content}

          Question: {question}

          Answer:"""

      
     
        # -----------------------------
        # Generate answer
        # -----------------------------

        with st.spinner("Generating answer..."):

            result = generate(
                prompt,
                max_new_tokens=200,
                do_sample=False,
             
                return_full_text=False
            )


        answer = result[0]["generated_text"].strip()


        # -----------------------------
        # Display
        # -----------------------------

        st.subheader("Answer")

        st.write(answer)


        # Show retrieved chunks

        with st.expander("View retrieved chunks"):

            for i, doc in enumerate(ret_doc):

                st.write(
                    f"### Chunk {i + 1}"
                )

                st.write(doc)
