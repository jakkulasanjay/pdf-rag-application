 
import streamlit as st
import numpy as np
import faiss
import re

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

    generate = pipeline(
        "text-generation",
        model="HuggingFaceTB/SmolLM2-360M-Instruct",
        device=-1
    )

    return embedding_model, generate


embedding_model, generate = load_models()


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

        sentences = re.split(
        r'(?<=[.!?])\s+',
        page_text)
   
        for sentence in sentences:
            if len(sentence) + len(temp_chunk) <= chunk_size:

               temp_chunk += sentence + " "

            else:

                 if temp_chunk:
                  
                    document.append(
                    temp_chunk.strip()   )

                    temp_chunk = sentence + " "


        if temp_chunk:
           document.append(
                    temp_chunk.strip()
                      )


    st.success(
        f"Created {len(document)} chunks"
    )

    st.write("Number of pages:", len(pages))
    st.write("Number of chunks:", len(document))
    # -----------------------------
    # Create embeddings
    # -----------------------------

    document_emd = embedding_model.encode(
        document
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
        # Search
        # -----------------------------

        k = min(5, len(document))

        distance, indices = index.search(
            question_emd,
            k
        )
        st.write("Distances:", distance[0])


        best_distance=distance[0][0]
        if best_distance > 1.0:
            st.warning("The information is not available in the provided document.")
            st.stop()
     
         
         
        # -----------------------------
        # Retrieve documents
        # -----------------------------

        ret_doc = [
            document[i]
            for i in indices[0]
        ]


        content = "\n\n".join(
            ret_doc
        )


        # -----------------------------
        # Prompt
        # -----------------------------

        prompt = f"""
        You are a strict document question-answering assistant.

         Your task is to answer the question ONLY from the CONTEXT provided below.

         IMPORTANT RULES:
         1. The CONTEXT is your only source of information.
         2. Do NOT use your own knowledge or information from outside the CONTEXT.
         3. Do NOT guess, infer, assume, or complete missing information.
         4. If the answer is not explicitly supported by the CONTEXT, respond exactly:
            "The information is not available in the provided document."
         5. If the question is completely unrelated to the CONTEXT, respond exactly:
            "The information is not available in the provided document."
         6. Even if you know the answer from your general knowledge, DO NOT answer it.
         7. Keep the answer clear and concise.
         8. Answer only what the user asked.

         CONTEXT:
         {content}

         QUESTION:
         {question}

         ANSWER:
         """
        # -----------------------------
        # Generate answer
        # -----------------------------

        with st.spinner("Generating answer..."):

            result = generate(
                prompt,
                max_new_tokens=100,
                return_full_text=False
            )


        answer = result[0]["generated_text"]


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
