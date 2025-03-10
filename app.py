import streamlit as st
from PyPDF2 import PdfReader
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from langchain.text_splitter import RecursiveCharacterTextSplitter
import boto3
import os
import json
from io import BytesIO
import pickle


# -------------------------------------------------------------------
# 1. Student Credentials & Authentication
# -------------------------------------------------------------------
student_credentials = {
    "alice":    {"password": "secret1", "level": "beginner"},
    "bob":      {"password": "secret2", "level": "intermediate"},
    "charlie":  {"password": "secret3", "level": "advanced"},
}

s3 = boto3.client('s3')
bucket_name = "edurag"  

# Function to upload the FAISS index file to S3
def upload_to_s3(local_file_path, s3_file_name):
    with open(local_file_path, 'rb') as data:
        s3.upload_fileobj(data, bucket_name, s3_file_name)

# Function to download the FAISS index file from S3
def download_from_s3(s3_file_name, local_file_path):
    with open(local_file_path, 'wb') as data:
        s3.download_fileobj(bucket_name, s3_file_name, data)

# Function to list files in the S3 bucket
def list_files_in_s3():
    response = s3.list_objects_v2(Bucket=bucket_name)
    files = []
    if 'Contents' in response:
        files = []
        for item in response['Contents']:
            if item['Key'][-6:]==".index":
                files.append(item['Key'])
        print(files)
    return files


def authenticate_user(username, password):
    """Returns the student's level if credentials match, else None."""
    user = student_credentials.get(username)
    if user and user["password"] == password:
        return user["level"]
    return None

# -------------------------------------------------------------------
# 2. TeachingAssistant Class
# -------------------------------------------------------------------
class TeachingAssistant:
    def __init__(self, text_chunks="", groq_api_key="",pdf_name="",index=None,embedding_model=None):
        self.text_chunks = text_chunks
        self.index = index
        self.embedding_model = embedding_model
        self.bm25 = None
        self.GROQ_API_KEY = groq_api_key
        self.pdf_name = pdf_name

    def build_indexes(self, model_name="all-MiniLM-L6-v2"):
        """Build FAISS and BM25 indexes from text_chunks."""
        # 1) Load/initialize sentence transformer
        self.embedding_model = SentenceTransformer(model_name)

        # 2) Create embeddings & build FAISS index
        embeddings = np.array([self.embedding_model.encode(chunk) for chunk in self.text_chunks])
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

        # 3) Build BM25 index
        tokenized_corpus = [chunk.split() for chunk in self.text_chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # 4) Save index to file and upload to S3
        faiss_index_file = f"{self.pdf_name}_faiss.index"
        faiss.write_index(self.index, faiss_index_file)  # Save index locally
        upload_to_s3(faiss_index_file, faiss_index_file)  # Upload index to S3
        os.remove(faiss_index_file)  # Clean up local index file

        # 5) Save BM25 index to a file and upload to S3
        bm25_index_file = f"{self.pdf_name}_bm25.pkl"
        with open(bm25_index_file, 'wb') as f:
            pickle.dump(self.bm25, f)  # Save BM25 index using pickle
        upload_to_s3(bm25_index_file,bm25_index_file)  # Upload BM25 index to S3
        os.remove(bm25_index_file)  # Clean up local file

        # 4) Save text_chunks to a file and upload to S3
        text_chunks_file = f"{self.pdf_name}_text_chunks.json"
        with open(text_chunks_file, 'w') as f:
            json.dump(self.text_chunks, f)  # Save text_chunks as JSON
        upload_to_s3(text_chunks_file, text_chunks_file)  # Upload text_chunks to S3
        os.remove(text_chunks_file)  # Clean up local file
    
    def load_index_from_s3(self):
        """Load FAISS index from S3."""
        faiss_index_file = f"{self.pdf_name}_faiss.index"
        print(faiss_index_file)
        download_from_s3(faiss_index_file, faiss_index_file)  # Download index from S3
        self.index = faiss.read_index(faiss_index_file)  # Load FAISS index
        print(self.index)
        os.remove(faiss_index_file)  # Clean up local index file

        # Download BM25 index from S3
        bm25_index_file = f"{self.pdf_name}_bm25.pkl"
        download_from_s3(bm25_index_file,bm25_index_file)
        with open(bm25_index_file, 'rb') as f:
            self.bm25 = pickle.load(f)
        os.remove(bm25_index_file)  # Clean up local file
        
        # Download text_chunks from S3
        text_chunks_file = f"{self.pdf_name}_text_chunks.json"
        download_from_s3(text_chunks_file, text_chunks_file)
        with open(text_chunks_file, 'r') as f:
            self.text_chunks = json.load(f)  # Load text_chunks
        os.remove(text_chunks_file)  # Clean up local file

    def retrieve_course_materials(self, query, top_k=5):
        """Hybrid retrieval using FAISS + BM25."""
        if not self.index or not self.embedding_model:
            return []

        # --- FAISS Retrieval ---
        query_vector = np.array(self.embedding_model.encode(query)).reshape(1, -1)
        distances, indices = self.index.search(query_vector, top_k)
        faiss_results = [
            (self.text_chunks[i], 1 / (distances[0][idx] + 1e-5))
            for idx, i in enumerate(indices[0])
            if i < len(self.text_chunks)
        ]

        # --- BM25 Retrieval ---
        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_k_bm25_indices = np.argsort(bm25_scores)[::-1][:top_k]
        bm25_results = [(self.text_chunks[i], bm25_scores[i]) for i in top_k_bm25_indices]

        # --- Combine & Sort ---
        combined_results = faiss_results + bm25_results
        combined_results = sorted(combined_results, key=lambda x: x[1], reverse=True)

        final = []
        seen = set()
        for chunk, score in combined_results:
            if chunk not in seen:
                final.append(chunk)
                seen.add(chunk)
            if len(final) == top_k:
                break
        return final

    def generate_response_with_groq(self, prompt):
        """Call Groq's API for an LLM response (optional)."""
        if not self.GROQ_API_KEY:
            return "Groq API key not provided. Cannot generate response."

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.GROQ_API_KEY}",
            "Content-Type": "application/json",
        }
        data = {
            "model": "llama3-70b-8192",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1000,
        }
        try:
            response = requests.post(url, headers=headers, json=data)
            response_json = response.json()
            if "choices" in response_json:
                return response_json["choices"][0]["message"]["content"]
            return f"Error: 'choices' not found.\nFull response: {response_json}"
        except Exception as e:
            return f"Error calling Groq: {str(e)}"

    def user_input(self, user_question, student_level="beginner"):
        """
        Retrieves docs, creates a prompt with level-specific instructions,
        and calls the LLM for a final answer.
        """

        # 1) Retrieve relevant course materials
        retrieved_docs = self.retrieve_course_materials(user_question)
        if not retrieved_docs:
            return {"response": "No relevant info found or PDFs not processed yet."}

        context = "\n".join(retrieved_docs)

        # 2) Add instructions based on level
        if student_level == "beginner":
            level_instructions = (
                "Explain thoroughly, define all terms, keep it simple, and give step-by-step guidance."
            )
        elif student_level == "intermediate":
            level_instructions = (
                "Give a balanced explanation, assume familiarity with common terms, but add detail when necessary."
            )
        else:  # "advanced"
            level_instructions = (
                "Be concise, focus on advanced details, and assume knowledge of fundamentals."
            )

        # 3) Build the prompt
        prompt = f"""
        You are an AI Teaching Assistant. Answer the user's question strictly from the course material below.
        If you do not find the answer in the course material, say so.

        Student level: {student_level}
        Level Instructions: {level_instructions}

        Course Material:
        {context}

        Question: {user_question}

        Answer:
        """

        # 4) Generate the final answer
        answer = self.generate_response_with_groq(prompt)
        return {"response": answer}

# -------------------------------------------------------------------
# 3. Helper Functions for Reading PDFs
# -------------------------------------------------------------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text() or ""
            text += page_text
    return text

def get_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

# -------------------------------------------------------------------
# 4. Streamlit App
# -------------------------------------------------------------------
def main():
    st.set_page_config("Chat PDF")
    st.header("Personalized Teaching Assistant")

    # Session flags
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if "student_level" not in st.session_state:
        st.session_state["student_level"] = None
    if "assistant" not in st.session_state:
        st.session_state["assistant"] = None

    # -------------------------
    # LOGIN LOGIC
    # -------------------------
    if not st.session_state["logged_in"]:
        # Show login form if not logged in
        st.subheader("Login to Access:")
        username = st.text_input("Username", key="username_input")
        password = st.text_input("Password", type="password", key="password_input")

        if st.button("Login"):
            student_level = authenticate_user(username, password)
            if student_level:
                st.session_state["student_level"] = student_level
                st.session_state["logged_in"] = True
                st.success(f"Welcome {username}! Your level is {student_level}.")
            else:
                st.error("Invalid credentials. Please try again.")

        # Stop here so we don’t render the rest of the page
        return  
    else:
        # If logged in, provide Logout
        if st.button("Logout"):
            st.session_state["logged_in"] = False
            st.session_state["student_level"] = None
            st.session_state["assistant"] = None
            st.success("You have been logged out. Returning to login page...")
            st.stop()  # Stop execution. On next rerun, user will see login form.

        # Show the rest of the UI for logged-in users
        st.write(f"You are logged in as `{st.session_state['student_level']}` level.")

    # -------------------------
    # PDF Upload & Processing
    # -------------------------
    with st.sidebar:
        st.title("PDF Menu")
        pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True)
        # Text box for entering the file name
        file_name = st.text_input("Enter a name for the PDF file", "")
        
        # Disable submit button if no PDF file is uploaded or file name is empty
        submit_disabled = not pdf_docs or not file_name
        if st.button("Submit & Process",disabled=submit_disabled):
            if not pdf_docs:
                st.warning("Please upload at least one PDF.")
            else:
                with st.spinner("Processing PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_chunks(raw_text)

                    # Create a new TeachingAssistant & build indexes
                    assistant = TeachingAssistant(
                        text_chunks,
                        groq_api_key="",
                        pdf_name=file_name
                    )
                    assistant.build_indexes(model_name="all-MiniLM-L6-v2")
                    st.session_state["assistant"] = assistant
                    st.success("PDFs processed! You can ask questions now.")
        # List the files from S3
    uploaded_pdfs = list_files_in_s3()
    uploaded_pdfs_names = [name[:-12] for name in uploaded_pdfs]
    selected_pdf = st.selectbox("Select a PDF", uploaded_pdfs_names)
    # Dropdown for student level, defaulting to level set during login
    student_level_selection = st.selectbox(
        "Choose your understanding level for this subject:",
        ["beginner", "intermediate", "advanced"],
        index=["beginner", "intermediate", "advanced"].index(st.session_state["student_level"])
    )

    if selected_pdf:
        # Load the index from S3
        assistant = TeachingAssistant(
                        groq_api_key="",
                        embedding_model = SentenceTransformer("all-MiniLM-L6-v2"),
                        pdf_name=selected_pdf
        )
        assistant.load_index_from_s3()
        st.session_state["assistant"] = assistant

        # Display chat for the selected PDF
        user_question = st.text_input(f"Ask a question about {selected_pdf}")
        if user_question:
            if st.session_state["assistant"] is None:
                st.warning("Please upload & process PDFs first.")
            else:
                level = st.session_state["student_level"] or "beginner"
                result = st.session_state["assistant"].user_input(
                    user_question,
                    student_level=student_level_selection
                )
                st.write(result["response"])

    # # -------------------------
    # # Ask a Question
    # # -------------------------
    # user_question = st.text_input("Ask a question about the uploaded PDFs")

    # if user_question:
    #     if st.session_state["assistant"] is None:
    #         st.warning("Please upload & process PDFs first.")
    #     else:
    #         level = st.session_state["student_level"] or "beginner"
    #         result = st.session_state["assistant"].user_input(
    #             user_question,
    #             student_level=level
    #         )
    #         st.write(result["response"])

if __name__ == "__main__":
    main()



# import streamlit as st
# from PyPDF2 import PdfReader
# import faiss
# import numpy as np
# import requests
# from sentence_transformers import SentenceTransformer
# from rank_bm25 import BM25Okapi
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# # -------------------------------------------------------------------
# # 1. Student Credentials & Authentication
# # -------------------------------------------------------------------
# student_credentials = {
#     "alice":    {"password": "secret1", "level": "beginner"},
#     "bob":      {"password": "secret2", "level": "intermediate"},
#     "charlie":  {"password": "secret3", "level": "advanced"},
# }

# def authenticate_user(username, password):
#     """Returns the student's level if credentials match, else None."""
#     user = student_credentials.get(username)
#     if user and user["password"] == password:
#         return user["level"]
#     return None

# # -------------------------------------------------------------------
# # 2. TeachingAssistant Class
# # -------------------------------------------------------------------
# class TeachingAssistant:
#     def __init__(self, text_chunks, groq_api_key=""):
#         self.text_chunks = text_chunks
#         self.index = None
#         self.embedding_model = None
#         self.bm25 = None
#         self.GROQ_API_KEY = groq_api_key

#     def build_indexes(self, model_name="all-MiniLM-L6-v2"):
#         """Build FAISS and BM25 indexes from text_chunks."""
#         # 1) Load/initialize sentence transformer
#         self.embedding_model = SentenceTransformer(model_name)

#         # 2) Create embeddings & build FAISS index
#         embeddings = np.array([self.embedding_model.encode(chunk) for chunk in self.text_chunks])
#         dimension = embeddings.shape[1]
#         self.index = faiss.IndexFlatL2(dimension)
#         self.index.add(embeddings)

#         # 3) Build BM25 index
#         tokenized_corpus = [chunk.split() for chunk in self.text_chunks]
#         self.bm25 = BM25Okapi(tokenized_corpus)

#     def retrieve_course_materials(self, query, top_k=5):
#         """Hybrid retrieval using FAISS + BM25."""
#         if not self.text_chunks or not self.index or not self.embedding_model:
#             return []

#         # --- FAISS Retrieval ---
#         query_vector = np.array(self.embedding_model.encode(query)).reshape(1, -1)
#         distances, indices = self.index.search(query_vector, top_k)
#         faiss_results = [
#             (self.text_chunks[i], 1 / (distances[0][idx] + 1e-5))
#             for idx, i in enumerate(indices[0])
#             if i < len(self.text_chunks)
#         ]

#         # --- BM25 Retrieval ---
#         tokenized_query = query.split()
#         bm25_scores = self.bm25.get_scores(tokenized_query)
#         top_k_bm25_indices = np.argsort(bm25_scores)[::-1][:top_k]
#         bm25_results = [(self.text_chunks[i], bm25_scores[i]) for i in top_k_bm25_indices]

#         # --- Combine & Sort ---
#         combined_results = faiss_results + bm25_results
#         combined_results = sorted(combined_results, key=lambda x: x[1], reverse=True)

#         final = []
#         seen = set()
#         for chunk, score in combined_results:
#             if chunk not in seen:
#                 final.append(chunk)
#                 seen.add(chunk)
#             if len(final) == top_k:
#                 break
#         return final

#     def generate_response_with_groq(self, prompt):
#         """Call Groq's API for an LLM response (optional)."""
#         if not self.GROQ_API_KEY:
#             return "Groq API key not provided. Cannot generate response."

#         url = "https://api.groq.com/openai/v1/chat/completions"
#         headers = {
#             "Authorization": f"Bearer {self.GROQ_API_KEY}",
#             "Content-Type": "application/json",
#         }
#         data = {
#             "model": "llama3-70b-8192",
#             "messages": [{"role": "user", "content": prompt}],
#             "max_tokens": 200,
#         }
#         try:
#             response = requests.post(url, headers=headers, json=data)
#             response_json = response.json()
#             if "choices" in response_json:
#                 return response_json["choices"][0]["message"]["content"]
#             return f"Error: 'choices' not found.\nFull response: {response_json}"
#         except Exception as e:
#             return f"Error calling Groq: {str(e)}"

#     def user_input(self, user_question, student_level="beginner"):
#         """
#         Retrieves docs, creates a prompt with level-specific instructions,
#         and calls the LLM for a final answer.
#         """

#         # 1) Retrieve relevant course materials
#         retrieved_docs = self.retrieve_course_materials(user_question)
#         if not retrieved_docs:
#             return {"response": "No relevant info found or PDFs not processed yet."}

#         context = "\n".join(retrieved_docs)

#         # 2) Add instructions based on level
#         if student_level == "beginner":
#             level_instructions = (
#                 "Explain thoroughly, define all terms, keep it simple, and give step-by-step guidance."
#             )
#         elif student_level == "intermediate":
#             level_instructions = (
#                 "Give a balanced explanation, assume familiarity with common terms, but add detail when necessary."
#             )
#         else:  # "advanced"
#             level_instructions = (
#                 "Be concise, focus on advanced details, and assume knowledge of fundamentals."
#             )

#         # 3) Build the prompt
#         prompt = f"""
#         You are an AI Teaching Assistant. Answer the user's question strictly from the course material below.
#         If you do not find the answer in the course material, say so.

#         Student level: {student_level}
#         Level Instructions: {level_instructions}

#         Course Material:
#         {context}

#         Question: {user_question}

#         Answer:
#         """

#         # 4) Generate the final answer
#         answer = self.generate_response_with_groq(prompt)
#         return {"response": answer}

# # -------------------------------------------------------------------
# # 3. Helper Functions for Reading PDFs
# # -------------------------------------------------------------------
# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             page_text = page.extract_text() or ""
#             text += page_text
#     return text

# def get_chunks(text):
#     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     return splitter.split_text(text)

# # -------------------------------------------------------------------
# # 4. Streamlit App
# # -------------------------------------------------------------------
# def main():
#     st.set_page_config("Chat PDF")
#     st.header("Chat with Personal Teacher")

#     # Session flags
#     if "logged_in" not in st.session_state:
#         st.session_state["logged_in"] = False
#     if "student_level" not in st.session_state:
#         st.session_state["student_level"] = None
#     if "assistant" not in st.session_state:
#         st.session_state["assistant"] = None

#     # -------------------------
#     # LOGIN LOGIC
#     # -------------------------
#     if not st.session_state["logged_in"]:
#         # Show login form if not logged in
#         st.subheader("Login to Access:")
#         username = st.text_input("Username", key="username_input")
#         password = st.text_input("Password", type="password", key="password_input")

#         if st.button("Login"):
#             student_level = authenticate_user(username, password)
#             if student_level:
#                 st.session_state["student_level"] = student_level
#                 st.session_state["logged_in"] = True
#                 st.success(f"Welcome {username}! Your level is {student_level}.")
#             else:
#                 st.error("Invalid credentials. Please try again.")

#         # Stop here so we don’t render the rest of the page
#         return  
#     else:
#         # If logged in, provide Logout
#         if st.button("Logout"):
#             st.session_state["logged_in"] = False
#             st.session_state["student_level"] = None
#             st.session_state["assistant"] = None
#             st.success("You have been logged out. Returning to login page...")
#             st.stop()  # Stop execution. On next rerun, user will see login form.

#         # Show the rest of the UI for logged-in users
#         st.write(f"You are logged in as `{st.session_state['student_level']}` level.")

#     # -------------------------
#     # PDF Upload & Processing
#     # -------------------------
#     with st.sidebar:
#         st.title("PDF Menu")
#         pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True)
#         if st.button("Submit & Process"):
#             if not pdf_docs:
#                 st.warning("Please upload at least one PDF.")
#             else:
#                 with st.spinner("Processing PDFs..."):
#                     raw_text = get_pdf_text(pdf_docs)
#                     text_chunks = get_chunks(raw_text)

#                     # Create a new TeachingAssistant & build indexes
#                     assistant = TeachingAssistant(
#                         text_chunks,
#                         groq_api_key=""
#                     )
#                     assistant.build_indexes(model_name="all-MiniLM-L6-v2")
#                     st.session_state["assistant"] = assistant
#                     st.success("PDFs processed! You can ask questions now.")

#     # -------------------------
#     # Ask a Question
#     # -------------------------
#     user_question = st.text_input("Ask a question about the uploaded PDFs")

#     if user_question:
#         if st.session_state["assistant"] is None:
#             st.warning("Please upload & process PDFs first.")
#         else:
#             level = st.session_state["student_level"] or "beginner"
#             result = st.session_state["assistant"].user_input(
#                 user_question,
#                 student_level=level
#             )
#             st.write(result["response"])

# if __name__ == "__main__":
#     main()
