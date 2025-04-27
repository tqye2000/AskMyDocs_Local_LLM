##################################################################
# RAG based on local LLMs
#
# History
# When      | Who            | What
# 27/04/2025| Tian-Qing Ye   | Created
##################################################################
import libs
from libs import Locale
from langchain.prompts import PromptTemplate
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
#from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.llms import LlamaCpp
from langchain.callbacks.base import BaseCallbackHandler
from langchain_community.vectorstores import FAISS
#from langchain_community.utilities import DuckDuckGoSearchAPIWrapper # --> no KEY required!

from llama_cpp import Llama
from InstructorEmbedding import INSTRUCTOR

import streamlit as st

import sys
import os
from datetime import datetime
import shutil


import logging
logging.basicConfig(level=logging.ERROR)
os.environ["LLAMA_CPP_LOG_LEVEL"] = "ERROR"

MODEL_PATH = "D:/hf_models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"    # LLM: Mistral-7B-Instruct (quantized model)
EMBEDDINGS_MODEL_PATH = r"D:/hf_models/hkunlp/instructor-xl"        # Embedding: Instructor-XL

FAISS_PATH = "./saved_faiss_index"
SCORE_THRESHOLD = 0.5   # Set a threshold for semantic distance. Above this, the document is considered to be irrelevant.

# Define the folder path where your documents are
DATA_PATH = "./tempDir"
if not os.path.exists(DATA_PATH):
    os.mkdir(DATA_PATH)

AI_ROLE_OPTIONS_EN = [
    "helpful assistant",
    "code assistant",
    "code reviewer",
    "text improver",
    "english teacher",
]

en = Locale(
    ai_role_options=AI_ROLE_OPTIONS_EN,
    ai_role_prefix="You are an assistant",
    ai_role_postfix="Answer as concisely as possible.",
    title="Ask Your Docs",
    language="English",
    lang_code="en",
    chat_placeholder="Your Question:",
    chat_run_btn="Ask",
    chat_clear_btn="Clear",
    chat_save_btn="Save",
    select_placeholder1="Select Model",
    select_placeholder2="Select Role",
    select_placeholder3="Create Role",
    radio_placeholder="Role Interaction",
    radio_text1="Select",
    radio_text2="Create",
    stt_placeholder="To Hear The Voice Of AI Press Play",
)

# 
class LangChainInstructorEmbeddings:
    '''
    '''
    def __init__(self, model_path: str):
        self.model = INSTRUCTOR(model_path)

    def embed_documents(self, texts):
        inputs = [["Represent the document for retrieval:", text] for text in texts]
        return self.model.encode(inputs)

    def embed_query(self, query):
        inputs = [["Represent the query for retrieval:", query]]
        return self.model.encode(inputs)[0]
    
    def __call__(self, text):
        # This is the important part!
        return self.embed_query(text)


class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""
        self.container.empty()

    def reset(self):
        """Reset the internal buffer"""
        self.text = ""
        self.container.empty()

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

def save_log(text) -> None:
    now = datetime.now() # current date and time
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    
    f = open("QA.log", "a", encoding='utf-8',)
    f.write(f'[{date_time}]: {text}\n\n')
    f.close()

def show_text_input() -> None:
    st.text_area(label=st.session_state.locale.chat_placeholder, value=st.session_state.user_text, key="user_text")
            
@st.cache_data
def save_uploadedfile(uploadedfile) -> None:

    if not os.path.exists(DATA_PATH):
        os.mkdir(DATA_PATH)

    with open(os.path.join(DATA_PATH, uploadedfile.name),"wb") as f:
         f.write(uploadedfile.getbuffer())

     #return st.success("Saved File:{} to tempDir".format(uploadedfile.name))

def Load_Files():
    uploaded_files = st.file_uploader("Load your file(s)", type=['docx', 'txt', 'pdf'], accept_multiple_files=True)
    count = 0
    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            bytes_data = uploaded_file.read()
            file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type}
            save_uploadedfile(uploaded_file)
            count += 1

        # st.write("filename:", uploaded_file.name)
        # st.write(bytes_data)
    
    # nfiles = len(os.listdir(DATA_PATH))
    # return nfiles

    return count
    
def Clear_Chat() -> None:
    st.session_state.generated = []
    st.session_state.topic_chats = []
    st.session_state.user_text = ""
    
def Clear_FAISS_Index():
    if os.path.exists(FAISS_PATH):
        shutil.rmtree(FAISS_PATH)
        print("FAISS index cleared.")

def Delete_Files(folder = DATA_PATH) -> None:
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    Clear_FAISS_Index()
    st.session_state.new_file_loaded = False

def Display_Chat(chat_history):
    """
    Display chat history in the Streamlit app.
    """
    with st.container():
        for chat in reversed(chat_history):
            query = chat['You']
            st.chat_message("user").markdown(query)
            resp = chat['Bot']
            st.chat_message("assistant").markdown(resp, unsafe_allow_html=True)  # <-- Important: use markdown!

def Hybrid_Search(query, search_index, all_documents, top_k_semantic=4, top_k_final=6, score_threshold=1.0):
    """
    Perform a hybrid search: semantic (FAISS) + keyword match, with score filtering.

    Args:
        query (str): The user query.
        search_index (FAISS): The FAISS search index object.
        all_documents (List[Document]): List of all preloaded documents (for keyword scanning).
        top_k_semantic (int): Number of semantic top hits to retrieve.
        top_k_final (int): Number of documents to return finally after merging.
        score_threshold (float): Maximum allowed semantic distance (lower = more strict).

    Returns:
        List[Document]: Merged and filtered list of top relevant documents.
    """
    if not query.strip():
        return []

    # 1. Semantic Search
    semantic_hits = search_index.similarity_search_with_score(query, k=top_k_semantic)

    # 2. Filter out low-confidence semantic matches
    good_semantic_docs = []
    for doc, score in semantic_hits:
        if score <= score_threshold:
            good_semantic_docs.append(doc)
            save_log(f"Included: {doc.page_content} ({score})")

    print(f"Semantic hits: {len(semantic_hits)}, Good semantic hits (score <= {score_threshold}): {len(good_semantic_docs)}")

    # 3. Keyword Search
    keywords = query.lower().split()
    keyword_hits = []
    for doc in all_documents:
        text = doc.page_content.lower()
        if any(keyword in text for keyword in keywords):
            keyword_hits.append(doc)

    print(f"Keyword hits: {len(keyword_hits)}")

    # 4. Merge Results without duplication
    seen = set()
    combined_docs = []

    # Add good semantic hits first
    for doc in good_semantic_docs:
        if doc.page_content not in seen:
            combined_docs.append(doc)
            seen.add(doc.page_content)

    ##=== Note: Uncomment this if you want to add keyword hits after semantic hits ===
    # # Then add keyword hits
    # for doc in keyword_hits:
    #     if doc.page_content not in seen:
    #         combined_docs.append(doc)
    #         seen.add(doc.page_content)

    # 5. Limit to top_k_final
    if len(combined_docs) > top_k_final:
        combined_docs = combined_docs[:top_k_final]

    return combined_docs


def Build_Search_Index(docPath, files):
    # 1. Check if FAISS index already exists
    if os.path.exists(os.path.join(FAISS_PATH, "index.faiss")):
        print("Loading existing FAISS index...")
        
        # Load embeddings
        embeddings_model_path = EMBEDDINGS_MODEL_PATH
        embeddings = LangChainInstructorEmbeddings(embeddings_model_path)
        
        # Load FAISS index
        search_index = FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        
        st.session_state.new_file_loaded = False
        return search_index

    # Otherwise build as normal
    print("No FAISS index found, building a new one...")
    with st.spinner("Building a new index..."):
        sources = []
        for file in files:
            file = os.path.join(docPath, file)
            print(file)
            if file.split(".")[-1] == 'txt':
                sources.append(libs.get_unstructured_data(file))
            elif file.split(".")[-1] == 'docx':
                sources.append(libs.get_docx_data(file))
            elif file.split(".")[-1] == 'pdf':
                sources.append(libs.get_pdf_data(file))
            elif file.split(".")[-1] == 'ppt':
                sources.append(libs.get_ppt_data(file))
                    
        # Smart chunking using RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,          # Max size limit for LLM friendliness
            chunk_overlap=100,       # Preserve some overlap for context
            separators=["\n\n", "\n", ".", " ", ""]  # Try heading first, then paragraph, then sentence, fallback
        )

        source_chunks = []
        for source in sources:
            if not source.page_content.strip():
                print(f"Warning: Empty document encountered: {source.metadata}")
                continue

            chunks = splitter.split_text(source.page_content)
            for chunk in chunks:
                source_chunks.append(Document(page_content=chunk, metadata=source.metadata))

        if not source_chunks:
            return None

        # Prepare documents for embedding
        print(f"Total chunks created: {len(source_chunks)}")

        # Only proceed if chunks exist
        texts = [doc.page_content for doc in source_chunks]
        if not texts:
            return None

        # Load embedding model
        embeddings_model_path = EMBEDDINGS_MODEL_PATH #r"D:/hf_models/hkunlp/instructor-xl"
        #embeddings = INSTRUCTOR(embeddings_model_path)
        embeddings = LangChainInstructorEmbeddings(embeddings_model_path)

        # Build FAISS index
        search_index = FAISS.from_documents(source_chunks, embeddings)

        # SAVE the FAISS index to disk
        search_index.save_local(FAISS_PATH)

    st.session_state.new_file_loaded = False  #All docs are indexed.

    return search_index

@st.cache_resource
def Create_Model_Chain(_callbacks=None):

    model_path = st.session_state.use_llm
    llm = LlamaCpp(
        model_path=model_path,
        n_gpu_layers=50,  # You can adjust this depending on VRAM
        n_ctx=8192,        # context length: max 16384
        #temperature=0.2,
        #max_tokens=1024,
        streaming=True,   # <-- enable streaming
        callbacks=_callbacks if _callbacks else [] # Pass the callback handler here
    )

    template = """Given the following context and a question, create an answer with reasoning and the references ('source'). Please format the answer using pretty format, such as markdown or html format!
        If you don't know the answer, just say you don't know.
        context: {summaries}
        question: {question}
        ANSWER:
        """

    PROMPT = PromptTemplate(template=template, input_variables=["summaries", "question"])
    chain = load_qa_with_sources_chain(llm=llm, chain_type="stuff", prompt=PROMPT)

    return chain


##############################################
################ MAIN ########################
##############################################
def main(argv):

    model_name = st.selectbox("Choose the Model", ("Mistral-7B-Instruct",))
    if model_name.startswith("Mistral-7B-Instruct"):
        st.session_state.use_llm = MODEL_PATH
    # elif model_name.startswith("Codegemma"):
    #     st.session_state.use_llm = "google/codegemma-7b-it"
    else:
        # Use local Mistral model
        st.session_state.use_llm = MODEL_PATH

    nfiles = Load_Files()
    if nfiles > 0:
        st.session_state.new_file_loaded = True
        #st.success("File loading done")

    if st.session_state.new_file_loaded:
        Clear_FAISS_Index()
        st.session_state.new_file_loaded = False

    st.session_state.menu_placeholder = st.empty()        # some menu buttons
    st.session_state.input_placeholder = st.empty()        # user input
    st.session_state.status_placeholder = st.container()
    st.session_state.output_container = st.empty()      # AI streaming output
    st.session_state.chats_placeholder = st.empty()        # chat history

    # Remove documents from the temp folder
    c1, c2 = st.session_state.menu_placeholder.columns(2)
    with c1:
        st.button("New Topic", on_click=Clear_Chat)
    with c2:
        st.button("Clear Documents", on_click=Delete_Files)

    # Build search Index
    files = os.listdir(DATA_PATH)
    search_index = Build_Search_Index(DATA_PATH, files)
    if search_index is None:
        st.session_state.status_placeholder.write("Warning: No documents loaded or found!")
    
    ## Build Model Chain
    try:
        with st.spinner('Wait ...'):
            # Create the Streamlit streaming callback
            streaming_callback = StreamlitCallbackHandler(st.session_state.output_container)
            chain = Create_Model_Chain(_callbacks=[streaming_callback])
            chain.llm_chain.llm.callbacks = [streaming_callback]
    except Exception as e:
        st.write(e)
        return

    history = []
    st.session_state.chat = {}

    with st.session_state.input_placeholder.form(key="input_form", clear_on_submit = True):
        col1, col2 = st.columns([7,1])
        with col1:
            st.session_state.input_text = st.text_area("Your Query:")
        with col2:
            st.write("\xa0")
            st.write("\xa0")
            send_button = st.form_submit_button(label="Send")

    if send_button:
        if(st.session_state.input_text.strip() != ''):
            query = st.session_state.input_text.strip()
            print(f"Query: {query}")
            history.append(f"You: {query}")
            st.session_state.chat['You'] = query
            good_docs = []
            if search_index:
                # # Assuming you cached all_documents somewhere
                # good_docs = Hybrid_Search(query, search_index, all_documents, top_k_semantic=4, top_k_final=6)
                wide_semantic_docs = search_index.similarity_search(query, k=50) # Get top 50 documents
                good_docs = Hybrid_Search(query, search_index, wide_semantic_docs, top_k_semantic=4, top_k_final=6, score_threshold=SCORE_THRESHOLD)
                save_log(f"Query: {query}\n Contexts:")
                for doc in good_docs:
                    save_log(f"Included: {doc.page_content[:200]}...")

                print(f"Total good_docs: {len(good_docs)}")

            # === Build the Prompt using the chat history and the current query
            print(f"Topic chat history: {st.session_state.topic_chats}")
            prompt = libs.build_mistral_chat_prompt(st.session_state.topic_chats, query)
            print(f"Prompt: {prompt}")

            # == LLM answering
            if len(good_docs) == 0 and search_index:
                answer = "I don't know based on the provided information."
            else:
                # == LLM answering
                with st.spinner('Wait ...'):
                    results = chain({"input_documents": good_docs, "question": prompt}, return_only_outputs=True)
                answer = results["output_text"]

            # === Build the Sources List ===
            sources_set = set()  # Use set to avoid duplicates
            for doc in good_docs:
                source = doc.metadata.get('source', 'Unknown Source')
                page = doc.metadata.get('page', None)
                if page is not None:
                    source_entry = f"{source} (page {page})"
                else:
                    source_entry = f"{source}"

                sources_set.add(source_entry)

            # Turn into a nice string
            if sources_set:
                sources_list = "\n".join([f"- 📄 {s}" for s in sorted(sources_set)])
                sources_markdown = f"\n\n**Sources:**\n{sources_list}"
            else:
                sources_markdown = ""

            # === Append sources to the answer ===
            answer_with_sources = answer + sources_markdown

            st.session_state.chat['Bot'] = answer_with_sources
            st.session_state.chat_history.append(st.session_state.chat)
            #st.session_state.topic_chats.append(st.session_state.chat)
            st.session_state.topic_chats.append(("User", query))
            st.session_state.topic_chats.append(("AI", answer_with_sources))
            st.session_state.chat = {}  # Reset chat for next input

            # displaying chat history
            if st.session_state.chat_history:
                with st.session_state.chats_placeholder:
                    Display_Chat(st.session_state.chat_history)
                #Clear streaming output
                streaming_callback.reset()
                
            history.append(f"Bot: {answer_with_sources}")
            save_log("\n".join(history))

##############################
# Program Start
##############################
if __name__ == "__main__":

    # Initialising session state
    if "use_llm" not in st.session_state:
        st.session_state.use_llm = MODEL_PATH

    if 'new_file_loaded' not in st.session_state:
        st.session_state.new_file_loaded = False

    if "locale" not in st.session_state:
        st.session_state.locale = en

    if "generated" not in st.session_state:
        st.session_state.generated = []

    if "chat" not in st.session_state:
        st.session_state.chat = {"User" : "",  "Bot" : {}}

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []      # the chat history of the entire session

    if "topic_chats" not in st.session_state:
        st.session_state.topic_chats = []          # the chat history of a topic. Reset when a new topic is selected.

    if "user_text" not in st.session_state:
        st.session_state.user_text = ""

    st.markdown(f"<h1 style='text-align: center;'>{st.session_state.locale.title[0]}</h1>", unsafe_allow_html=True)

    main(sys.argv)
    
