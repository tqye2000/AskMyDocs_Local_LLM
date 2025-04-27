from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_community.document_loaders import PyPDFLoader

from typing import List
import os

class Locale:    
    ai_role_options: List[str]
    ai_role_prefix: str
    ai_role_postfix: str
    title: str
    language: str
    lang_code: str
    chat_placeholder: str
    chat_run_btn: str
    chat_clear_btn: str
    chat_save_btn: str
    select_placeholder1: str
    select_placeholder2: str
    select_placeholder3: str
    radio_placeholder: str
    radio_text1: str
    radio_text2: str
    stt_placeholder: str
    
    def __init__(self, 
                ai_role_options, 
                ai_role_prefix,
                ai_role_postfix,
                title,
                language,
                lang_code,
                chat_placeholder,
                chat_run_btn,
                chat_clear_btn,
                chat_save_btn,
                select_placeholder1,
                select_placeholder2,
                select_placeholder3,
                radio_placeholder,
                radio_text1,
                radio_text2,
                stt_placeholder,
                
                ):
        self.ai_role_options = ai_role_options, 
        self.ai_role_prefix= ai_role_prefix,
        self.ai_role_postfix= ai_role_postfix,
        self.title= title,
        self.language= language,
        self.lang_code= lang_code,
        self.chat_placeholder= chat_placeholder,
        self.chat_run_btn= chat_run_btn,
        self.chat_clear_btn= chat_clear_btn,
        self.chat_save_btn= chat_save_btn,
        self.select_placeholder1= select_placeholder1,
        self.select_placeholder2= select_placeholder2,
        self.select_placeholder3= select_placeholder3,
        self.radio_placeholder= radio_placeholder,
        self.radio_text1= radio_text1,
        self.radio_text2= radio_text2,
        self.stt_placeholder= stt_placeholder,

# def get_pdf_data(filepath:str) -> str:
#     '''
#     File types: pdf
#     '''
#     loader = PyPDFLoader(filepath)
#     docs = loader.load()
#     doc = docs[0]

#     return doc.page_content

# def get_unstructured_data(filepath) -> str:
#     '''
#     File types: text, html
#     '''
#     loader = UnstructuredFileLoader(filepath)
#     docs = loader.load()
#     doc = docs[0]

#     return doc.page_content

def get_docx_data(filepath):
    '''
    File types: docx
    '''
    loader = UnstructuredWordDocumentLoader(filepath)

    data = loader.load()
    doc = data[0]

    file_name = os.path.basename(filepath)
    doc.metadata["source"] = file_name
    return doc

def get_ppt_data(filepath):
    '''
    File types: powerpoint document
    '''
    loader = UnstructuredPowerPointLoader(filepath)
    docs = loader.load()
    doc = docs[0]

    file_name = os.path.basename(filepath)
    doc.metadata["source"] = file_name

    '''
    [Document(page_content='Adding a Bullet Slide\n\nFind the bullet slide layout\n\nUse _TextFrame.text for first bullet\n\nUse _TextFrame.add_paragraph() for subsequent bullets\n\nHere is a lot of text!\n\nHere is some text in a text box!', lookup_str='', metadata={'source': 'example_data/fake-power-point.pptx'}, lookup_index=0)]
    '''
    return doc

def get_pdf_data(filepath):
    '''
    File types: pdf
    '''
    loader = PyPDFLoader(filepath)
    docs = loader.load()
    doc = docs[0]

    file_name = os.path.basename(filepath)
    doc.metadata["source"] = file_name
    return doc

def get_unstructured_data(filepath):
    '''
    File types: text, html, pdf
    '''
    loader = UnstructuredFileLoader(filepath)
    docs = loader.load()
    doc = docs[0]

    file_name = os.path.basename(filepath)
    doc.metadata["source"] = file_name
    return doc

def text_preprocessing(filepath):
    '''
    Readin and Preprocessing training data
    '''
    with open(filepath, encoding="utf-8") as f:
        magbasics = f.read()

    file_name = os.path.basename(filepath)
    doc = Document(page_content=magbasics, metadata={"source": f"{file_name}"},)

    return doc

def build_mistral_chat_prompt(chat_history_pairs, current_user_query):
    '''
    Build the chat prompt for Mistral model.
    @param chat_history_pairs: List of tuples containing the role and message. 
                               Each tuple should be in the format (role, message), where role is either "User" or "AI".
    The format is as follows:
    <s>[INST] User message [/INST] AI message </s><s>[INST] User message [/INST] AI message </s><s>[INST] User message [/INST]
    [INST] current user query [/INST]

    '''
    prompt = "<s>"  # Start token
    for role, message in chat_history_pairs:
        if role == "User":
            prompt += f"[INST] {message.strip()} [/INST]"
        else:
            prompt += f" {message.strip()} </s><s>"
    # Add current user question, waiting for model to answer
    prompt += f"[INST] {current_user_query.strip()} [/INST]"
    return prompt
