# Introduction 
An example of using a locally hosted open-source LLM (quantized model) for general query and/or RAG tasks.

This example demonstrates:
- How to use a quantized model, e.g. mistral-7b-instruct-v0.2.Q4_K_M.gguf, compeletely locally.
- How to build a fully function Chat bot with RAG (Retrieval Argumented Generation) capability by using
  langchain and streamlit (UI)
 
# Getting Started
1.	Clone (or download) the repository to your local machine
2.	Install the requirements
3.	Download the model from the Hugging Face: TheBloke/Mistral-7B-Instruct-v0.2-GGUF,
    <br>and save it to a local folder, e.g.: d:\hf_models
4.  Download the embedding model from the Hugging Face: hkunlp/instructor-xl,
5.  <br>and save all the model files to a local folder, e.g.: d:\hf_models\hkunlp\instructor-xl
6.  Open the app.py file and edit the MODEL_PATH and EMBEDDINGS_MODEL_PATH variables.
7.	Run the startup bat file: run.bat

# Build and Test
TODO: Describe and show how to build your code and run the tests. 

# Contribute
TODO: Explain how other users and developers can contribute to make your code better. 
