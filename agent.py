# agent.py

import os
import tempfile

# Import langchain dependencies
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Bring in streamlit for UI development
import streamlit as st

# Create an Ollama instance configured for llama3.2
llm = OllamaLLM(
    model="llama3.2",
    temperature=0.7,
)

# This function loads multiple PDFs and creates a vector database
@st.cache_resource
def load_pdfs(pdf_files):
    try:
        documents = []
        
        # Load all PDFs
        for pdf_file in pdf_files:
            st.write(f"Loading PDF: {pdf_file.name}")
            
            # Create a temporary file for each PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf_file.getvalue())
                
                # Load the document and add metadata
                loader = PyPDFLoader(tmp_file.name)
                docs = loader.load()
                # Add metadata to each document
                for doc in docs:
                    doc.metadata["source"] = pdf_file.name
                    doc.metadata["title"] = pdf_file.name.replace('.pdf', '')
                    # Add page number if available
                    if 'page' in doc.metadata:
                        doc.metadata["location"] = f"Page {doc.metadata['page']}"
                documents.extend(docs)
                
                # Clean up temp file
                os.unlink(tmp_file.name)
        
        # Split the documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200  # Increased overlap
        )
        texts = text_splitter.split_documents(documents)
        
        if not texts:
            st.error("No text content found in PDFs")
            return None
            
        # Create embeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2")
        
        # Create vector store
        vectorstore = FAISS.from_documents(texts, embeddings)
        
        st.success(f"Successfully loaded {len(pdf_files)} PDFs with {len(texts)} text chunks")
        return vectorstore
        
    except Exception as e:
        st.error(f"Error loading PDFs: {str(e)}")
        return None

# Update the file uploader to accept multiple files
uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    vectorstore = load_pdfs(uploaded_files)
    
    if vectorstore is None:
        st.error("Failed to load PDFs. Please try uploading again.")
        st.stop()
        
    try:
        # Create a list of paper titles for context
        paper_titles = [f.name for f in uploaded_files]
        
        # Create a custom prompt template
        prompt_template = """
        You are a knowledgeable AI assistant analyzing scientific papers. You have access to the following papers:

        {paper_list}

        When responding to the user’s questions, please follow these instructions:

        1. **If relevant information is available in the papers, cite it specifically** and label it as "Information from Papers." Mention the paper's title, author(s), and year for each point you cite.
        2. **If additional general knowledge is needed to provide a complete answer, clearly label it as "General Knowledge."** Ensure that this general information is distinct from paper-specific insights.
        3. **Organize information by topic** if multiple points are covered, and use subheadings for clarity.
        4. **Provide specific methods, findings, or techniques from each cited paper** rather than generalizing the information. Give detailed explanations as presented in the papers.

        **Response Format Example:**

        **Information from Papers:**
        - "[Specific finding or technique] according to Baghbaderani et al. (2008), large-scale suspension culture systems were crucial for..."

        **General Knowledge:**
        - "[Relevant general knowledge that complements paper findings]."

        User Question: {question}

        Context from papers: {context}

        **Important:**
        The user may or may not ask you questions about the papers. If the user asks you questions about the papers, you must provide information from the papers. Otherwise engage in the conversation prompted by the user.
        You can answer questions unrelated to the papers as well. 
        """

        # Create the prompt
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["paper_list", "question", "context"]
        )

        # Create the chain
        chain = LLMChain(
            llm=llm,
            prompt=PROMPT,
            verbose=True
        )
        
    except Exception as e:
        st.error(f"Error creating chain: {str(e)}")
        st.stop()

# Set up the streamlit title
st.title("RAG PDF Minion")

# Setup a session state message variable to hold all the old messages
if 'messages' not in st.session_state:
    st.session_state.messages = []
     
# Display all the historical messages
for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message['content'])

# Build a prompt input template to display the prompts
prompt = st.chat_input("Enter your prompt here")

# If the user hits enter then
if prompt:
    # Display the prompt in the chat
    st.chat_message("user").markdown(prompt)
    # Store the prompt in the session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    try:
        # Get relevant documents from vectorstore
        docs = vectorstore.similarity_search(prompt, k=6)
        context = "\n\n".join([doc.page_content for doc in docs])
        paper_list = "\n".join([f"- {title}" for title in paper_titles])
        
        # Run the chain
        response = chain.invoke({
            "paper_list": paper_list,
            "question": prompt,
            "context": context
        })
        
        response_text = response.get('text', 'No response generated')
        
        # Add source information
        sources = set([doc.metadata.get('source', 'Unknown') for doc in docs])
        #  response_text += "\n\n**Sources consulted:**\n" + "\n".join([f"- {source}" for source in sources])
        
        # Display the response in the chat
        st.chat_message("assistant").markdown(response_text)
        # Store the response in the session state
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")
        st.stop()

