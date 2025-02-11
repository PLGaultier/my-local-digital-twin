import streamlit as st
import yaml
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM

# Set page configuration
st.set_page_config(page_title="Digital Twin of Pierre-Louis Gaultier", page_icon="🤖")

# Load configuration
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

user_info = config['user_info']
model_settings = config['model_settings']

# --- Document Processing (same as before) ---
@st.cache_resource
def load_vector_db():
    # Load/split your CV
    loader = PyPDFLoader("../data/cv_03_02.pdf")
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(pages)
    
    # Use free embeddings (no OpenAI)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)

db = load_vector_db()

# --- Ollama Setup ---
llm = OllamaLLM(
    model=model_settings['model_name'],  # Your chosen local model
    temperature=model_settings['temperature'],  # Reduce randomness for factual answers
    num_gpu=model_settings['num_gpu'],  # Use GPU layers (M3 Pro optimization)
    system=f"You are a digital twin of {user_info['name']}. Answer questions strictly using the provided context about their skills, experience, and preferences. Be concise and professional.",
)

# --- Streamlit UI Enhancements ---

# Sidebar with user information
st.sidebar.title(f"About {user_info['name']}")
st.sidebar.info(
    f"""
    **Name:** {user_info['name']}  
    **Profession:** {user_info['profession']}  
    **Specialization:** {user_info['specialization']}  
    """
)

st.title("My Local Digital Twin 👤")
st.subheader("Chat with your digital twin to learn more about your skills and experiences.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

# Input for user prompt
if prompt := st.chat_input("Ask me about my skills, experience, etc."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    # Retrieve relevant context from local FAISS DB
    relevant_docs = db.similarity_search(prompt, k=4)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    # Display retrieved context
    with st.expander("Retrieved Context"):
        st.write(context)
    
    # Generate response with Ollama
    response = llm.invoke(f"""
        [CONTEXT ABOUT {user_info['name'].upper()}]
        {context}
        
        [QUESTION]
        {prompt}
        
        Answer as {user_info['name']}'s digital twin:
    """)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)
