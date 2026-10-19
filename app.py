import os
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- 1. Streamlit UI & Page Configuration ---
st.set_page_config(
    page_title="בוט מידע - קבוצת הפניקס", 
    page_icon="🏢", 
    layout="centered"
)

st.title("🏢 בוט המידע הרשמי של קבוצת הפניקס")
st.caption("מערכת מענה אוטומטית המבוססת אך ורק על מסמכי החברה הרשמיים")

# --- 2. Secure API Key Retrieval from Streamlit Secrets ---
# Retrieve OpenAI API key safely from Streamlit server secrets or environment
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
elif os.environ.get("OPENAI_API_KEY"):
    pass
else:
    st.error("מפתח API אינו מוגדר במערכת. אנא הגדר OPENAI_API_KEY ב-Secrets של Streamlit.")
    st.stop()

# --- 3. RAG System Initialization & Caching ---
@st.cache_resource(show_spinner="מעבד את מסמכי הפניקס...")
def init_rag_system():
    """
    Loads PDF documents from 'data/' directory, splits text into chunks,
    and indexes them into a local Chroma vector database using OpenAI embeddings.
    """
    # Load all PDF files from the local 'data' folder
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()
    
    if not documents:
        return None

    # Chunking: Split documents into smaller contiguous texts
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    splits = text_splitter.split_documents(documents)

    # Embedding & Vector Storage
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    
    # Return retriever object fetching top 3 relevant chunks
    return vectorstore.as_retriever(search_kwargs={"k": 3})

# Ensure the 'data' directory exists locally or on the server
if not os.path.exists("data"):
    os.makedirs("data")
    st.info("נוצרה תיקייה בשם 'data'. אנא שים בה את קובצי ה-PDF של הפניקס והרץ מחדש.")
    st.stop()

retriever = init_rag_system()

if retriever is None:
    st.warning("תיקיית 'data' ריקה. אנא הוסף קובצי PDF והרץ מחדש.")
    st.stop()

# --- 4. System Prompt Configuration (Strict Grounding) ---
# Strict system prompt in Hebrew to prevent hallucinations
system_prompt = (
    "אתה עוזר מידע רשמי ומדויק של קבוצת הפניקס.\n"
    "תפקידך לענות על שאלות **אך ורק** על סמך הקטעים המצורפים מתוך המסמכים (Context).\n"
    "חוקים קשיחים:\n"
    "1. אם התשובה לא מופיעה בצורה מפורשת במידע המצורף, ענה בדיוק: 'מצטער, המידע אינו מופיע במסמכים שברשותי. מומלץ לבדוק ישירות באתר הפניקס.'\n"
    "2. אל תמציא, אל תנחש ואל תשתמש במידע חיצוני שלא סופק לך.\n"
    "3. תשובותיך חייבות להיות מקצועיות, ברורות ובעברית פשוטה.\n\n"
    "המידע מהמסמכים:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# Initialize LLM with temperature=0 for deterministic, factual outputs
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# --- 5. Chat History & User Interface ---
# Initialize session state for persistent conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages in the UI
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Process new user input
if user_query := st.chat_input("שאל שאלה על ביטוחי/מוצרי הפניקס..."):
    # Append user query to session state and render
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate model response using RAG chain
    with st.chat_message("assistant"):
        with st.spinner("מחפש במסמכי הפניקס..."):
            response = rag_chain.invoke({"input": user_query})
            answer = response["answer"]
            st.markdown(answer)
            
    # Append assistant response to session state
    st.session_state.messages.append({"role": "assistant", "content": answer})
