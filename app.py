import os
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- 1. Design and interface settings (Streamlit) ---
st.set_page_config(page_title="בוט מידע - קבוצת הפניקס", page_icon="🏢", layout="centered")

st.title("🏢 בוט המידע הרשמי של קבוצת הפניקס")
st.caption("מערכת מענה אוטומטית המבוססת אך ורק על מסמכי החברה הרשמיים")

# Enter the API key in the sidebar
with st.sidebar:
    st.header("הגדרות מערכת")
    api_key = st.text_input("הזן OpenAI API Key:", type="password")
    if not api_key:
        st.warning("יש להזין מפתח API כדי להפעיל את הבוט.")
        st.stop()
    os.environ["OPENAI_API_KEY"] = api_key

# --- 2. Load files and create the vector store (RAG Engine) ---
@st.cache_resource(show_spinner="מעבד את מסמכי הפניקס...")
def init_rag_system():
    # A. Load all PDF files from the data directory
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()
    
    if not documents:
        return None

    # B. Split the text into focused chunks (chunking)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    splits = text_splitter.split_documents(documents)

    # C. Convert the text to vectors and store them in a local database (Chroma)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    
    return vectorstore.as_retriever(search_kwargs={"k": 3})

# Ensure that a directory named data exists
if not os.path.exists("data"):
    os.makedirs("data")
    st.info("נוצרה תיקייה בשם 'data'. אנא שמי בה את קבצי ה-PDF של הפניקס והריצי מחדש.")
    st.stop()

retriever = init_rag_system()

if retriever is None:
    st.warning("תיקיית 'data' ריקה. אנא הוסיפי קבצי PDF והרעי מחדש.")
    st.stop()

# --- 3. Define the system prompt (to prevent hallucinations) ---
system_prompt = (
    "אתה עוזר מידע רשמי ומדויק של קבוצת הפניקס.\n"
    "תפקידך לענות על שאלות **אך ורק** על סמך הקטעים المצורפים מתוך המסמכים (Context).\n"
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

# Create the RAG engine
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # temperature=0 for maximum accuracy
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# --- 4. Conversation management and chat interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display the message history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Get a question from the user
if user_query := st.chat_input("שאלי שאלה על ביטוחי/מוצרי הפניקס..."):
    # Display the user's question
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate the answer
    with st.chat_message("assistant"):
        with st.spinner("מחפש במסמכי הפניקס..."):
            response = rag_chain.invoke({"input": user_query})
            answer = response["answer"]
            st.markdown(answer)
            
    # Save the answer to the history
    st.session_state.messages.append({"role": "assistant", "content": answer})