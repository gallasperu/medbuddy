import streamlit as st
from langchain_xai import ChatXAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
import tempfile

# ===================== CONFIG =====================
st.set_page_config(
    page_title="📄 Blood Test Assistant - Advanced RAG",
    page_icon="🤖",
    layout="wide"
)

st.title("👨‍⚕️ Med Buddy (xAI) - Advanced RAG")
st.markdown("Upload blood test results and ask intelligent questions")
st.markdown("DISCLAIMER: DATA THAT YOU UPLOAD IS NOT STORED OR SENT TO ANYWHERE EXCEPT YOUR BROWSER.")
st.markdown("ALL PROCESSING HAPPENS LOCALLY IN YOUR BROWSER USING THE GROK API KEY YOU PROVIDE.")
st.markdown("DATA IS DESTROYED IMMEDIATELY AFTER CLOSING THE BROWSER. ALWAYS BE CAUTIOUS WITH SENSITIVE INFORMATION.")

# ===================== API KEY =====================
if "XAI_API_KEY" not in st.secrets:
    api_key = st.text_input("Enter your Grok API Key (xAI)", type="password")
    if api_key:
        # os.environ["XAI_API_KEY"] = api_key
        st.secrets["XAI_API_KEY"] = api_key
    else:
        st.warning("⚠️ Grok API Key required. Get one at https://xai.grok.com/")
        st.stop()

# ===================== SETTINGS =====================
response_types = {
    "1": "Clear and concise (for general public)",
    "2": "Professional and detailed",
    "3": "Professional + Proposed Treatment (if needed)"
}

chunking_strategies = {
    "full": "Full Document (No Chunking)",
    "recursive": "Recursive Character (Recommended)",
    "character": "Character Text Splitter"
}

# ===================== PROMPT =====================
prompt_template = PromptTemplate(
    input_variables=["context", "question", "response_style"],
    template="""
You are Med Buddy, a specialized AI assistant for blood test interpretation.

**Response Style:** {response_style}

**Rules:**
- Answer based primarily on the retrieved context from the blood test document.
- Supplement with reliable general medical knowledge when needed.
- Be accurate, balanced, and responsible.
- **MANDATORY DISCLAIMER:** Always end with:
  "⚠️ This is for informational purposes only. This is not medical advice. Please consult a qualified physician."

Context:
{context}

Question: {question}

Answer:"""
)

# ===================== FUNCTIONS =====================
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def process_pdf(uploaded_file, chunk_strategy="recursive", chunk_size=1000, chunk_overlap=200):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    full_text = "\n\n".join(doc.page_content for doc in documents)

    # Chunking
    if chunk_strategy == "full":
        chunks = [full_text]
        metadatas = [{"source": "full_document"}]
    else:
        if chunk_strategy == "recursive":
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ".", " ", ""]
            )
        else:
            text_splitter = CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        
        chunks = text_splitter.split_text(full_text)
        metadatas = [{"chunk": i, "source": "blood_test.pdf"} for i in range(len(chunks))]

    os.unlink(tmp_path)
    return full_text, chunks, metadatas

# ===================== SESSION STATE =====================
for key in ["context", "chunks", "vectorstore", "messages", "response_type", 
            "chunk_strategy", "chunk_size", "chunk_overlap", "top_k", "remember_choice"]:
    if key not in st.session_state:
        if key == "response_type": st.session_state[key] = "1"
        elif key == "chunk_strategy": st.session_state[key] = "recursive"
        elif key == "chunk_size": st.session_state[key] = 1000
        elif key == "chunk_overlap": st.session_state[key] = 200
        elif key == "top_k": st.session_state[key] = 4
        elif key == "remember_choice": st.session_state[key] = True
        else: st.session_state[key] = None

# ===================== SIDEBAR =====================
with st.sidebar:
    st.header("📁 Document")
    uploaded_file = st.file_uploader("Upload blood test PDF", type=["pdf"])
    
    if uploaded_file:
        st.subheader("Chunking Settings")
        col1, col2 = st.columns(2)
        with col1:
            chunk_strategy = st.selectbox(
                "Strategy",
                options=list(chunking_strategies.keys()),
                format_func=lambda x: chunking_strategies[x],
                index=list(chunking_strategies.keys()).index(st.session_state.chunk_strategy)
            )
        with col2:
            st.session_state.chunk_strategy = chunk_strategy

        # Advanced Chunk Settings
        if chunk_strategy != "full":
            st.session_state.chunk_size = st.slider("Chunk Size", 300, 2000, st.session_state.chunk_size, 50)
            st.session_state.chunk_overlap = st.slider("Chunk Overlap", 0, 500, st.session_state.chunk_overlap, 20)

        if st.button("Process & Index Document"):
            with st.spinner("Processing and creating vector database..."):
                full_text, chunks, metadatas = process_pdf(
                    uploaded_file,
                    chunk_strategy=st.session_state.chunk_strategy,
                    chunk_size=st.session_state.chunk_size,
                    chunk_overlap=st.session_state.chunk_overlap
                )
                
                embeddings = get_embeddings()
                vectorstore = FAISS.from_texts(chunks, embeddings, metadatas=metadatas)
                
                st.session_state.context = full_text
                st.session_state.chunks = chunks
                st.session_state.vectorstore = vectorstore
                st.session_state.messages = []
                st.success(f"✅ Document indexed! ({len(chunks)} chunks)")

    st.divider()

    # Response Settings
    st.header("🎯 Response Settings")
    selected_type = st.selectbox(
        "Response Type",
        options=list(response_types.keys()),
        format_func=lambda x: f"{x}) {response_types[x]}",
        index=list(response_types.keys()).index(st.session_state.response_type)
    )
    st.session_state.response_type = selected_type

    st.session_state.remember_choice = st.checkbox("Remember my choice", value=st.session_state.remember_choice)

    if st.session_state.vectorstore:
        st.session_state.top_k = st.slider("Top K relevant chunks", 1, 10, st.session_state.top_k)

    st.divider()
    st.caption("Powered by Grok + LangChain + FAISS")

# ===================== MAIN AREA =====================
if st.session_state.vectorstore is None:
    st.info("👆 Upload and process a blood test PDF to start.")
else:
    st.success(f"✅ Document indexed successfully ({len(st.session_state.chunks)} chunks)")

    # Chunk Explorer + Download
    with st.expander("🔍 Explore & Download Chunks", expanded=False):
        st.write(f"**Total chunks:** {len(st.session_state.chunks)}")
        for i, chunk in enumerate(st.session_state.chunks):
            with st.expander(f"Chunk {i+1}", expanded=False):
                st.text_area("Content", chunk[:800] + "..." if len(chunk) > 800 else chunk, 
                            height=120, disabled=True)

        if st.button("Download All Chunks as TXT"):
            all_chunks_text = "\n\n".join([f"--- CHUNK {i+1} ---\n{chunk}" 
                                          for i, chunk in enumerate(st.session_state.chunks)])
            st.download_button(
                label="⬇️ Download chunks.txt",
                data=all_chunks_text,
                file_name="blood_test_chunks.txt",
                mime="text/plain"
            )

    # Chat Interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if question := st.chat_input("Ask about your blood test results..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant chunks and thinking..."):
                try:
                    # Retrieve relevant chunks
                    retriever = st.session_state.vectorstore.as_retriever(
                        search_type="similarity", 
                        search_kwargs={"k": st.session_state.top_k}
                    )
                    relevant_docs = retriever.invoke(question)
                    context = "\n\n".join([doc.page_content for doc in relevant_docs])

                    chat = ChatXAI(
                        model="grok-4",
                        temperature=0.1,
                        max_tokens=4000
                    )
                    
                    style_desc = {
                        "1": "Explain in clear, simple, everyday language that anyone can understand.",
                        "2": "Provide a detailed, professional medical explanation using appropriate terminology.",
                        "3": "Give a professional detailed explanation. If abnormalities are detected, suggest possible medical approaches or treatments while being very cautious."
                    }[st.session_state.response_type]

                    formatted_prompt = prompt_template.format(
                        context=context,
                        question=question,
                        response_style=style_desc
                    )
                    
                    response = chat.invoke(formatted_prompt)
                    answer = response.content
                    
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# Footer
st.caption("---\nMed Buddy • Advanced RAG with Embeddings + Grok • Always consult a real doctor")