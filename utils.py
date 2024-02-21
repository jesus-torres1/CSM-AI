import os
from PyPDF2 import PdfReader
import docx
import openai
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI

def get_files_text(uploaded_files):
    """
    Gets text from uploaded files and concatenate them into a single text string.

    Args:
        uploaded_files (list): List of uploaded files.

    Returns:
        str: Concatenated text from all uploaded files.
    """
    text = ""
    for uploaded_file in uploaded_files:
        split_tup = os.path.splitext(uploaded_file.name)
        file_extension = split_tup[1]
        if file_extension == ".pdf":
            text += get_pdf_text(uploaded_file)
        elif file_extension == ".docx":
            text += get_docx_text(uploaded_file)
        elif file_extension == ".csv":
            text += get_csv_text(uploaded_file)
        else:
            # I placed this handle for unsupported file types or other extensions to add in the future
            pass
    return text

def get_pdf_text(pdf):
    """
    Extracts text from a PDF file.

    Args:
        pdf (file): PDF file object.

    Returns:
        str: Extracted text from the PDF file.
    """
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_docx_text(file):
    """
    Extracts text from a DOCX file.

    Args:
        file (file): DOCX file object.

    Returns:
        str: Extracted text from the DOCX file.
    """
    doc = docx.Document(file)
    allText = []
    for docpara in doc.paragraphs:
        allText.append(docpara.text)
    text = ' '.join(allText)
    return text

def get_csv_text(file):
    """
    Extracts text from a CSV file.

    Args:
        file (file): CSV file object.

    Returns:
        str: Extracted text from the CSV file.
    """
    csv_text = ""
    csv_reader = csv.reader(file)
    for row in csv_reader:
        csv_text += ",".join(row) + "\n"
    return csv_text

def get_text_chunks(text):
    """
    Split text into chunks.

    Args:
        text (str): Input text.

    Returns:
        list: List of text chunks.
    """
    # To split the text into chuncks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=900,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    """
    Creates a vector store from text chunks.

    Args:
        text_chunks (list): List of text chunks.

    Returns:
        VectorStore: Vector store containing embeddings for the text chunks.
    """
    embeddings = HuggingFaceEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    knowledge_base = FAISS.from_texts(text_chunks,embeddings)
    return knowledge_base

def get_conversation_chain(vetorestore,openai_api_key):
    """
    Creates a conversation chain using a vector store and OpenAI API.

    Args:
        vetorestore (VectorStore): Vector store containing text embeddings.
        openai_api_key (str): OpenAI API key.

    Returns:
        ConversationalRetrievalChain: Conversation chain for chatbot interactions.
    """
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name = 'gpt-3.5-turbo',temperature=0)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vetorestore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_user_input(user_question):
    """
    Handles user input and generates a response from the chatbot.

    Args:
        user_question (str): User's question.

    Returns:
        None
    """
    if "generate LAMMPS script" or "generate lammps script" in user_question.lower():
        # Will need to further develop the prompt (Future Work)
        prompt = """
        Generate a LAMMPS input script for a simulation based on the request description.
        
        User Input: {user_input}

        {missing_data_prompt} 
        """

        # I added this to request a completion from the OpenAI API
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=200,
            stop=None
        )

        # For extracting the generated LAMMPS script from the response
        lammps_script = response.choices[0].text.strip()

        # This is to update the Streamlit app's state to set the LAMMPS script in the text area
        st.session_state.lammps_script = lammps_script

        # To display a message to the user indicating that the LAMMPS script has been generated
        st.info("LAMMPS script generated. You can now run the simulation.")

    elif "run LAMMPS simulation" or "run lammps simulation" in user_question.lower():
        if "lammps_script" in st.session_state:
            # Saves the LAMMPS input script to a file
            with open("lammps_input.in", "w") as f:
                f.write(st.session_state.lammps_script)

            # Runs the LAMMPS simulation
            st.info("Running LAMMPS simulation...")
            process = subprocess.Popen(["lammps", "-in", "lammps_input.in"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()

            if process.returncode != 0:
                st.error(f"Error running LAMMPS simulation:\n{stderr.decode('utf-8')}")
            else:
                st.success("LAMMPS simulation completed successfully.")
        else:
            st.warning("No LAMMPS script generated yet. Please generate the script before running the simulation.")

    else:
        # For handling other inputs here if needed in the future
        pass

    with get_openai_callback() as cb:
        response = st.session_state.conversation({'question': user_question})
        chat_history = st.session_state.chat_history or []
        chat_history.append(response)
        st.session_state.chat_history = chat_history

    # Layout of input/response containers
    response_container = st.container()

    with response_container:
        for i, messages in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                message(messages.content, is_user=True, key=str(i))
            else:
                message(messages.content, key=str(i))
        st.write(f"Total Tokens: {cb.total_tokens}" f", Prompt Tokens: {cb.prompt_tokens}" f", Completion Tokens: {cb.completion_tokens}" f", Total Cost (USD): ${cb.total_cost}")

