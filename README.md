AI Agent CSM

This a project with the goal of developing an AI agent specialized in computational soft matter research. 

This agent will have dual functions:

To perform in-depth literature reviews, identifying cutting-edge developments and emerging trends in the field.

To assist researchers by suggesting innovative research problems, grounded in the latest academic findings.

To be equipped to autonomously execute and manage complex simulations, thereby accelerating the research process and fostering groundbreaking discoveries in computational soft matter.

This Streamlit app allows users to upload files, process them, run a LAMMPS simulation, and chat with an AI agent. The app utilizes various libraries and tools, including Streamlit, PyPDF2, docx, subprocess, OpenAI, Hugging Face, and more.

Setup

1. Install the required packages:
pip install streamlit PyPDF2 python-docx langchain streamlit-chat python-dotenv

2. Ensure you have the necessary APIs and keys for OpenAI and Hugging Face.

3. Make sure you have installed on your system the Large-scale Atomic/Molecular Massively Parallel Simulator (LAMMPS) program
   
Running the App

1. Run the Streamlit app using:
streamlit run main.py

2. Upload your files (PDF or DOCX).

3. Add your OpenAI API key and process the files.

4. Ask questions about the files or run a LAMMPS simulation.

File Processing
- PDF files are processed using PyPDF2 to extract text.
- DOCX files are processed using the python-docx library to extract text.
- CSV files are currently not supported.
- Text Chunking
- Text is split into chunks to improve processing efficiency. This is done using a custom CharacterTextSplitter class.

Vector Store Creation
- A vector store is created from the text chunks using Hugging Face embeddings and FAISS.

Conversation Chain
- A conversation chain is created using a combination of a vector store and an OpenAI API key. The chain allows for conversational interactions with the AI agent.

LAMMPS Simulation
- Paste your LAMMPS input script into the provided text area.
- Click "Run Simulation" to execute the LAMMPS simulation using subprocess.
  
Chat with AI Agent
- Ask questions about the uploaded files using the chat input.
- The AI agent will provide responses based on the content of the files and previous interactions.

