import streamlit as st
import os
import requests
import PyPDF2
from io import BytesIO
import glob
import re
import langchain
import sqlite3
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain import OpenAI

# Replace 'your_openai_api_key' with your actual OpenAI API key
openai_api_key = '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'
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Load documents and create vector store and retrieval QA objects
loader = DirectoryLoader('Extractedpapers', glob="**/*.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=2500, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Load documents and create vector store and retrieval QA objects
try:
    vecstore = Chroma.from_documents(texts, embeddings)
except sqlite3.OperationalError as e:
    # Drop the existing table and recreate the vector store
    if(vecstore):
        vecstore.drop_table()
    else:
        vecstore = Chroma.from_documents(texts, embeddings)

qa = RetrievalQA.from_chain_type(
    llm=OpenAI(openai_api_key=openai_api_key),
    chain_type='stuff',
    retriever=vecstore.as_retriever()
)
def query(q):
    st.write("Query:", q)
    st.write("Answer:", qa.run(q))


# Function to download PDF from the URL and extract text
def organize_text(input_file_path, output_file_path):
    # Read the text from the input file
    with open(input_file_path, 'r', encoding='utf-8') as file:
        extracted_text = file.read()

    # Convert all text to lowercase
    extracted_text = extracted_text.lower()

    # Define regular expressions to identify key elements in the text
    title_pattern = r'^[a-z][a-z0-9\s:]*$'  # Matches lines starting with lowercase letters, numbers, and colons
    author_pattern = r'^[a-z][a-z\s]+\n'  # Matches lines with author names in lowercase
    email_pattern = r'[a-z0-9_.+-]+@[a-z0-9-]+\.[a-z0-9-.]+'  # Matches email addresses
    affiliation_pattern = r'^[a-z\s]+\n'  # Matches lines with affiliations

    # Initialize variables to store organized information
    title = None
    authors = []
    emails = []
    affiliations = []
    introduction = None
    keywords = None
    other_text = []

    # Process each line of the extracted text
    for line in extracted_text.splitlines():
        line = line.strip()

        # Check for title match
        if not title and re.match(title_pattern, line):
            title = line
        # Check for author match
        elif re.match(author_pattern, line):
            authors.append(line)
        # Check for email match
        elif re.search(email_pattern, line):
            emails.append(re.findall(email_pattern, line)[0])
        # Check for affiliation match
        elif re.match(affiliation_pattern, line):
            affiliations.append(line)
        # Check for introduction match
        elif not introduction and line.lower().startswith("introduction"):
            introduction = line
        # Check for keywords match
        elif not keywords and line.lower().startswith("keywords"):
            keywords = line.replace("keywords:", "").strip()
        else:
            other_text.append(line)

    # Format the organized information as desired
    organized_text = f"Title: {title}\n"
    organized_text += "Authors: " + " ".join(authors) + "\n"
    organized_text += "Emails: " + " ".join(emails) + "\n"
    organized_text += "Affiliations: " + " ".join(affiliations) + "\n"
    if introduction:
        organized_text += f"\nIntroduction: {introduction}\n"
    if keywords:
        organized_text += f"\nKeywords: {keywords}\n"
    organized_text += "\nOther Text:\n" + " ".join(other_text)

    # Write the organized text to the output file
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.write(organized_text)
def process_article(url):
    response = requests.get(url)
    pdf_io = BytesIO(response.content)
    pdf = PyPDF2.PdfReader(pdf_io)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text
def organize_articles(txt_files, output_directory):
    for input_file_path in txt_files:
        filename = os.path.basename(input_file_path)
        output_file_path = os.path.join(output_directory, filename)
        organize_text(input_file_path, output_file_path)
        st.write(f"'{filename}' has been organized.")
def empty_output_folder(output_folder):
    for file_path in glob.glob(os.path.join(output_folder, '*.txt')):
        os.remove(file_path)

# Set Streamlit page configuration
st.set_page_config(
    page_title="Article Processing App",
    page_icon="📚",
    layout="wide"
)


def main():
    st.title("Article Processing App")
    # Get input from the user (article URLs)
    st.header("Enter Article URLs:")
    user_input = st.text_area("Paste the article URLs here, one URL per line.")

    # Process the articles if the user provided URLs
    if st.button("Process Articles"):
        # Split the user input into a list of URLs
        paper_urls = user_input.strip().split('\n')

        # Folder name where the papers will be stored
        output_folder = "Extractedpapers"

        # Create the output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        else:
            # Empty the output folder if it already exists
            empty_output_folder(output_folder)

        # Download and process papers
        if len(paper_urls) != 0:
            for idx, url in enumerate(paper_urls, 1):
                st.write(f"Downloading and processing Paper {idx}...")
                paper_text = process_article(url)

                # Write the paper text to a file in the output folder
                file_path = os.path.join(output_folder, f"paper_{idx}.txt")
                with open(file_path, "w", encoding="utf-8") as file:
                    file.write(paper_text)
                    st.write(f"Paper {idx} processed successfully.")
        else:
            st.write("No URLs provided.")
        
        # Get the list of all processed text files
        txt_files = glob.glob(os.path.join(output_folder, '*.txt'))

        # Organize the extracted text if articles were processed
        if len(txt_files) != 0:
            st.write("Organizing the extracted text...")
            organize_articles(txt_files, output_folder)

        else:
            st.write("No articles were processed.")

    # Add Q&A functionality
    st.header("Query and Answer:")
    user_query = st.text_input("Ask a question about the processed articles:")
    if st.button("Ask"):
        query(user_query)

if __name__ == "__main__":
    main()
