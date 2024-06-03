import csv
import google.generativeai as genai
import os
import time
import pandas as pd

from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.base import VectorStoreRetriever
from pathlib import Path
from typing import Generator, Optional
from sklearn.metrics import confusion_matrix

# Load environmental variables from .env
load_dotenv()

# Authenticate genai configuration for python session
genai.configure(api_key=os.getenv("GOOGLE_GEMINI_API_KEY"))

# USE THIS EMBEDDING FUNCTION THROUGHOUT THIS FILE
embedding_function = HuggingFaceEmbeddings(
    model_name=os.getenv("EMBEDDING_MODELS"),
    # This continued to error in Windows until i pointed it to my user profile .cache folder
    cache_folder=os.getenv("HUG_CACHE")
)


# Chunk the pdfs by every five files
def chunk_pdfs(pdfs_gen: Generator, chunk_size: int = 5) -> list[list]:
    """
    Breaks a generator of PDF documents into chunks of specified size.

    :param pdfs_gen: A generator object yielding PDF documents.
    :param chunk_size: (int, optional): The number of PDFs in each chunk. Defaults to 5.

    :return: list[list[PDF]]: A list of lists, where each inner list represents a chunk of PDFs.
    """
    pdfs_list = list(pdfs_gen)  # Convert generator to list for easier chunking
    return [pdfs_list[i: i + chunk_size] for i in range(0, len(pdfs_list), chunk_size)]


def remove_db_file() -> None:
    """
    Deletes the Faiss index files (index.faiss and index.pkl) from the designated directory.

    This function is useful for resetting the index data or freeing up storage space.
    """
    fasiss_dir = os.getenv("FAISS_DIR")
    try:
        os.remove(fasiss_dir + "index.faiss")
        os.remove(fasiss_dir + "index.pkl")
    except FileNotFoundError:
        print(f"One or both index files not found in {fasiss_dir}")


def get_retriever() -> VectorStoreRetriever:
    """
    Loads a Faiss vector store from the local directory and initializes a VectorStoreRetriever.

    The retriever allows you to search the vector store for relevant documents based on embeddings.

    :return: vectorstoreretriever: a retriever object for document retrieval.
    """
    fasiss_dir = os.getenv("FAISS_DIR")
    db = FAISS.load_local(fasiss_dir, embedding_function,
                          index_name="index", allow_dangerous_deserialization=True)  # Load Faiss index
    retriever = VectorStoreRetriever(vectorstore=db, search_kwargs={"k": 20})  # Retrieve top 20 documents
    return retriever


def generate_response(prompt: str, model: str, temperature: float = 0, second_try: bool = False) -> str:
    """
    Generates a response from a language model for a given prompt.

    :param model:  the large langauge model in use (gemini-pro)
    :param temperature: controls randomness in generation (0 is deterministic)
    :param second_try: whether this is a retry attempt

    :returns: The generated response from the model.
    """

    response = "If this shows, something went wrong while generating a response from google genai."
    if second_try:
        time.sleep(3)  # Wait for 3 seconds before retrying

    try:
        # model_name = model  (This line seems redundant)
        genai.GenerationConfig(temperature=temperature)
        response = genai.GenerativeModel(model_name=model).generate_content(prompt).text  # removed model_name
    except Exception as e:
        print(f"API call failed. Error message: {e}")  # Log error for debugging
        if not second_try:
            print("Retrying in 3 seconds...")
            return generate_response(prompt, model, temperature, second_try=True)  # Recursive retry

    return response


def create_knowledge_base(report_paths: list) -> None:
    """
    Creates a knowledge base for a chatbot by processing and indexing PDF reports.

    This function performs the following steps:
    1. Loads PDF reports from the specified paths.
    2. Splits the reports into smaller chunks of text for efficient processing.
    3. Computes embedding vectors for each text chunk using an embedding function.
    4. Builds a FAISS index to store the embeddings for fast similarity search.
    5. Saves the FAISS index to a local directory.

    :param report_paths: A list of file paths to the PDF reports.
    """

    # Convert report paths to a list if it's a generator
    report_paths = list(report_paths)

    # Get the directory for processed documents
    doc_dir = Path(os.getenv("PROCESSED_DOCUMENTS_DIR"))  # Assumes environment variable is set
    print(f"Loading documents from {doc_dir}")  # Log loading process

    # Load PDF documents
    docs = []  # Initialize empty list for storing loaded documents
    for report_path in report_paths:
        print(f"Loading {report_path}")  # Log each file being loaded
        loader = UnstructuredPDFLoader(report_path)
        docs.extend(loader.load())

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
    docs = text_splitter.split_documents(docs)
    print(f"Created {len(docs)} documents")  # Log number of chunks

    # Extract text and metadata from documents
    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]

    print("""
        Computing embedding vectors and building FAISS db.
        WARNING: This may take a long time. 
        You may want to consider optimizing this process for large datasets.
    """)  # Updated warning message

    # Create FAISS index from texts and metadata
    db = FAISS.from_texts(texts, embedding_function, metadatas=metadatas)

    # Save the FAISS index locally
    db.save_local(os.getenv("FAISS_DIR"))  # Assumes environment variable is set

    print(f"FAISS VectorDB has {db.index.ntotal} documents")  # Log number of indexed documents


def get_precision_recall_f1(actual: list[str], predicted: list[str]) -> None:
    """
        Calculates the precision, recall, and f1_score for model performance

        :param actual: list of actual labels
        :param predicted: list of predicted labels from the model
    """
    # Unique labels
    labels = list(set(actual + predicted))

    # Calculate confusion matrix
    cm = confusion_matrix(actual, predicted, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print("Confusion Matrix:\n", cm_df)

    # Calculate metrics for each class
    metrics = {}
    for label in labels:
        tp = cm_df.loc[label, label]  # True positives
        fp = cm_df[label].sum() - tp  # False positives
        fn = cm_df.loc[label].sum() - tp  # False negatives

        # Avoid division by zero
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

        metrics[label] = {"Precision": precision, "Recall": recall, "F1-Score": f1}

    # Create results DataFrame
    results_df = pd.DataFrame(metrics).transpose()

    # Print the results
    print("\nMetrics per Class:")
    print(results_df.to_markdown(numalign='left', stralign='left', floatfmt='.2f'))


def generate_genai_response(model: str, gemini_prompt: str, retriever: Optional[VectorStoreRetriever] = None,
                            retriever_prompt: str = "", temperature: float = 1, use_retriever: bool = True) -> str:
    """
    Generates a response to a user query using a knowledge base and a language model.

    This function performs the following steps:

    1. Retrieves relevant documents from the knowledge base based on the query.
    2. Concatenates the content of the relevant documents into a single string.
    3. Constructs a prompt for the language model by combining the query and the relevant documents.
    4. Generates a response using the specified language model and temperature.

    :model: the name of the language model to use (gemini-pro)
    :gemini_prompt: the user's query.
    :retriever: a retriever object to search the knowledge base
    :retriever_prompt: the prompt used to guide the retrieval process
    :temperature (float, optional): The temperature parameter, controls randomness
    :use_

    :return: the generated response from the language model.
    """

    if use_retriever:
        # Retrieve the most relevant documents from the knowledge base based on the retriever_prompt
        relevant_docs = retriever.get_relevant_documents(retriever_prompt, k=10)

        # Combine the content of the relevant documents into a single string
        relevant_docs_str = ""
        for doc in relevant_docs:
            relevant_docs_str += doc.page_content + "\n\n"

        # Create a prompt for the language model by combining the query and the relevant documents
        prompt_full = f"""Answer based on the following context: 
            {relevant_docs_str}
            Question: {gemini_prompt}"""
    else:
        prompt_full = f"""Answer based on the following context: 
                    {retriever_prompt}
                    Question: {gemini_prompt}"""

    # Generate the response using the specified model and temperature
    response = generate_response(prompt_full, model=model, temperature=temperature)

    return response


def clean_and_format_table(table_str: str) -> str:
    """
    Cleans, formats, and standardizes a table represented as a string.

    This function performs the following steps:
    1. Splits the input table string into lines and removes empty lines.
    2. Uses `csv.reader` to parse the lines into rows, handling '|' as the delimiter.
    3. Cleans each field in every row by stripping leading/trailing whitespace and surrounding it with double quotes.
    4. Joins the cleaned fields with commas to create a standard CSV format.
    5. Reassembles the formatted rows back into a single string.

    :table_str: the raw table string to be processed.

    :return:The cleaned and formatted table string in CSV format.
    """

    # Remove empty lines and leading/trailing whitespace from each line
    lines = [line.strip() for line in table_str.split("\n") if line.strip()]

    # Parse the lines into rows using the pipe delimiter
    rows = csv.reader(lines, delimiter='|')

    # Clean and format each field in every row
    formatted_rows = []
    for row in rows:
        formatted_row = [f'"{field.strip()}"' for field in row]  # Add quotes and strip whitespace
        formatted_rows.append(",".join(formatted_row))  # Join fields with commas

    # Combine the formatted rows into a single string
    return "\n".join(formatted_rows)


def data_retriever_main() -> None:
    """
    Extracts building data from PDF reports using a FAISS knowledge base and GEMINI language model.

    This function iterates over chunks of PDFs, performs the following steps for each chunk:
    1. Creates a knowledge base using the `create_knowledge_base` function.
    2. Initializes a retriever for searching the knowledge base.
    3. Formulates prompts for retrieving relevant documents and extracting structured information.
    4. Generates a response using a language model based on the prompts.
    5. Cleans and formats the response into a CSV-like format.
    6. Appends the formatted response to a summary file.

    """
    report_docs = Path(os.getenv("REPORTS_DOCUMENTS_DIR"))
    pdfs = report_docs.glob("*.pdf")
    chunks = chunk_pdfs(pdfs)
    summary_file = Path(os.getenv("SUMMARY_FILE"))

    retriever_prompt = ("""Retrieve building address, borough, building identification number, control number, 
    Exterior wall type, Exterior wall materials, and summary of building .""")

    prompt = ("""Return a csv format separated by "|" as delimeter for building address, borough, building 
    identification number, control number, Exterior wall type, Exterior wall materials. The building identification 
    number or BIN is a 7 digit number and number starts numbers 1-5. The control number is a six digit number that 
    starts with the cycle number 9. Make sure to format in to six columns""")
    temperature = int(os.getenv("TEMPERATURE"))

    for chunk in chunks:
        create_knowledge_base(chunk)
        retriever = get_retriever()

        output_response = generate_genai_response(
            model=os.getenv("MODEL"), gemini_prompt=prompt, retriever=retriever, retriever_prompt=retriever_prompt,
            temperature=temperature
        )

        cleaned_response = clean_and_format_table(output_response)

        with open(summary_file, 'a') as f:  # Open file in append mode
            f.write(cleaned_response + "\n")  # Append cleaned output


def fisp_preditor_dumb():
    labels = ["safe", "swarmp", "unsafe"]
    max_length = 30000

    predict_docs = Path(os.getenv("SAMPLES_DOCUMENTS_DIR"))
    summary_file = Path(os.getenv("SUMMARY_FILE"))
    pdfs = predict_docs.glob("*.pdf")

    prompt = ("""Return a response of on 2 things separated by a "|" symbol of the control number, and a prediction 
    if the building FISP status is either [SAFE, SWARMP or UNSAFE] based of the information provided""")
    # temperature = int(os.getenv("TEMPERATURE"))

    for pdf_path in pdfs:
        loader = UnstructuredPDFLoader(pdf_path)
        document = loader.load()

        # Process document content (lowercase, remove labels, truncate)
        document_content = document[0].page_content.lower()
        for label in labels:
            document_content = document_content.replace(label, "")
        document_content = document_content[:max_length]

        output_response = generate_genai_response(model=os.getenv("MODEL"),
                                                  gemini_prompt=prompt,
                                                  retriever_prompt=document_content,
                                                  temperature=2,
                                                  use_retriever=False)

        with open(summary_file, 'a') as f:  # Open file in append mode
            f.write(output_response + "\n")  # Append cleaned output


def fisp_preditor_with_retriever():
    # LABELS = ["safe", "swarmp", "unsafe"]
    # MAX_LENGTH = 10000

    predict_docs = Path(os.getenv("SAMPLES_DOCUMENTS_DIR"))
    summary_file = Path(os.getenv("SUMMARY_FILE"))
    predict_pdfs = predict_docs.glob("*.pdf")

    report_docs = Path(os.getenv("REPORTS_DOCUMENTS_DIR"))
    report_pdfs = list(report_docs.glob("*.pdf"))
    create_knowledge_base(report_pdfs)
    retriever = get_retriever()
    temperature = int(os.getenv("TEMPERATURE"))

    for pdf_path in predict_pdfs:
        filename = os.path.basename(pdf_path).split("-")[1]
        create_knowledge_base([pdf_path])

        retriever_prompt = "Building conditions for " + filename
        prompt = (f"""Pretend you are an architect examining FISP report details provided. Please only return the 
        prediction of {filename}. Is it [SAFE, SWARMP or UNSAFE]""")

        output_response = generate_genai_response(model=os.getenv("MODEL"),
                                                  gemini_prompt=prompt,
                                                  retriever=retriever,
                                                  retriever_prompt=retriever_prompt,
                                                  temperature=temperature)

        with open(summary_file, 'a') as f:  # Open file in append mode
            f.write(filename + "," + output_response + "\n")  # Append cleaned output

        time.sleep(5)


if __name__ == "__main__":
    data_retriever_main()
    # fisp_preditor_dumb()
    # fisp_preditor_with_retriever()
    # csv_file = pd.read_csv(filepath_or_buffer="predictions_with_retriever.csv",sep=",")
    # get_precision_recall_f1(list(csv_file["actual"]), list(csv_file["predicted"]))
    # csv_file = pd.read_csv(filepath_or_buffer="predictions_no_retriever.csv",sep=",")
    # get_precision_recall_f1(list(csv_file["actual"]), list(csv_file["predicted"]))
