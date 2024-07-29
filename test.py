import os
import random
import string
import sys
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import shutil


# Set OpenAI API key
#os.environ["OPENAI_API_KEY"] = ''

def generate_random_filename(length=10):
    """Generate a random filename with the given length."""
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(length)) + '.md'

def main():
    # Generate a random filename
    random_filename = generate_random_filename()

    # Construct the command
    command = f"code2prompt --path . --suppress-comments --tokens --encoding cl100k_base --output {random_filename}"

    # Run the command
    exit_code = os.system(command)

    if exit_code == 0:
        print(f"Command executed successfully. Output saved to {random_filename}")
        
        # Load and process the data
        loader = UnstructuredMarkdownLoader(random_filename)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(data)

        # Create vectorstore and retriever 
        vectorstore = Chroma.from_documents(splits, embedding=OpenAIEmbeddings())
        
        retriever = vectorstore.as_retriever()

        # Set up the language model
        llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

        # Define the system prompt
        system_prompt = (
            "You are an advanced GitHub assistant with comprehensive knowledge of software development, version control, and project management. "
            "You have access to the context of a specific GitHub repository, including its structure, code, issues, and pull requests. "
            "Your capabilities include:\n"
            "1. Analyzing and explaining code\n"
            "2. Generating new code or suggesting code improvements\n"
            "3. Assisting with git commands and workflows\n"
            "4. Helping with project management tasks\n"
            "5. Answering questions about the repository's structure and content\n"
            "6. Providing best practices for software development and collaboration\n\n"
            "Use the following context from the GitHub repository to inform your responses:\n"
            "{context}\n\n"
            "If you're asked to generate code, provide it in appropriate markdown code blocks. "
            "If you don't have enough information to answer a question or perform a task, "
            "clearly state what additional information you need. "
            "Aim to be helpful, accurate, and concise in your responses."
        )

        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        # Create the question-answering chain
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        # Invoke the chain
        question = input("What is your question: ")
        response = rag_chain.invoke({"input": question})
        print(response["answer"])

        sys.exit(0)  # Exit with success status
    else:
        print(f"An error occurred while executing the command. Exit code: {exit_code}")
        sys.exit(1)  # Exit with error status

if __name__ == "__main__":
    main()