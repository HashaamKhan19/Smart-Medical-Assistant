import os

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

# Load the existing Chroma vector store
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

# Check if the Chroma vector store already exists
if os.path.exists(persistent_directory):
    print("Loading existing vector store...")
    db = Chroma(persist_directory=persistent_directory,
                embedding_function=None)
else:
    raise FileNotFoundError(
        f"The directory {persistent_directory} does not exist. Please check the path."
    )

# Define the embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Load the existing vector store with the embedding function
db = Chroma(persist_directory=persistent_directory,
            embedding_function=embeddings)

# This function creates a retriever object that fetches documents from the VS. 
def get_retriever(filter_by_metadata):
    return db.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 3,
            "filter": {"user": filter_by_metadata["user"]}
        }
    )

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

# Contextualize question prompt, For reformulating user questions based on chat history
contextualize_q_system_prompt = (
    "Given a patient's medical history, current symptoms, and relevant medical literature, "
    "reformulate the latest user question, which might reference context in the chat history, "
    "into a standalone question that can be understood without the chat history. "
    "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
)

# Create a prompt template for contextualizing the question
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Answering question prompt, For answering questions based on retrieved context like an assistant.
qa_system_prompt = (
    "You are an AI-powered assistant for doctors, designed to reduce the time spent searching for medical information. "
    "Use the following pieces of retrieved context, including a patient's medical history, current symptoms, "
    "and relevant medical literature, "
    "to provide diagnostic suggestions, treatment plans, or identify potential drug interactions. "
    "If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise."
    "\n\n"
    "{context}"
)

# Create a prompt template for answering questions
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

#(Question answering chain)
# Uses LLM and QA prompt to answer questions based on the retrieved documents
# create_stuff_documents_chain creates a chain for passing the retrieved documents to the LLM
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Retreival Chain with History awareness
def create_rag_chain(filter_by_metadata):
    # Creates a retriever based on metadata.
    retriever = get_retriever(filter_by_metadata)
    # Creates a history-aware retriever that uses the contextualize_q_prompt, to make the retriever aware of the chat history.
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    # Combines the history_aware_retriever and the question_answer_chain to create a RAG chain.
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Pulls a prompt from the LangChain hub for the ReAct agent.
react_docstore_prompt = hub.pull("hwchase17/react")

# Creates a tool for answering questions using the RAG chain. The tool invokes the RAG chain to generate a response.
def create_answer_tool(filter_by_metadata):
    rag_chain = create_rag_chain(filter_by_metadata)
    return Tool(
        name="Answer Question",
        func=lambda input, **kwargs: rag_chain.invoke(
            {
                "input": input, 
                "chat_history": kwargs.get("chat_history", []),
                "filter_by_metadata": filter_by_metadata
            }
        ),
        description="useful for when you need to answer questions about the context",
    )

# Create the ReAct Agent to interact with document store retriever
def create_agent(filter_by_metadata):
    tools = [create_answer_tool(filter_by_metadata)]
    return create_react_agent(llm=llm, tools=tools, prompt=react_docstore_prompt)

# Creates an agent executor that manages the execution of the agent and tools, handling errors if needed.
def create_agent_executor(filter_by_metadata):
    agent = create_agent(filter_by_metadata)
    return AgentExecutor.from_agent_and_tools(
        agent=agent, tools=[create_answer_tool(filter_by_metadata)], handle_parsing_errors=True, verbose=True
    )

# Main loop
chat_history = []
filter_by_metadata = {"user": "user-1"}  # Set the user you want to query about
agent_executor = create_agent_executor(filter_by_metadata)

while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    response = agent_executor.invoke(
        {
            "input": query,
            "chat_history": chat_history,
            "filter_by_metadata": filter_by_metadata
        }
    )
    print(f"AI: {response['output']}")

    # Update history
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=response["output"]))