import os

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()


# Load the exising Chroma vector store
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

# Retreiver for querying the vector store
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
    # search_kwargs=additional arguments for the search (e.g., number of results to return)
)

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")


# Contextualize question prompt, helps the AI understand that it should reformulate the question
# based on the chat history to make it a standalone question

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

# History-aware retriever, component that combines the functionality of a retriever with the
# capability to consider the conversation history when reformulating user queries
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Answering question prompt to help the AI understand that it should provide concise answers based on the retrieved context
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

# Create a chain to combine documents for question answering
# `create_stuff_documents_chain` feeds all retrieved context into the LLM
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)


# Create a retrieval chain that combines the history-aware retriever and the question answering chain
rag_chain = create_retrieval_chain(
    history_aware_retriever, question_answer_chain)


# agent, think about stuff, take action, make observations, in a loop
react_docstore_prompt = hub.pull("hwchase17/react")


tools = [
    Tool(
        name="Answer Question",
        func=lambda input, **kwargs: rag_chain.invoke(
            {
                "input": input, 
                "chat_history": kwargs.get("chat_history", []),
                "filter_by_metadata": kwargs.get("filter_by_metadata", {})
            }
        ),
        description="useful for when you need to answer questions about the context",
    )
]

# Create the ReAct Agent with document store retriever
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_docstore_prompt,
)


# agent_executor = AgentExecutor.from_agent_and_tools(
#     agent=agent, tools=tools, handle_parsing_errors=True, verbose=True,
# )
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, handle_parsing_errors=True,
)

chat_history = []
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    response = agent_executor.invoke(
        {
            "input": query,
            "chat_history": chat_history,
            "filter_by_metadata": {"user": "user-1"}  # Add desired metadata filter here
        }
    )
    print(f"AI: {response['output']}")

    # Update history
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=response["output"]))