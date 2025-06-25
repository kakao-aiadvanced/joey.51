import os
import streamlit as st
from pprint import pprint
from typing import List, TypedDict

from google.colab import userdata

from tavily import TavilyClient

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langgraph.graph import END, StateGraph


tavily = TavilyClient(api_key='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')
llm = ChatOpenAI(model="gpt-4o-mini", temperature = 0)

st.set_page_config(
    page_title="Research Assistant",
    page_icon=":orange_heart:",
)

### Index

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
# docs>sublist>item
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)

doc_splits = text_splitter.split_documents(docs_list)

# Add to vectorDB
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
)
retriever = vectorstore.as_retriever()

### Relevance Checker


system = """You are a grader assessing relevance
    of a retrieved document to a user question. If the document contains keywords related to the user question,
    grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
    """

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "question: {question}\n\n document: {document} "),
    ]
)

retrieval_grader = prompt | llm | JsonOutputParser()

### Generate

system = """You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise.
    context를 참고할 땐 그 출처를 꼭 명시해. 출처는 metatdata.source를 확인하면 돼.
    """

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "question: {question}\n\n context: {context} "),
    ]
)

# Chain
rag_chain = prompt | llm | StrOutputParser()

### Hallucination Grader

system = """You are a grader assessing whether
    an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate
    whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a
    single key 'score' and no preamble or explanation."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "documents: {documents}\n\n answer: {generation} "),
    ]
)

hallucination_grader = prompt | llm | JsonOutputParser()



### State
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    question: str
    generation: str
    web_search: str
    documents: List[str]
    retry: int


### Nodes

def retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    print(question)
    print(documents)
    return {"documents": documents, "question": question, "retry": 0}

def generate(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    retry = state["retry"]        
    return {"documents": documents, "question": question, "generation": generation, "retry": retry + 1}

def web_search(state):
    """
    Web search based based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    print("---WEB SEARCH---")
    print(state)
    question = state["question"]

    # Web search
    docs = tavily.search(query=question)['results']
    web_results = []
    for d in docs:
      wd = Document(page_content=d["content"], metadata={"source": d["url"]})
      web_results.append(wd)
      
    print("########## web_results")
    print(web_results)

    retry = state["retry"]        
    return {"documents": web_results, "question": question, "retry": retry + 1}

### Conditional edge
def relevance_checker(state):
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---RELEVANCE CHECKER---")
    question = state["question"]
    documents = state["documents"]
    retry = state["retry"]

    if retry > 1:
      raise ValueError("failed: not relevant")
    
    # Score each doc
    filtered_docs = []
    web_search = "No"

    for d in documents:
      result = retrieval_grader.invoke(
          {"question": question, "document": d.page_content}
      )
      score = result["score"]

      if score.lower() == "yes":
          print("---GRADE: DOCUMENT RELEVANT---")
          filtered_docs.append(d)
      else:
          print("---GRADE: DOCUMENT NOT RELEVANT---")
    
    print("########### filtered_docs")
    print(filtered_docs)
    if len(filtered_docs) > 0:
      return "generate"
    else:
      return "websearch"
    
def hallucination_checker(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    retry = state["retry"]

    if retry > 2:
      raise ValueError("failed: hallucination")

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score["score"]

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION ADDRESSES QUESTION---")
        return "useful"
    else:
        pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"


# Build graph

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("generate", generate)  # generatae
workflow.add_node("websearch", web_search)  # web search

workflow.set_entry_point("retrieve")

workflow.add_conditional_edges(
    "retrieve",
    relevance_checker,
    {
        "generate": "generate",
        "websearch": "websearch"
    },
)

workflow.add_conditional_edges(
    "websearch",
    relevance_checker,
    {
        "generate": "generate",
        "websearch": "websearch"
    },
)

workflow.add_conditional_edges(
    "generate",
    hallucination_checker,
    {
        "useful": END,
        "not supported": "generate"
    },
)


# Compile
app = workflow.compile()

# ----------------------------------------------------------------------
# Streamlit 앱 UI
st.title("Research Assistant powered by OpenAI")

input_topic = st.text_input(
    "질문을 입력하세요.",
    value="Superfast Llama 3 inference on Groq Cloud",
)

generate_report = st.button("답해줘")

if generate_report:
    with st.spinner("생성중"):
        inputs = {"question": input_topic}
        for output in app.stream(inputs):
            for key, value in output.items():
                print(f"Finished running: {key}:")
        final_report = value["generation"]
        st.markdown(final_report)

st.sidebar.markdown("---")
if st.sidebar.button("Restart"):
    st.session_state.clear()
    st.experimental_rerun()

