import os
import json
import asyncio
import logging

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from livekit.agents import function_tool, RunContext, ToolError
from langchain.text_splitter import RecursiveCharacterTextSplitter


from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RunContext,
    ToolError,
    WorkerOptions,
    cli,
    function_tool,
)
from livekit.plugins import openai


# Constants
VECTORSTORE_DIR = "faiss_index"
PDF_PATH = "Konspekt - Batyskaf.docx.pdf"

logger = logging.getLogger("pdf_agent")


def build_vectorstore_from_pdf(pdf_path: str, persist_dir: str):
    logger.info(f"Building vectorstore from: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)

    vectorstore.save_local(persist_dir)
    logger.info(f"Vectorstore saved to '{persist_dir}'")
    return vectorstore


def load_or_build_vectorstore(pdf_path: str, persist_dir: str) -> FAISS:
    if os.path.exists(os.path.join(persist_dir, "index.faiss")):
        return FAISS.load_local(persist_dir, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    else:
        return build_vectorstore_from_pdf(pdf_pah, persist_dir)


def build_qa_chain() -> RetrievalQA:
    '''
    Get vectorstore and return RetrievalQA
    '''
    vectorstore = load_or_build_vectorstore(PDF_PATH, VECTORSTORE_DIR)
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 10})

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "Odpowiedz po polsku na pytanie na podstawie poniższego dokumentu.\n"
            "Dokument: {context}\n"
            "Pytanie: {question}\n"
            "Odpowiedź:"
        )
    )

    return RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0, model_name="gpt-4o-mini"),
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )


@function_tool
async def search_docs(ctx: RunContext, question: str):
    """Tool: Search the document for a question."""
    pdf_qa_chain = await asyncio.to_thread(build_qa_chain)

    response = await asyncio.to_thread(pdf_qa_chain.invoke, {"query": question})
    logger.info(f"Docs count: {len(response['source_documents'])}")

    answer = response.get("result")

    return answer


instructions = (
    "Jesteś pomocnym asystentem. Twoim jedynym zadaniem jest odpowiadanie na pytania "
    "na podstawie informacji zawartych w załadowanym pliku PDF. "
    "Zawsze korzystaj wyłącznie z narzędzia `search_docs`, aby znaleźć odpowiedź. "
    "Nie wymyślaj odpowiedzi, jeśli nie są dostępne w dokumencie. "
    "Odpowiadaj po polsku, jasno i zwięźle."
)

async def entrypoint(ctx: JobContext):
    """Main entrypoint for LiveKit job"""
    await ctx.connect()
    
    agent = Agent(
        instructions=instructions,
        tools=[search_docs]
    )
    session = AgentSession(llm=openai.realtime.RealtimeModel())
    await session.start(agent=agent, room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
