import logging
import os
import random
import asyncio
from typing import Dict, Any, List
from tqdm import tqdm

from naptha_sdk.schemas import AgentDeployment, AgentRunInput
from naptha_sdk.inference import InferenceClient
from naptha_sdk.storage.storage_provider import StorageProvider
from naptha_sdk.storage.schemas import (
    CreateStorageRequest,          
    ReadStorageRequest,
    ListStorageRequest,
    DeleteStorageRequest,
    DatabaseReadOptions
)

from arxiv_daily_summary.schemas import ArxivKBConfig
from arxiv_daily_summary.schemas import SystemPromptSchema, InputSchema
from arxiv_daily_summary.scraper import scrape_arxiv  
from arxiv_daily_summary.embedder import ArxivEmbedder  

logger = logging.getLogger(__name__)


class ArxivDailySummaryAgent:
    def __init__(self, deployment: Dict[str, Any]):
        self.deployment = deployment
        self.config = self.deployment.config  # (Agent config)
        
        # Grab the KB deployment
        kb_deployment = self.deployment.kb_deployments[0]
        # Convert the standard KBConfig → ArxivKBConfig:
        self.kb_config = ArxivKBConfig.model_validate(**kb_deployment.config.model_dump())

        self.storage_provider = StorageProvider(kb_deployment.node)
        self.storage_type = self.kb_config.storage_type
        self.table_name   = self.kb_config.path
        self.schema       = self.kb_config.schema

        # Instead of .get(...), we do dot-attribute:
        embedder_cfg = self.kb_config.embedder
        if not embedder_cfg:
            raise ValueError("No embedder config found in kb_config!")
        embedder_model = embedder_cfg.model
        
        self.embedder = ArxivEmbedder(model=embedder_model)
        self.system_prompt = SystemPromptSchema(role=self.config["system_prompt"]["role"])
        self.inference_provider = InferenceClient(self.deployment.node)


    async def run_arxiv_agent(self, module_run: AgentRunInput):
        """Entry point that selects a method based on the input."""
        inputs = InputSchema(**module_run.inputs)
        tool_name = inputs.tool_name
        tool_input_data = inputs.tool_input_data

        method = getattr(self, tool_name, None)
        if not method:
            raise ValueError(f"Invalid tool name: {tool_name}")
        return await method(tool_input_data)

    async def init(self, *args, **kwargs):
        """
        Create the database table for storing arXiv papers.
        """
        create_table_request = CreateStorageRequest(
        storage_type=self.storage_type,
        path=self.table_name,
        data={"schema": self.schema}   # store the schema under 'data'
        )

        # the storage_provider no longer has .create(), so we do:
        await self.storage_provider.execute(create_table_request)

        return {"status": "success", "message": f"Initialized table '{self.table_name}'"}

    async def add_data(self, input_data: Dict[str, Any], *args, **kwargs):
        """
        Scrape arXiv based on a given query, generate embeddings, and store the papers in the table.
        """
        query = input_data.get("query",
            "ti:Decentralized AND (abs:large language models AND abs:AI OR abs:DeFi)"
        )

        logger.info(f"Scraping arXiv for query: {query}")
        papers = scrape_arxiv(query=query, max_results=30)
        if not papers:
            return {"status": "error", "message": "No papers scraped."}

        texts = []
        meta_list = []
        for paper in papers:
            text = f"Title: {paper['title']}\nSummary: {paper['summary']}"
            texts.append(text)
            meta_list.append({"title": paper["title"], "summary": paper["summary"]})

        logger.info("Generating embeddings for the scraped texts.")
        embeddings = self.embedder.embed_batch(texts)

        documents = []
        for i, emb in enumerate(embeddings):
            doc = {
                "id": random.randint(1, 999999999),
                "title": meta_list[i]["title"],
                "summary": meta_list[i]["summary"],
                "embedding": emb,
                "metadata": {}  # Optional extra metadata
            }
            documents.append(doc)

        logger.info("Storing documents in the table.")
        for doc in tqdm(documents):
            create_row_req = CreateStorageRequest(
                storage_type=self.storage_type,
                path=self.table_name,
                data={"data": doc}  # wrap your doc
            )
            await self.storage_provider.execute(create_row_req)


        return {"status": "success", "message": f"Added {len(documents)} papers to '{self.table_name}'"}

    async def run_query(self, input_data: Dict[str, Any], *args, **kwargs):
        """
        Retrieve the top matching paper summaries using vector similarity search,
        then use the LLM to produce a consolidated answer.
        """
        query = input_data.get("query", "")
        question = input_data.get("question", "Summarize relevant papers.")
        if not query:
            return {"status": "error", "message": "No query provided."}

        logger.info("Embedding user query for vector similarity search.")
        query_embedding = self.embedder.embed_text(query)

        retriever_cfg = self.kb_config.retriever
        if retriever_cfg is not None:
            top_k = retriever_cfg.k
        else:
            top_k = 5
        read_opts = DatabaseReadOptions(
            query_vector=query_embedding,
            vector_col="embedding",
            top_k=top_k,
            include_similarity=True
        )
        read_req = ReadStorageRequest(
            storage_type=self.storage_type,
            path=self.table_name,
            options=read_opts
        )

        logger.info(f"Performing vector similarity search in table '{self.table_name}'.")
        read_result = await self.storage_provider.read(read_req)
        if not read_result:
            return {"status": "success", "message": "No matching papers found."}

        combined_summaries = []
        for row in read_result:
            text_summary = row.get('summary', '')
            combined_summaries.append(text_summary)
        joined_summaries = "\n\n".join(combined_summaries)

        messages = [
            {"role": "system", "content": self.system_prompt.role},
            {"role": "user", "content": f"The following are summaries of relevant arXiv papers:\n\n{joined_summaries}\n\nBased on these, please answer the question: {question}"}
        ]
        logger.info(f"Sending prompt to LLM: {messages}")

        llm_response = await self.inference_provider.run_inference({
            "model": self.config["llm_config"]["model"],
            "messages": messages,
            "temperature": self.config["llm_config"]["temperature"],
            "max_tokens": self.config["llm_config"]["max_tokens"]
        })

        if isinstance(llm_response, dict):
            answer = llm_response['choices'][0]['message']['content']
        else:
            answer = llm_response.choices[0].message.content

        return {"status": "success", "answer": answer}

    async def list_rows(self, input_data: Dict[str, Any], *args, **kwargs):
        """
        List a specified number of rows (default 10) from the table.
        """
        limit = input_data.get("limit", 10)
        list_req = ListStorageRequest(
            storage_type=self.storage_type,
            path=self.table_name,
            options=DatabaseReadOptions(limit=limit)
        )
        list_result = await self.storage_provider.list(list_req)
        logger.info(f"List of rows retrieved: {list_result}")
        return {"status": "success", "rows": list_result}

    async def delete_table(self, input_data: Dict[str, Any], *args, **kwargs):
        """
        Delete the entire table from storage.
        """
        table_name = input_data.get("table_name", self.table_name)
        delete_req = DeleteStorageRequest(
            storage_type=self.storage_type,
            path=table_name
        )
        del_result = await self.storage_provider.delete(delete_req)
        logger.info(f"Deleted table result: {del_result}")
        return {"status": "success", "message": f"Deleted table '{table_name}'"}


async def run(module_run: Dict[str, Any], *args, **kwargs):
    """
    Entry point for the Naptha agent run.
    Parses the input using InputSchema and calls the appropriate tool method.
    """
    module_run_input = AgentRunInput(**module_run)
    inputs = InputSchema(**module_run_input.inputs)
    agent = ArxivDailySummaryAgent(module_run_input.deployment)
    method = getattr(agent, inputs.tool_name, None)
    if not method:
        raise ValueError(f"Invalid tool name: {inputs.tool_name}")
    return await method(inputs.tool_input_data)


if __name__ == "__main__":
    from dotenv import load_dotenv
    from naptha_sdk.client.naptha import Naptha
    from naptha_sdk.configs import setup_module_deployment

    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    naptha = Naptha()

    deployment = asyncio.run(setup_module_deployment(
        "agent",
        "arxiv_daily_summary/configs/deployment.json",
        node_url=os.getenv("NODE_URL")  
    ))

    init_run = {
        "inputs": {
            "tool_name": "init",
            "tool_input_data": {}
        },
        "deployment": deployment,
        "consumer_id": naptha.user.id,
        "signature": ""  
    }
    init_response = asyncio.run(run(init_run))
    print("Init Response:", init_response)

    add_data_run = {
        "inputs": {
            "tool_name": "add_data",
            "tool_input_data": {"query": "large language models AI OR DeFi"}
        },
        "deployment": deployment,
        "consumer_id": naptha.user.id,
        "signature": ""
    }
    add_data_response = asyncio.run(run(add_data_run))
    print("Add Data Response:", add_data_response)

    query_run = {
        "inputs": {
            "tool_name": "run_query",
            "tool_input_data": {
                "query": "Trends in decentralized AI",
                "question": "What are the main themes in decentralized AI recently?"
            }
        },
        "deployment": deployment,
        "consumer_id": naptha.user.id,
        "signature": ""
    }
    query_response = asyncio.run(run(query_run))
    print("Query Response:", query_response)

    list_rows_run = {
        "inputs": {
            "tool_name": "list_rows",
            "tool_input_data": {"limit": 5}
        },
        "deployment": deployment,
        "consumer_id": naptha.user.id,
        "signature": ""
    }
    list_response = asyncio.run(run(list_rows_run))
    print("List Rows Response:", list_response)

