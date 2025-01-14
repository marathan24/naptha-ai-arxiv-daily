import logging
import os
import random
import asyncio
from typing import Dict, Any, List
from tqdm import tqdm

from naptha_sdk.inference import InferenceClient
from naptha_sdk.storage.storage_provider import StorageProvider
from naptha_sdk.storage.schemas import (
    CreateStorageRequest,
    ReadStorageRequest,
    ListStorageRequest, # not used for now
    DeleteStorageRequest,
    StorageType
)
from naptha_sdk.schemas import AgentRunInput

from arxiv_daily_summary.schemas import (
    ArxivStorageConfig, 
    SystemPromptSchema,
    InputSchema
)
from arxiv_daily_summary.scraper import scrape_arxiv
from arxiv_daily_summary.embedder import ArxivEmbedder

logger = logging.getLogger(__name__)

class ArxivDailySummaryAgent:
    def __init__(self, deployment: Dict[str, Any]):
        self.deployment = deployment
        self.config = deployment.config
        if hasattr(self.config, 'model_dump'):
            config_dict = self.config.model_dump() 
        
        system_prompt_config = config_dict.get('system_prompt', {})
        llm_config = config_dict.get('llm_config', {})

        kb_deployment = self.deployment.kb_deployments[0]
        
        # Hard coding this as I was facing "'KBConfig' object has no attribute 'embedder'" continuously and hence I have decided to move ahead with this approach
        kb_config_dict = {
            "storage_type": kb_deployment.config.storage_type.value,  
            "path": kb_deployment.config.path,
            "schema": kb_deployment.config.schema,
            # Hardcoding embedder and retriever config for now
            # Extract embedder config
            "embedder": {
              "model": "text-embedding-3-small",
              "chunk_size": 1000,
              "chunk_overlap": 200,
              "separators": ["\n\n", "\n", ". ", " ", ""],
              "embedding_dim": 1536
          },
            # Extract retriever config
            "retriever": {
              "type": "vector",
              "field": "embedding",
              "k": 5
          }
        }
        
        self.storage_config = ArxivStorageConfig(**kb_config_dict)
        self.storage_provider = StorageProvider(kb_deployment.node)
        self.embedder = ArxivEmbedder(model=self.storage_config.embedder.model)
        
        self.system_prompt = SystemPromptSchema(
            role=system_prompt_config.get('role', "You are a helpful research assistant.")
        )
        self.llm_config = llm_config
        self.inference_provider = InferenceClient(self.deployment.node)

    async def run_arxiv_agent(self, module_run: AgentRunInput):
        """
        Main entry point for agent operations.
        
        Args:
            module_run: Contains the function name and input data
        """
        inputs = InputSchema(**module_run.inputs)
        method = getattr(self, inputs.tool_name, None)
        if not method:
            raise ValueError(f"Invalid tool name: {inputs.tool_name}")
        return await method(inputs.tool_input_data)

    async def init(self, *args, **kwargs):
        """Initialize the vector database table for storing papers"""
        try:
            create_request = CreateStorageRequest(
                storage_type=StorageType.DATABASE,
                path=self.storage_config.path,
                data={"schema": self.storage_config.schema}
            )
            
            result = await self.storage_provider.execute(create_request)
            logger.info(f"Table creation result: {result}")
            return {"status": "success", "message": f"Initialized table '{self.storage_config.path}'"}
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            return {"status": "error", "message": f"Failed to initialize: {str(e)}"}

    async def add_data(self, input_data: Dict[str, Any], *args, **kwargs):
        try:
            query = input_data.get("query")
            logger.info(f"Fetching papers with query: {query}")
            papers = scrape_arxiv(query=query, max_results=20)
            
            if not papers:
                return {"status": "error", "message": "No papers found"}
            
            documents = []
            for paper in papers:
                try:
                    text = f"Title: {paper['title']}\nSummary: {paper['summary']}"
                    embedding = self.embedder.embed_text(text) or [0] * 1536
                    
                    doc = {
                        "data": {  
                            "id": random.randint(1, 999999999),
                            "title": paper["title"],
                            "summary": paper["summary"],
                            "embedding": embedding,
                            "metadata": {"source": "arxiv", "query": query}
                        }
                    }
                    documents.append(doc)
                except Exception as e:
                    logger.warning(f"Error processing paper: {str(e)}")
                    continue
            
            for doc in tqdm(documents, desc="Storing papers"):
                try:
                    create_request = CreateStorageRequest(
                        storage_type=StorageType.DATABASE,
                        path=self.storage_config.path,
                        data=doc
                    )
                    await self.storage_provider.execute(create_request)
                except Exception as e:
                    logger.error(f"Error adding document: {str(e)}")
                    continue
            
            return {"status": "success", "message": f"Added {len(documents)} papers"}
        except Exception as e:
            logger.error(f"Error adding papers: {str(e)}")
            return {"status": "error", "message": str(e)}
        
    async def run_query(self, input_data: Dict[str, Any], *args, **kwargs):
        """
        Search papers and generate an analysis.
        
        input_data: Contains search query and specific question
        """
        try:
            query = input_data.get("query", "")
            question = input_data.get("question", "Summarize relevant papers.")
            
            if not query:
                return {"status": "error", "message": "Query is required"}
                
            logger.info("Generating query embedding")
            query_embedding = self.embedder.embed_text(query)
            
            # Using just ReadStorageRequest, need to use DatabaseReadOptions too in future for retrieval (Getting http errors for DatabaseReadOptions hence removed it)
            read_options = {
                "columns": ["title", "summary", "embedding"],
                "limit": 20
            }
            
            read_request = ReadStorageRequest(
                storage_type=StorageType.DATABASE,
                path=self.storage_config.path,
                options=read_options
            )
            
            results = await self.storage_provider.execute(read_request)
            logger.info(f"Simple query result: {results.data}")
            logger.info(f"Type of data result : {type(results.data)}")

            summaries = []
            for i, result in enumerate(results.data):
                summary = f"Paper {i+1}:\nTitle: {result.get('title')}\n{result.get('summary')}"
                summaries.append(summary)
            
            context = "\n\n".join(summaries)

            messages = [
                {"role": "system", "content": self.system_prompt.role},
                {"role": "user", "content": (
                    f"Based on these research papers:\n\n{context}\n\n"
                    f"Please answer the following question: {question}"
                )}
            ]

            llm_response = await self.inference_provider.run_inference({
                "model": self.llm_config.get("model", "gpt-4o-2024-11-20"),
                "messages": messages,
                "temperature": self.llm_config.get("temperature", 0.1),
                "max_tokens": self.llm_config.get("max_tokens", 2000)
            })
            
            answer = llm_response.choices[0].message.content

            return {
                "status": "success", 
                "answer": answer,
                "metadata": {
                    "papers_analyzed": len(results.data),
                    "query": query,
                    "question": question
                },
                
            }            
            
        except Exception as e:
            logger.error(f"Error in simple query: {str(e)}")
            return None

        
    async def delete_table(self, input_data: Dict[str, Any], *args, **kwargs):
        """Delete the storage table (Not used for now)"""
        try:
            table_name = input_data.get("table_name", self.storage_config.path)
            delete_request = DeleteStorageRequest(
                storage_type=StorageType.DATABASE,
                path=table_name
            )
            await self.storage_provider.execute(delete_request)
            return {"status": "success", "message": f"Deleted table '{table_name}'"}
        except Exception as e:
            logger.error(f"Error deleting table: {str(e)}")
            return {"status": "error", "message": str(e)}

async def run(module_run: Dict[str, Any], *args, **kwargs):
    """
    Main entry point for the agent.
    
    Args:
        module_run: Contains agent configuration and run parameters
    """
    try:
        module_run_input = AgentRunInput(**module_run)
        inputs = InputSchema(**module_run_input.inputs)
        
        agent = ArxivDailySummaryAgent(module_run_input.deployment)
        method = getattr(agent, inputs.tool_name, None)
        
        if not method:
            raise ValueError(f"Invalid tool name: {inputs.tool_name}")
        
        return await method(inputs.tool_input_data)
    except Exception as e:
        logger.error(f"Error in agent run: {str(e)}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    from dotenv import load_dotenv
    from naptha_sdk.client.naptha import Naptha
    from naptha_sdk.configs import setup_module_deployment
    from naptha_sdk.user import sign_consumer_id

    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    naptha = Naptha()

    deployment = asyncio.run(setup_module_deployment(
        "agent",
        "arxiv_daily_summary/configs/deployment.json",
        node_url=os.getenv("NODE_URL")
    ))

    test_runs = [
        {
            "name": "Initialize",
            "inputs": {
                "tool_name": "init",
                "tool_input_data": {}
            }
        },
        {
            "name": "Add Data",
            "inputs": {
                "tool_name": "add_data",
                "tool_input_data": {
                    "query": "AI Agents Decentralized payments"
                }
            }
        },
        {
            "name": "Query",
            "inputs": {
                "tool_name": "run_query",
                "tool_input_data": {
                    "query": "AI Agents Decentralized payments",
                    "question": "Exaplin all those papers which you find relevant to Blockchain or decentralized payments"
                }
            }
        },
        {
            "name": "Delete Table",
            "inputs": {
                "tool_name": "delete_table",
                "tool_input_data": {

                }
            }
        }
    ]

    for test_run in test_runs:
        logger.info(f"\nExecuting {test_run['name']}...")
        run_config = {
            "inputs": test_run["inputs"],
            "deployment": deployment,
            "consumer_id": naptha.user.id,
            "signature": sign_consumer_id(naptha.user.id, os.getenv("PRIVATE_KEY"))
        }
        
        try:
            response = asyncio.run(run(run_config))
            logger.info(f"{test_run['name']} Response: {response}")
        except Exception as e:
            logger.error(f"Error in {test_run['name']}: {str(e)}")