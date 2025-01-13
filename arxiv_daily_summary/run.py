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
    ListStorageRequest,
    DeleteStorageRequest,
    DatabaseReadOptions,
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
        
        # Hard coding this as I was facing some error continuously and hence I have decided to move ahead with this approach
        kb_config_dict = {
            "storage_type": kb_deployment.config.storage_type.value,  
            "path": kb_deployment.config.path,
            "schema": kb_deployment.config.schema,
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
            query = input_data.get("query", "ti:Decentralized AND (abs:large language models)")
            logger.info(f"Fetching papers with query: {query}")
            papers = scrape_arxiv(query=query, max_results=30)
            
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
        """
        try:
            query = input_data.get("query", "")
            question = input_data.get("question", "Summarize relevant papers.")
            
            if not query:
                return {"status": "error", "message": "Query is required"}

            logger.info("Generating query embedding")
            query_embedding = self.embedder.embed_text(query)
            
            try:
                db_read_options = DatabaseReadOptions(
                    columns=["title", "summary"],
                    query_vector=query_embedding,
                    vector_col="embedding",
                    top_k=self.storage_config.retriever.k,
                    include_similarity=True
                )
                logger.info(f"Created read options: {db_read_options.model_dump()}")
            except Exception as e:
                logger.error(f"Error creating read options: {str(e)}")
                raise

            try:

                read_request = ReadStorageRequest(
                    storage_type=StorageType.DATABASE,
                    path=self.storage_config.path,
                    options=db_read_options.model_dump()
                )
                logger.info(f"Created read request: {read_request.model_dump()}")
            except Exception as e:
                logger.error(f"Error creating read request: {str(e)}")
                raise

            # Everything going right till here and the issue comes after this code below
            try:
                results = await self.storage_provider.execute(read_request)
                logger.info(f"Got results: {results}")
            except Exception as e:
                logger.error(f"Error executing storage request: {str(e)}")
                raise

            if not results or not results.data:
                return {"status": "success", "message": "No matching papers found"}


        except Exception as e:
            logger.error(f"Error in query: {str(e)}", exc_info=True)
            return {"status": "error", "message": str(e)}


    async def delete_table(self, input_data: Dict[str, Any], *args, **kwargs):
        """Delete the storage table"""
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
                    "query": "large language models AI OR DeFi"
                }
            }
        },
        {
            "name": "Query",
            "inputs": {
                "tool_name": "run_query",
                "tool_input_data": {
                    "query": "Trends in decentralized AI",
                    "question": "What are the main themes in decentralized AI recently?"
                }
            }
        },
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