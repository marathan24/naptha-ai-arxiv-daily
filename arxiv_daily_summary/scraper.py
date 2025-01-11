import arxiv
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

def scrape_arxiv(query: str, max_results: int = 50) -> List[Dict[str, Any]]:
    """
    Scrape arXiv for papers matching the query.
    Returns a list of dicts with 'title' and 'summary'.
    """
    try:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.LastUpdatedDate
        )
        papers = []
        for result in search.results():
            papers.append({
                "title": result.title,
                "summary": result.summary
            })
        logger.info(f"Scraped {len(papers)} papers for query: {query}")
        return papers
    except Exception as e:
        logger.error(f"Error during arXiv scraping: {e}")
        return []
