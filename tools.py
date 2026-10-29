from langchain.tools import tool
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from tavily import TavilyClient
import os
from dotenv import load_dotenv
from rich import print
load_dotenv()

tavily=TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

#making a function for creating a tool
@tool
def web_search(query: str) -> str:
    """ Search the web for recent and reliable information on a topic.Returns Titles,URLs and snippets """
    
    result = tavily.search(query=query,max_results=5)
    
    out = []
    
    for r in result['results']:
        out.append(
            f"Title: {r['title']}\nURL: {r['url']}\nSnippet: {r['content'][:300]}\n"
        )
    
    return "\n----\n".join(out)

#print(web_search.invoke("what is the recent news of the war!"))

#for beautiful soup
@tool
def scrape_url(url: str) -> str:
    """Scrape a webpage and return its URL, status, and clean text."""
    try:
        parsed = urlparse(url)

        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            return "ERROR: Invalid URL."

        resp = requests.get(
            url,
            timeout=8,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)[:3000]

        if not text:
            return f"ERROR: No readable text found.\nURL: {url}"

        return f"URL: {resp.url}\nSTATUS: Successfully scraped\nCONTENT:\n{text}"

    except Exception as e:
        return f"ERROR: Could not scrape URL: {url}\nDETAILS: {str(e)}"

    
#print(scrape_url.invoke("https://www.icc-cricket.com/about/cricket/history-of-cricket/early-cricket"))