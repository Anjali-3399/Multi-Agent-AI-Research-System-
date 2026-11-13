from agents import (
    build_search_agent,
    build_reader_agent,
    writer_chain,
    verification_chain,
    critic_chain
)
from langchain_core.messages import HumanMessage
from tools import scrape_url

def run_research_pipeline(topic: str)-> dict:
    
    state ={}
    
    #search agent working
    print("\n"+" =*50")
    print("step 1- search agent is working ...")
    print("="*50)
    
    search_agent = build_search_agent()

    search_result = search_agent.invoke({
    "messages": [
        ("user", f"Find recent, reliable and detailed information about: {topic}")
    ]
})
    print("\nDEBUG - Search Agent messages:\n", search_result["messages"]) #for testing and debugging

    state["search_results"] = "\n\n".join(
    msg.content
    for msg in search_result["messages"]
    if msg.type == "tool"
)

    print("\nSearch results:", state["search_results"])
    

    
        
    
    

    # Step 2: Reader agent working
    print("\n" + "=" * 50)
    print("Step 2 - Reader agent is scraping multiple websites ...")
    print("=" * 50)

    reader_agent = build_reader_agent()

    # Extract URLs from search results
    import re

    urls = re.findall(r"URL:\s*(https?://\S+)", state["search_results"])

    # Remove duplicate URLs
    urls = list(dict.fromkeys(urls))

    scraped_pages = []

    for url in urls[:5]:
        reader_result = reader_agent.invoke({
            "messages": [
                ("user", f"Scrape and analyze this webpage: {url}")
            ]
        })

        scraped_pages.append(
    f"""
SOURCE RECORD:
Requested URL: {url}

READER OUTPUT:
{reader_result["messages"][-1].content}
"""
)

    state["scraped_content"] = "\n\n".join(scraped_pages)

    print("\nScraped content:\n", state["scraped_content"])
           
    

    #step 3 - writer chain 

    
    print("\n"+" ="*50)
    print("step 3 - Writer is drafting the report ...")
    print("="*50)

    research_combined = (
        f"SEARCH RESULTS : \n {state['search_results']} \n\n"
        f"DETAILED SCRAPED CONTENT : \n {state['scraped_content']}"
    )

    state["report"] = writer_chain.invoke({
        "topic" : topic,
        "research" : research_combined
    })

    print("\n Final Report\n",state['report'])
    
    
    # Step 4: Source verification
    

    print("Step 4 - Verifying report sources ...")
    print("=" * 50)

    # Debug: Check verification input size
    print("\nDEBUG - Verification Input Size:")
    print("Report characters:", len(state["report"]))
    print("Scraped content characters:", len(state["scraped_content"]))
    print(
        "Total characters:",
        len(state["report"]) + len(state["scraped_content"])
    )

    state["verification"] = verification_chain({
    "report": state["report"],
    "research": state["scraped_content"]
})
    

    print("\nSource Verification:\n", state["verification"])

    #critic report 

    print("\n"+" ="*50)
    print("step 4 - critic is reviewing the report ")
    print("="*50)

    state["feedback"] = critic_chain.invoke({
    "report": state["report"],
    "verification": state["verification"]
})

    print("\n critic report \n", state['feedback'])

    return state



if __name__ == "__main__":
    topic = input("\n Enter a research topic : ")
    run_research_pipeline(topic)
    
    