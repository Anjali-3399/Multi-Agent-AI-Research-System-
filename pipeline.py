import re
import time

from agents import (
    build_search_agent,
    build_reader_agent,
    writer_chain,
    verification_chain,
    critic_chain
)


def filter_successful_sources(scraped_content: str) -> str:
    """
    Keeps only source records whose scraping status is successful.
    Failed sources must not be used as evidence or included in the report.
    """
    records = re.split(r"(?=SOURCE RECORD:)", scraped_content)
    successful_records = []

    for record in records:
        if "SOURCE RECORD:" not in record:
            continue

        success_match = re.search(
            r"(?im)^\s*(?:STATUS:\s*Successfully scraped|SCRAPING STATUS:\s*Successful)\s*$",
            record
        )

        if success_match:
            successful_records.append(record.strip())

    return "\n\n".join(successful_records)


def extract_improved_report(critic_output: str, original_report: str) -> str:
    """
    Extracts the complete improved report produced by the Critic Agent.
    The final downloaded report must contain the Critic Agent's corrections.
    """
    marker = "IMPROVED RESEARCH REPORT:"

    if marker not in critic_output:
        return original_report

    improved_report = critic_output.split(marker, 1)[1].strip()

    if not improved_report:
        return original_report

    return improved_report


def run_research_pipeline(topic: str) -> dict:
    state = {}
    start_time = time.time()

    # Step 1: Search Agent
    print("\n" + "=" * 50)
    print("Step 1 - Search Agent is working ...")
    print("=" * 50)

    search_agent = build_search_agent()

    search_result = search_agent.invoke({
        "messages": [
            (
                "user",
                f"Find recent, reliable and detailed information about: {topic}"
            )
        ]
    })

    state["search_results"] = "\n\n".join(
        message.content
        for message in search_result["messages"]
        if message.type == "tool"
    )

    print("\nSearch results:\n", state["search_results"])

    # Step 2: Reader Agent
    print("\n" + "=" * 50)
    print("Step 2 - Reader Agent is scraping multiple websites ...")
    print("=" * 50)

    reader_agent = build_reader_agent()

    urls = re.findall(r"URL:\s*(https?://\S+)", state["search_results"])
    urls = list(dict.fromkeys(urls))

    scraped_pages = []

    # Use only the first 3 URLs to reduce CPU processing time
    for url in urls[:3]:
        reader_result = reader_agent.invoke({
            "messages": [
                (
                    "user",
                    f"Scrape and analyze this webpage: {url}"
                )
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

    # Keep only successfully scraped sources
    successful_content = filter_successful_sources(state["scraped_content"])

    if not successful_content:
        state["original_report"] = ""
        state["verification"] = "ERROR: No successfully scraped source evidence available."
        state["critic_feedback"] = "ERROR: No successfully scraped source evidence available."
        state["final_report"] = ""
        state["topic"] = topic
        state["execution_time_seconds"] = round(time.time() - start_time, 2)
        state["status"] = "Failed: No successful sources"

        print("\nERROR: No successfully scraped source evidence available.")
        return state

    

     # Step 3: Writer Agent
    print("\n" + "=" * 50)
    print("Step 3 - Writer Agent is drafting the report ...")
    print("=" * 50)

    successful_content = filter_successful_sources(state["scraped_content"])

    if not successful_content:
        state["original_report"] = ""
        state["verification"] = "ERROR: No successfully scraped source evidence available."
        state["critic_feedback"] = "ERROR: No successfully scraped source evidence available."
        state["final_report"] = ""
        state["topic"] = topic
        state["execution_time_seconds"] = round(time.time() - start_time, 2)
        state["status"] = "Failed: No successful sources"

        print("\nERROR: No successfully scraped source evidence available.")
        return state

    research_combined = (
        f"SUCCESSFULLY SCRAPED SOURCE EVIDENCE:\n{successful_content}"
    )

    state["original_report"] = writer_chain.invoke({
        "topic": topic,
        "research": research_combined
    })

    print("\nOriginal Report:\n", state["original_report"])

    # Step 4: Verification Agent
    print("\n" + "=" * 50)
    print("Step 4 - Verification Agent is verifying report claims ...")
    print("=" * 50)

    state["verification"] = verification_chain({
        "report": state["original_report"],
        "research": successful_content
    })

    print("\nSource Verification:\n", state["verification"])

    # Step 5: Critic Agent
    print("\n" + "=" * 50)
    print("Step 5 - Critic Agent is reviewing and correcting the report ...")
    print("=" * 50)

    try:
        state["critic_feedback"] = critic_chain.invoke({
            "report": state["original_report"],
            "verification": state["verification"]
        })

        print("\nCritic Review:\n", state["critic_feedback"])

    except Exception as error:
        state["critic_feedback"] = f"ERROR: Critic Agent failed. {error}"
        state["final_report"] = state["original_report"]
        state["topic"] = topic
        state["execution_time_seconds"] = round(time.time() - start_time, 2)
        state["status"] = "Failed at Critic Agent"

        print("\nCritic Agent Error:", error)
        return state

    # Step 6: Generate final corrected report
    print("\n" + "=" * 50)
    print("Step 6 - Generating final corrected report ...")
    print("=" * 50)

    state["final_report"] = extract_improved_report(
        state["critic_feedback"],
        state["original_report"]
    )

    print("\nFinal Corrected Report:\n", state["final_report"])

    state["topic"] = topic
    state["execution_time_seconds"] = round(time.time() - start_time, 2)
    state["status"] = "Success"

    return state


if __name__ == "__main__":
    topic = input("\nEnter a research topic: ")
    result = run_research_pipeline(topic)

    print("\nExecution time:", result["execution_time_seconds"], "seconds")
    print("Status:", result["status"])