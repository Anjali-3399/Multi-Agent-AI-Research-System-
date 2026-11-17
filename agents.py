from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search, scrape_url
import os
from dotenv import load_dotenv

load_dotenv()

#model setup
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
    
)



#1st agent
def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search]
    )
    
    
    
#2nd agent

def build_reader_agent():
    reader_prompt = """
    
You are an evidence-based research reader.

1. Use scrape_url to read the provided webpage.
2. Preserve the exact source URL returned by the tool.
3. Report only facts explicitly supported by the scraped webpage.
4. Do not add facts from your own knowledge.
5. Do not invent statistics, dates, names, or citations.
6. If scraping fails, clearly report the error.
7. Return the result in this format:

SOURCE URL:
[Exact URL returned by the tool]

SCRAPING STATUS:
[Successful or Failed]

EVIDENCE:
- Fact 1 copied or faithfully extracted from the webpage.
- Fact 2 copied or faithfully extracted from the webpage.

LIMITATIONS:
[State what could not be verified]
"""
    

    return create_agent(
        model=llm,
        tools=[scrape_url],
        system_prompt=reader_prompt
    )
  
#writer chain

writer_prompt = ChatPromptTemplate.from_messages([
  ("system", """You are a strict evidence-based research writer.

Use ONLY the successfully scraped source evidence provided.
Do not use search snippets, unscraped URLs, YouTube pages, or outside knowledge.
Do not name authors, years, organizations, or publication details unless they appear directly in the scraped evidence.
Do not convert a cited reference into the author of the article.
If a source is not successfully scraped, do not include it in the report or Sources section.
If a claim cannot be supported by the scraped evidence, omit it or label it "Not verified".

Use only URLs explicitly present in the successfully scraped source evidence.
Every source in the Sources section must use an exact URL found in the provided research.
Never invent citations, author names, publication years, DOIs, or article titles.
If source details are unavailable, list only the verified URL.
Do not invent source titles or publication details.

Clearly distinguish evidence from general explanation.
If the research is insufficient, state that limitation honestly.

A source may be included in the Sources section only if:
- Its scraping status is Successful.
- Its evidence supports at least one claim in the report.

Do not include sources that provide no relevant evidence.
Do not use search snippets as proof when scraped evidence is available.
Do not make conclusions that go beyond the provided evidence."""),
  ("human", """Write a research report on this topic:

Topic: {topic}

Successfully Scraped Source Evidence:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list only successfully scraped URLs that support claims in the report)

Be detailed, factual, professional, and concise."""),
])
writer_chain = writer_prompt | llm | StrOutputParser()


# Source verification chain

import re


# 1. Extract factual statements from the original report
def extract_report_claims(report: str) -> list:

    claims = []

    # Ignore everything from Sources section onward
    report = re.split(
        r"(?im)^\s*(?:#{1,6}\s*)?\*{0,2}Sources\*{0,2}\s*$",
        report
    )[0]

    for line in report.splitlines():

        line = line.strip()

        if not line:
            continue

        # Remove Markdown headings
        line = re.sub(r"^#{1,6}\s*", "", line)

        # Skip headings
        if re.match(
            r"^\*{0,2}(Introduction|Key Findings|Conclusion)\*{0,2}$",
            line,
            re.IGNORECASE
        ):
            continue

        # Remove bullet or numbered-list markers
        line = re.sub(r"^[-*]\s+", "", line)
        line = re.sub(r"^\d+\.\s+", "", line)

        # Remove bold Markdown formatting
        line = line.replace("**", "")

        # Remove introductory labels before colon
        # Example: Definition of AI: AI is ...
        if ":" in line:
            possible_heading, remaining = line.split(":", 1)

            if len(possible_heading.split()) <= 8:
                line = remaining.strip()

        # Split sentences
        sentences = re.split(
            r"(?<=[.!?])\s+(?=[A-Z])",
            line
        )

        for sentence in sentences:

            sentence = sentence.strip()

            if len(sentence.split()) < 5:
                continue

            if sentence not in claims:
                claims.append(sentence)

    return claims


# 2. Keep only successfully scraped source evidence
def extract_successful_sources(research: str) -> str:

    source_records = re.split(
        r"(?=SOURCE RECORD:)",
        research
    )

    valid_sources = []

    for record in source_records:

        if "SOURCE RECORD:" not in record:
            continue

        status_success = re.search(
            r"(?im)^\s*(?:STATUS:\s*Successfully scraped|SCRAPING STATUS:\s*Successful)\s*$",
            record
        )

        if not status_success:
            continue

        url_match = re.search(
            r"(?im)^\s*(?:SOURCE URL|Requested URL):\s*(https?://\S+)",
            record
        )

        evidence_match = re.search(
            r"(?is)EVIDENCE:\s*(.*?)(?=\n\s*LIMITATIONS:|$)",
            record
        )

        if not url_match or not evidence_match:
            continue

        url = url_match.group(1).strip()
        evidence = evidence_match.group(1).strip()

        if not evidence:
            continue

        valid_sources.append(
            f"SOURCE URL: {url}\nEVIDENCE:\n{evidence}"
        )

    return "\n\n".join(valid_sources)


# 3. Claim-level verification prompt
verification_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a strict source verification agent.

Your task is to verify the exact claims supplied in the CLAIMS section.

STRICT RULES:

1. Verify ONLY the claims supplied in CLAIMS.
2. Never create new claims.
3. Never split, merge, rewrite, or duplicate claims.
4. Every supplied claim must appear exactly once.
5. Use only successfully scraped source evidence.
6. Match evidence only with its own source URL.
7. Never transfer evidence between sources.
8. Never use failed sources, limitations, snippets, or outside knowledge.
9. Supported means direct evidence supports the complete claim.
10. Partially Supported means evidence supports only part of the claim.
11. Unsupported means no direct supporting evidence exists.
12. If evidence is insufficient, mark Unsupported.
13. Never invent URLs or evidence.
14. Do not claim that a statement is verified by multiple sources unless each source independently supports it.
15. Use exact source URLs.

For every claim, return:

CLAIM:
[Exact supplied claim]

Supporting Source URL:
[Exact URL or None]

Evidence:
[Relevant evidence or No supporting evidence found.]

Verification Status:
Supported / Partially Supported / Unsupported

Reason:
[Brief explanation]

After all claims, return:

OVERALL VERIFICATION SUMMARY:
- Total claims checked:
- Supported:
- Partially Supported:
- Unsupported:

The number of verification entries must exactly equal the number of supplied claims."""),
    ("human", """CLAIMS:
{claims}

SUCCESSFULLY SCRAPED SOURCE EVIDENCE:
{research}

Verify every supplied claim exactly once.""")
])

verification_llm_chain = verification_prompt | llm | StrOutputParser()


# 4. Verification function used by pipeline.py
def verification_chain(inputs: dict) -> str:

    claims = extract_report_claims(inputs["report"])

    valid_sources = extract_successful_sources(
        inputs["research"]
    )

    formatted_claims = "\n".join(
        f"CLAIM {index}: {claim}"
        for index, claim in enumerate(claims, start=1)
    )

    print("\nDEBUG - Extracted claims:", len(claims))
    print("DEBUG - Valid source evidence characters:", len(valid_sources))

    if not claims:
        return "ERROR: No factual claims extracted from the report."

    if not valid_sources:
        return "ERROR: No successfully scraped source evidence available."

    return verification_llm_chain.invoke({
        "claims": formatted_claims,
        "research": valid_sources
    })



#critic_chain
# Critic Agent - Review and Improve Report

critic_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a strict research critic and report improvement agent.

Your job is to evaluate and improve the research report using the source verification results.

RULES:
1. Evaluate the original research report and verification results together.
2. Identify unsupported and partially supported claims.
3. Remove factual claims that are unsupported by the verification results.
4. Rewrite partially supported claims cautiously without adding new facts.
5. Improve clarity, grammar, structure, and readability.
6. Remove unnecessary repetition.
7. Improve the introduction and conclusion using only information already supported by the report and verification.
8.Do not add new examples, applications, statistics, facts, or generalizations.
9.Use only information already supported by the original report and verification results.
10.If a claim is unsupported, remove it instead of replacing it with a new claim.

Return your response in this exact format:

CRITIC REVIEW:

Score: X/10

Strengths:
- ...

Issues Identified:
- ...

Improvements Made:
- ...

IMPROVED RESEARCH REPORT:
[Write the complete revised research report here.]"""),

    ("human", """Original Research Report:
{report}

Source Verification Results:
{verification}

Evaluate the report and produce the improved version.""")
])

critic_chain= critic_prompt|llm|StrOutputParser()
