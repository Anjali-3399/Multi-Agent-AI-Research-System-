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
writer_prompt=ChatPromptTemplate.from_messages([
  ("system", """You are a strict evidence-based research writer.
Use ONLY facts explicitly supported by the provided research.
Never invent facts, statistics, dates, authors, studies, DOIs, or URLs.
If a claim cannot be verified from the research, omit it or label it "Not verified".
Use only URLs explicitly present in the research.
Every source in the Sources section must use an exact URL found in the provided research.
Never invent citations, author names, publication years, DOIs
, or article titles. If source details are unavailable, list only the verified URL.
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

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional."""),
])

writer_chain = writer_prompt| llm |StrOutputParser()


# Source verification chain

verification_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a strict claim-level source verification agent.

Your ONLY task is to verify factual claims already written in the Research Report.

STRICT RULES:

CLAIM EXTRACTION:
1. Extract factual claims ONLY from the Research Report.
2. Never extract claims from Scraped Source Evidence.
3. Never create, expand, split, duplicate, or rewrite report claims.
4. Every original factual claim must appear exactly once.
5. Ignore headings, source lists, URLs, and general instructions.
6. Ignore statements describing webpage limitations or scraping results.
7. Preserve each extracted claim's original wording.

SOURCE MATCHING:
8. Treat every SOURCE RECORD as an independent source.
9. Match a claim against evidence belonging to that exact source record.
10. Never transfer evidence between different source URLs.
11. Use only successfully scraped EVIDENCE sections.
12. Never use LIMITATIONS sections as supporting evidence.
13. Never use search snippets, outside knowledge, assumptions, or guesses.
14. A related topic or general definition does not prove a specific claim.
15. Check all available source records before deciding the status.

VERIFICATION STATUS:
16. Supported: Direct evidence supports the complete claim.
17. Partially Supported: Evidence supports only part of the claim.
18. Unsupported: No direct supporting evidence exists.
19. If evidence is insufficient or ambiguous, do not mark Supported.
20. Never invent evidence or source URLs.

OUTPUT RULES:
21. Give every original factual claim exactly one verification entry.
22. Do not include any new factual claims in the verification output.
23. Do not add market statistics, examples, dates, or facts.
24. Keep the original claim wording unchanged.
25. Summary counts must match the individual claim statuses.

Return EXACTLY this format:

CLAIM 1:
Claim: [Exact original claim]
Supporting Source URL: [Exact URL or None]
Evidence: [Exact relevant evidence or No supporting evidence found.]
Verification Status: Supported / Partially Supported / Unsupported
Reason: [Brief explanation]

Repeat for each original factual claim.

OVERALL VERIFICATION SUMMARY:
- Total claims checked:
- Supported:
- Partially Supported:
- Unsupported:

IMPORTANT:
The number of verification entries must equal the number of factual claims extracted from the Research Report.
Never verify claims found only in the scraped sources.
Never include source limitations as report claims.
"""),

    ("human", """Research Report:
{report}

Scraped Source Evidence:
{research}

Verify the original report claims only. Follow the exact output format.""")
])


verification_chain = verification_prompt | llm | StrOutputParser()
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
8. Do not invent facts, statistics, dates, examples, or sources.

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
