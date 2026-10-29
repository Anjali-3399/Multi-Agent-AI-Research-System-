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
If the research is insufficient, state that limitation honestly."""),
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

#critic_chain
critic_prompt=ChatPromptTemplate.from_messages([
  ("system", "you are a sharp and constructive research critic. Be honest and specific."),
  ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to improve:
- ...
- ...

One line verdict:
..."""),
])

critic_chain= critic_prompt|llm|StrOutputParser()
