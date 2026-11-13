from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import time

# Basic LLM without web retrieval
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

baseline_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a research report writer.

Write a clear and structured research report on the given topic.

Include:
1. Introduction
2. Key Findings
3. Applications or Examples
4. Advantages and Limitations
5. Conclusion

Rules:
- Use your existing knowledge only.
- Do not use web search or external retrieval.
- Do not invent citations or source URLs.
- If you are uncertain about a fact, mention the uncertainty.
"""),
    ("human", "Write a research report on: {topic}")
])

baseline_chain = baseline_prompt | llm | StrOutputParser()


def run_baseline(topic):
    start_time = time.time()

    report = baseline_chain.invoke({
        "topic": topic
    })

    execution_time = round(time.time() - start_time, 2)

    return report, execution_time