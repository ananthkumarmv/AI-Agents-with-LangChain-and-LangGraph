from pydantic import BaseModel, Field
from typing import List

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_tavily import TavilySearch

load_dotenv()

class Source(BaseModel):
    """Schema for a source used by the agent"""

    url:str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema for the response from the agent with answer and sources"""

    answer:str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(default_factory=list, description="List of sources used to generate the answer")


SYSTEM_PROMPT = (
    "Use the search tool as many times as needed to gather information. "
    "Once you have enough information to answer the user's question, you "
    "MUST call the AgentResponse tool exactly once with your final answer "
    "and the source URLs you used. Do not respond in plain text."
)

llm = ChatOllama(temperature=0, model="gemma4:31b-cloud")
# llm = ChatOllama(temperature=0, model="qwen3.5:9b")

tools = [TavilySearch()]
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
    response_format=ToolStrategy(AgentResponse),
)


def main():
    result = agent.invoke(
        {"messages": [HumanMessage(content="Search for 3 job posting for an AI engineer with 3 years of experience in Banagalore location.")]},
        config={"recursion_limit": 50},
    )
    print(result["structured_response"])


if __name__ == "__main__":
    main()