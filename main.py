from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain_tavily import TavilySearch

load_dotenv()

llm = ChatOllama(temperature=0, model="gemma4:31b-cloud")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)


def main():
    result = agent.invoke(
        {"messages": HumanMessage(content="Search for 3 job posting for an AI engineer with 3 years of experience in Banagalore location.")}
    )
    print(result)


if __name__ == "__main__":
    main()

