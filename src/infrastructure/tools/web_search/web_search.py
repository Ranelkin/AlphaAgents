from tavily import TavilyClient 
from dotenv import load_dotenv

load_dotenv()

tavily_client = TavilyClient()

def search_web(search: str): 
    return tavily_client.search(search, include_answer=True)['answer']

if __name__ == '__main__': 
    result = search_web("What is quantum mechanics?")
    print(result)