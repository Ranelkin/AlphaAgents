import tracemalloc
from dotenv import load_dotenv
from src.graph import run_debate
from src.util import setup_logging

logger = setup_logging('main')

def main(): 
    load_dotenv()
    tracemalloc.start()
    
    run_debate('AAPL')
        

if __name__ == '__main__': 
    main()