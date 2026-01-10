import tracemalloc
from dotenv import load_dotenv
from src.util import setup_logging
from src.chat import chat_interface
logger = setup_logging('main')

def main(): 
    load_dotenv()
    tracemalloc.start()
  
    chat_interface()
        

if __name__ == '__main__': 
    main()