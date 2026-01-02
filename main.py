from src.util.log_config import setup_logging
from dotenv import load_dotenv

logger = setup_logging('main')

def main(): 
    load_dotenv()
    pass

    
if __name__ == '__main__': 
    main()