from .context import get_mcp_manager, shutdown_mcp
from .session_manager import Session


session = Session()
_client = session.start()