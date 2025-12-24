from .session import Session
from typing import Optional

class Context:
    """Global singleton wrapper for MCPSessionManager"""
    _instance: Optional[Session] = None
    
    @classmethod
    def get_instance(cls) -> MCPSessionManager:
        """Get or create the global MCP manager instance"""
        if cls._instance is None:
            cls._instance = Session()
            cls._instance.start()
        return cls._instance
    
    @classmethod
    def shutdown(cls):
        """Shutdown the global MCP manager"""
        if cls._instance:
            cls._instance.stop()
            cls._instance = None


def get_mcp_manager() -> MCPSessionManager:
    """Get the global MCP manager"""
    return Context.get_instance()

def shutdown_mcp():
    """Shutdown MCP connection"""
    Context.shutdown()