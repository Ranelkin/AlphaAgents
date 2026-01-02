from langchain.tools import Tool
from typing import Optional
from .client.client import MCPClient
from ...util.log_config import setup_logging
import inspect

logger = setup_logging("mcp.session")

class Session:
    def __init__(self):
        """Initialize the session manager with no active MCP tool."""
        self.mcp_tool: MCPClient 
        
    def start(self):
        """Start persistent connection to the MCP server.
        """
        logger.info("Starting MCPSessionManager...")
        self.mcp_tool = MCPClient(["python3", "-m", "src.mcp_server.mcp_server"])
        self.mcp_tool.connect()
        return self.mcp_tool
    
    def stop(self):
        """Stop persistent connection to the MCP server.
        
        
        """
        logger.info("Stopping MCPSessionManager...")
        if self.mcp_tool:
            self.mcp_tool.disconnect()
    
    def _create_tool_wrapper(self, tool_func):
        """Create a wrapper function that calls the MCP server with the tool's parameters.
        
        Args:
            tool_func: The original tool function from grouped_tools
            
        Returns:
            A wrapper function that routes calls through the MCP server
        """
        tool_name = tool_func.__name__
        sig = inspect.signature(tool_func)
        params = list(sig.parameters.keys())
        assert self.mcp_tool != None
       
        if len(params) == 1:
            def wrapper(arg=None, **kwargs):
                """Dynamically generated wrapper for single-parameter tool."""
                # Handle both positional and keyword arguments
                if arg is not None:
                    call_args = {params[0]: arg}
                else:
                    call_args = kwargs
                logger.info(f"Calling MCP tool '{tool_name}' with args: {call_args}")
                result = self.mcp_tool.call_tool(tool_name, call_args)
                return result
        else:
            def wrapper(**kwargs):
                """Dynamically generated wrapper for multi-parameter tool."""
                logger.info(f"Calling MCP tool '{tool_name}' with args: {kwargs}")
                result = self.mcp_tool.call_tool(tool_name, kwargs)
                return result
        
        # Preserve metadata
        wrapper.__name__ = tool_name
        wrapper.__doc__ = tool_func.__doc__
        
        return wrapper
            
    def get_tools(self):
        """Retrieve tools that interact with the MCP server.
        
        Automatically converts all tools from grouped_tools into LangChain Tools
        without requiring manual wrapper functions.
        
        Raises:
            RuntimeError: If the session has not been started.
        
        Returns:
            list: A list of configured Tool objects dynamically generated from grouped_tools.
        
        
        """
        if not self.mcp_tool:
            raise RuntimeError("MCP session not started. Call start() first.")
        
        from src.infrastructure.tools import TOOLS
        
        langchain_tools = []
        
        # Dynamically create LangChain Tool objects for each registered tool
        for tool_func in TOOLS:
            tool_name = tool_func.__name__
            tool_description = tool_func.__doc__ or f"Tool: {tool_name}"
            
            # Create a wrapper that routes through MCP
            wrapper = self._create_tool_wrapper(tool_func)
            
            # Create LangChain Tool
            langchain_tool = Tool(
                name=tool_name,
                func=wrapper,
                description=tool_description.strip()
            )
            
            langchain_tools.append(langchain_tool)
            logger.info(f"Registered tool: {tool_name}")
        
        return langchain_tools

