import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from typing import Any, Dict, Optional
from ....util.log_config import setup_logging

logger = setup_logging("mcp.client")

class MCPClient:
    def __init__(self, server_command):
        self.server_command = server_command
        self.mcp_server = StdioServerParameters(
            command="python3",
            args=["-m", "src.infrastructure.mcp.server.server"],
            env=None  
        )
        self.session: Optional[ClientSession] = None
        self.read = None
        self.write = None
        self.loop = None
        self._context_task: Optional[asyncio.Task] = None  
        self._exit_event: Optional[asyncio.Event] = None  

    def connect(self):
        """Synchronous connection wrapper"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        async def _run_context():
            """Run the full context lifecycle in one task."""
            async with stdio_client(self.mcp_server) as (read, write):
                self.read, self.write = read, write
                async with ClientSession(read, write) as session:
                    self.session = session
                    await session.initialize()
                    connect_future.set_result(None)  
                    self._exit_event = asyncio.Event()
                    await self._exit_event.wait()  

        connect_future = connect_future = self.loop.create_future() 
        self._context_task = self.loop.create_task(_run_context())
        self.loop.run_until_complete(connect_future)  
        logger.info("MCP client connection established")
        
    def disconnect(self):
        """Synchronous disconnection wrapper"""
        if not self.loop or not self._context_task:
            return
        if self._exit_event:
            self._exit_event.set() 
        self.loop.run_until_complete(self._context_task)  
        self.loop.close()
        logger.info("MCP client connection closed")

    def call_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Synchronous tool call wrapper"""
        if self.session is None:
            self.connect()
        async def _call():
            """Call a tool asynchronously and format the result as a JSON string."""
            assert self.session
            result = await self.session.call_tool(tool_name, arguments=args)
            return json.dumps(result, indent=2) if isinstance(result, dict) else str(result)
        assert self.loop
        return self.loop.run_until_complete(_call())