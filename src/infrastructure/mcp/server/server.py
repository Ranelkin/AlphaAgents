from fastmcp import FastMCP
from ....util.log_config import setup_logging
from ...tools import TOOLS

logger = setup_logging("mcp.server")

mcp = FastMCP(
    name="AAS",
    instructions="""
        This server provides different mcp tools.
    """,
)

try: 

    for func in TOOLS: 
        mcp.tool(
            func, 
            name=None, 
            description=None
        )
except Exception as e: 
    logger.error(f"Failed to register tools: {e}")

if __name__ == '__main__': 
    mcp.run()