import os
import phoenix as px 
from openinference.instrumentation.autogen_agentchat import AutogenAgentChatInstrumentor
from phoenix.otel import register
from src.util import setup_logging

logger = setup_logging('Phoenix tracing config')

def init_phoenix():
    try:
        
        session = px.launch_app()
        tracer_provider = register(
            project_name='Alpha Agents'
        )
        instrumentor = AutogenAgentChatInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider) #type: ignore

        logger.info('Phoenix tracing initialized')
        return tracer_provider
    except Exception as e:
        logger.error(f'Failed to initialize phoenix {e}')