import os

import phoenix as px
from openinference.instrumentation._tracers import OITracer
from openinference.instrumentation.config import TraceConfig
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from src.util import setup_logging

logger = setup_logging("tracing.phoenix")

_tracer = None


def get_tracer():
    """Return the active tracer, or a no-op tracer before initialization."""
    global _tracer
    if _tracer is None:
        _tracer = OITracer(trace.get_tracer("alpha-agents-noop"), config=TraceConfig())
    return _tracer


def init_phoenix():
    global _tracer

    try:
        try:
            px.launch_app()
        except Exception:
            logger.info("Phoenix app not launched (may already be running)")

        endpoint = os.getenv("PHOENIX_ENDPOINT", "http://localhost:6006/v1/traces")
        resource = Resource.create(
            {"service.name": "alpha-agents", "project.name": "Alpha Agents"}
        )
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
        )
        trace.set_tracer_provider(provider)

        LangChainInstrumentor().instrument(tracer_provider=provider)

        _tracer = OITracer(trace.get_tracer("alpha-agents"), config=TraceConfig())

        logger.info("Phoenix tracing initialized successfully")
        return provider
    except Exception as exc:
        logger.error("Failed to initialize Phoenix: %s", exc)
        return None
