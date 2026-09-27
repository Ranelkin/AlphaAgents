from langchain.chat_models import init_chat_model


def init_agent_model(
    model, provider, temperature=0.7, max_retries=4, timeout=120, seed=None, **kwargs
):
    if seed is not None:
        kwargs["seed"] = seed
    return init_chat_model(
        model,
        model_provider=provider,
        temperature=temperature,
        max_retries=max_retries,
        timeout=timeout,
        **kwargs
    )
