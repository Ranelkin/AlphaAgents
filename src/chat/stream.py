# Stream generator for a nice UI experience 
import time
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from _collections_abc import Generator

def chat_stream(query: str) -> Generator[DataFrame | str]:
    for word in query.split(" "):
        yield word + " "
        time.sleep(0.02)

    yield pd.DataFrame(
        np.random.randn(5, 10),
        columns=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
    )

    for word in query.split(" "):
        yield word + " "
        time.sleep(0.02)
