from typing import List

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

from data import textify_table
from args import MainArgs
from log import log
from utils import element_wise_cosine_sim


def auto_selection(tables_df: List[pd.DataFrame], args: MainArgs):
    table_df = pd.concat(tables_df, axis=0)
    table_df = table_df.sample(frac=args.selection_rate)
    model = SentenceTransformer(args.lm_model_or_path)
    model.max_seq_length = args.max_seq_length
    model.to(args.device)
    sentences_before = textify_table(table_df)
    table_embeddings = model.encode(sentences_before, show_progress_bar=True,
                                    batch_size=args.batch_size, normalize_embeddings=True)
    selected_attrs = ["tid"]
    for item in table_df.items():
        name, col = item
        col_copy = col.copy(deep=True)
        if name == "tid":
            continue
        table_df[name] = col_copy.sample(frac=1).reset_index(drop=True)
        sentences_after = textify_table(table_df)
        table_df[name] = col
        table_embeddings_after = model.encode(sentences_after, show_progress_bar=True,
                                              batch_size=args.batch_size, normalize_embeddings=True)
        sim = element_wise_cosine_sim(table_embeddings, table_embeddings_after)
        mean_sim = np.mean(sim)
        if mean_sim <= args.col_sim_threshold:
            selected_attrs.append(name)
        log(f"col: {name}, sim: {mean_sim}")
    return selected_attrs
