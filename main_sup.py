from itertools import chain
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
import fasttext

from args import build_main_args
from data import Table, read_all_tables, read_test_ground_truth, textify_table, read_ground_truth
from log import init_logger, log_args, log_time, log
from timer import Timer
from metrics import evaluate_log
from selector import auto_selection
from merger import merge, merge_parallel
from pruner import pruning, pruning_parallel
from typing import List, Tuple, Set


def prediction_filter(ent: Set[int], prediction: List[Tuple]):
    return [p for p in prediction if not set(p).isdisjoint(ent)]


if __name__ == '__main__':
    args = build_main_args()
    file_name = f"main"
    log_file_name = init_logger(file_name)
    log_args(args)
    log(log_file_name)
    data_path = Path(args.data_path)
    full_data_path = data_path / args.data_name
    timer = Timer()
    # pd.df
    T, tables_df = read_all_tables(full_data_path)
    if args.eer_flag:
        timer.start()
        selected_attrs = auto_selection(tables_df, args)
        tm = timer.stop()
        log_time("selecting", tm)
        if args.fixed_eer:
            selected_attrs = args.fixed_eer
        log(f"fixed eer: {selected_attrs}")
    else:
        selected_attrs = None
    timer.start()
    T, tables_df = read_all_tables(
        full_data_path, selected_attrs=selected_attrs)
    tm = timer.stop()
    log_time("read all tables", tm)
    table_lens = [len(table) for table in tables_df]
    n = sum(table_lens)
    table_ids = [table["tid"].tolist() for table in tables_df]
    # data.Table
    tables = [Table(str(idx), table_id, list(range(len(table_id))))
              for idx, table_id in enumerate(table_ids)]
    timer.start()
    table_sentences = [textify_table(table) for table in tables_df]
    if args.fasttext:
        from tqdm import tqdm
        with open("tmp/fasttext_data.txt", "w") as wt:
            for sentences in table_sentences:
                for sentence in tqdm(sentences, desc="fasttext write file"):
                    wt.write(sentence+"\n")
        model = fasttext.train_unsupervised(
            "tmp/fasttext_data.txt", model="skipgram")
        all_embeddings = []
        for sentences in table_sentences:
            for sentence in tqdm(sentences, desc="fasttext write file"):
                all_embeddings.append(model.get_sentence_vector(sentence))
    else:
        model = SentenceTransformer(args.lm_model_or_path)
        model.max_seq_length = args.max_seq_length
        model.to(args.device)
        table_embeddings = [
            model.encode(sentences, show_progress_bar=True,
                         batch_size=args.batch_size, normalize_embeddings=True)
            for sentences in table_sentences]
        all_embeddings = list(chain(*table_embeddings))
    all_embeddings = np.array(all_embeddings)
    tm = timer.stop()
    log_time("encode all tables", tm)
    ground_truth = read_test_ground_truth(full_data_path)
    timer.start()
    if args.run_in_parallel:
        table = merge_parallel(tables, all_embeddings, args)
    else:
        table = merge(tables, all_embeddings, args)
    tm = timer.stop()
    log_time("merging", tm)
    prediction = table.get_tuples()
    ent = set(chain(*ground_truth))
    prediction = prediction_filter(ent, prediction)
    evaluate_log(ground_truth, prediction)
    timer.start()
    if args.run_in_parallel:
        new_prediction = pruning_parallel(prediction, all_embeddings, args)
    else:
        new_prediction = pruning(prediction, all_embeddings, args)
    tm = timer.stop()
    log_time("pruning", tm)
    new_prediction = prediction_filter(ent, new_prediction)
    evaluate_log(ground_truth, new_prediction)
