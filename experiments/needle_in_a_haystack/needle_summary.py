# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import argparse
import json
import os
from collections import Counter
from rouge_score import rouge_scorer


def summary(run_name: str, output_path: str, needle_path: str):
    pathlist = os.listdir(needle_path)

    datas, cs = [], set()
    for path in pathlist:
        if run_name in path and path.endswith(".json"):
            data = json.load(open(os.path.join(needle_path, path)))
            if data[0]["context_length"] in cs:
                continue
            datas.extend(data)
            cs.add(data[0]["context_length"])

    res = Counter()
    for ii in datas:
        res[(ii["context_length"], ii["depth_percent"])] += ii["correct"] == True
        if ii["correct"] is False:
            print(ii["response"])
    sorted(res.items())
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    full_answer = "The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day."
    for entry in datas:
        entry["rouge1_score"] = scorer.score(full_answer, entry["response"])["rouge1"].recall
        entry["rouge2_score"] = scorer.score(full_answer, entry["response"])["rouge2"].recall
        entry["rougeL_score"] = scorer.score(full_answer, entry["response"])["rougeL"].recall

    with open(f"{output_path}/{run_name}.json", "w") as json_file:
        json.dump(datas, json_file)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--run_name", type=str, default=None)
    args.add_argument("--output_path", type=str, default="results/needle/")
    args.add_argument("--needle_path", type=str, default="results/needle/")
    args = args.parse_args()

    summary(
        run_name=args.run_name,
        output_path=args.output_path,
        needle_path=args.needle_path,
    )
