import collections
import logging
import sys
import jsonlines
import csv

from lib.cmp_classifier import CMPClassifier

logging.basicConfig(
    format="[%(asctime)s %(levelname)-7s] %(message)s", level=logging.INFO
)

classifier = CMPClassifier()

chunks = collections.defaultdict(list)
for row in jsonlines.open("data/raw/manifesto_2023_labelled.jsonl"):
    chunks[row.pop("chunk_id")].append(row)

w = csv.writer(open("data/intermediate/topics_validation.csv", "w"))
w.writerow(
    [
        "chunkid",
        "relevant",
        "relevant_prob",
        "topic1",
        "topic1_score",
        "topic2",
        "topic2_score",
        "topic3",
        "topic3_score",
        "manual1",
        "manual2",
        "manual3",
    ]
)
for chunk, rows in chunks.items():
    context = " ".join(row["sentence"] for row in rows)
    for row in rows:
        sent = row["sentence"]
        relevant, relevant_prob = classifier.is_relevant(sent)
        codes = classifier.predict(sent, context=context)
        w.writerow(
            [chunk, relevant, relevant_prob]
            + list(codes[0])
            + list(codes[1])
            + list(codes[2])
            + row["labels"]
        )
