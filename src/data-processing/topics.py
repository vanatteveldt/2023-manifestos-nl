import collections
import logging
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import csv
from amcat4py import AmcatClient


logging.basicConfig(
    format="[%(asctime)s %(levelname)-7s] %(message)s", level=logging.INFO
)

classifier = CMPClassifier()

HOST = "https://open.amcat.nl/amcat"
INDEX = "2023-nl-manifestos"

logging.info(f"Logging into AmCAT at {HOST}")
amcat = AmcatClient(HOST)
amcat.login()
amcat.set_fields(INDEX, dict(cmp="tag", issue="tag", issue_confidence="double"))

logging.info("Querying documents")
for i, doc in enumerate(amcat.query(INDEX, fields=["text"])):
    if not i % 100:
        logging.info(i)
    cmp, confidence = list(classifier.predict(doc["text"]))[0]
    cmp_code = int(cmp.split(" ")[0])
    vu = classifier.topics[cmp_code]
    fields = dict(cmp=cmp, issue=vu, issue_confidence=confidence)
    amcat.update_document(INDEX, doc_id=doc["_id"], body=fields)
