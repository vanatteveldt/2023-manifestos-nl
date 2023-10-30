import collections
import logging
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import csv
from amcat4py import AmcatClient


class CMPClassifier:
    def __init__(self):
        logging.info("Loading CMP topics...")
        self.topics = {
            int(row["cmp"]): row["label"]
            for row in csv.DictReader(open("data/raw/cmp_topics.csv"))
        }
        logging.info("Loading CMP BERT model...")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "manifesto-project/manifestoberta-xlm-roberta-56policy-topics-context-2023-1-1",
            trust_remote_code=True,
        )
        logging.info("Loading CMP Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.cuda.empty_cache()
        logging.info("Using device {self.device}")
        self.model.to(self.device)

    def predict(self, text, context=None):
        inputs = self.tokenizer(
            text,
            context or text,
            return_tensors="pt",
            max_length=300,  # we limited the input to 300 tokens during finetuning
            padding="max_length",
            truncation=True,
        ).to(self.device)

        logits = self.model(**inputs).logits
        probabilities = torch.softmax(logits, dim=1).tolist()[0]

        for i, p in sorted(enumerate(probabilities), key=lambda item: -item[1]):
            yield self.model.config.id2label[i], p

    def predict_vu(self, text, context=None):
        result = collections.defaultdict(int)
        for c, p in self.predict(text, context):
            code = int(c.split(" ")[0])
            label = self.topics[code]
            result[label] += p
        return sorted(result.items(), key=lambda item: -item[1])


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
