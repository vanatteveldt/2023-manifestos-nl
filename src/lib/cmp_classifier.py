import collections
import csv
import logging
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline


class CMPClassifier:
    def __init__(self):
        logging.info("Loading CMP topics...")
        self.topics = {
            int(row["cmp"]): row["label"]
            for row in csv.DictReader(open("data/raw/cmp_topics.csv"))
        }
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.cuda.empty_cache()
        logging.info(f"Using device {self.device}")
        logging.info("Loading CMP BERT model...")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "manifesto-project/manifestoberta-xlm-roberta-56policy-topics-context-2023-1-1",
            trust_remote_code=True,
        )
        self.model.to(self.device)
        logging.info("Loading CMP Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
        logging.info("Loading relevance classifier")

        self.relevance_pipe = pipeline(
            "text-classification",
            model="joris/manifesto-dutch-binary-relevance",
            trust_remote_code=True,
            device=self.device,
        )

    def is_relevant(self, sentence):
        r = self.relevance_pipe(sentence)
        return r[0]["label"], r[0]["score"]

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

        return [
            (self.model.config.id2label[i], p)
            for i, p in sorted(enumerate(probabilities), key=lambda item: -item[1])
        ]

    def predict_vu(self, text, context=None):
        result = collections.defaultdict(int)
        for c, p in self.predict(text, context):
            code = int(c.split(" ")[0])
            label = self.topics[code]
            result[label] += p
        return sorted(result.items(), key=lambda item: -item[1])
