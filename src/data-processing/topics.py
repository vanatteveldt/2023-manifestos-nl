# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-classification", model="manifesto-project/manifestoberta-xlm-roberta-56policy-topics-context-2023-1-1", trust_remote_code=True)

print(pipe("Bestaanszekerheid is het belangrijkste onderwerp van 2023"))
