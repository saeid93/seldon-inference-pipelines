# Source
adapted from [FA2: Fast, Accurate Autoscaling for Serving Deep Learning Inference with SLA Guarantees](https://ieeexplore.ieee.org/abstract/document/9804606)


Two node pipeline

list of available models per node:

text-summerization: node 1
source: https://huggingface.co/models?pipeline_tag=summarization

list of models:
sshleifer/distilbart-cnn-12-6
sshleifer/distilbart-xsum-1-1
sshleifer/distill-pegasus-cnn-16-4
sshleifer/distill-pegasus-xsum-16-4
sshleifer/distilbart-xsum-12-3
sshleifer/distilbart-xsum-6-6
sshleifer/pegasus-cnn-ft-v2
sshleifer/distilbart-cnn-6-6
sshleifer/distilbart-xsum-12-6
sshleifer/distilbart-cnn-12-3
sshleifer/distilbart-xsum-12-1
sshleifer/distilbart-xsum-9-6
sshleifer/distill-pegasus-xsum-16-8
facebook/bart-large-cnn
google/roberta2roberta_L-24_bbc
google/pegasus-cnn_dailymail
google/roberta2roberta_L-24_cnn_daily_mail
google/pegasus-large

Question-answering: node 2
source: https://huggingface.co/models?pipeline_tag=question-answering

list of models:
deepset/roberta-base-squad2
deepset/xlm-roberta-large-squad2
distilbert-base-cased-distilled-squad
deepset/xlm-roberta-base-squad2