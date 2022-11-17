# Source
Adapted from [clarifai-asr-sentiment](https://clarifai.com/clarifai/main/workflows/asr-sentiment)

Two node pipeline

list of available models per node:

automatic-speech-recognition: node 1
https://huggingface.co/models?pipeline_tag=automatic-speech-recognition

facebook/s2t-small-librispeech-asr
facebook/s2t-medium-librispeech-asr
facebook/s2t-large-librispeech-asr
facebook/wav2vec2-base-960h
facebook/wav2vec2-large-960h


Question Answering: node 2
source: https://huggingface.co/models?pipeline_tag=question-answering

deepset/roberta-base-squad2
deepset/xlm-roberta-large-squad2
distilbert-base-cased-distilled-squad
deepset/xlm-roberta-base-squad2