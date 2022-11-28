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

sentiment analysis: node 2
source: https://huggingface.co/models?filter=text-classification
huggingface/distilbert-base-uncased-finetuned-mnli
huggingface/prunebert-base-uncased-6-finepruned-w-distil-mnli
distilbert-base-uncased-finetuned-sst-2-english
Souvikcmsa/BERT_sentiment_analysis
Souvikcmsa/SentimentAnalysisDistillBERT
Souvikcmsa/Roberta_Sentiment_Analysis