# Wav2VecSpeechEncoder

   
   `Wav2VecSpeechEncoder` is a speech encoder based on `wav2vec` - unsupervised pre-training for speech
    recognition presented and implemented by Facebook: https://github.com/pytorch/fairseq/tree/master/examples/wav2vec
    
   `Wav2VecSpeechEncoder` uses a pre-trained model to encode an audio signal from a `Batch x Signal Length`
    ndarray into a `Batch x Concatenated Features` ndarray.
    