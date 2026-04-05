import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

class SoundClassifier:
    def __init__(self, model_path='https://tfhub.dev/google/yamnet/1'):
        self.model = hub.load(model_path)
        # YAMNet expects 16kHz audio
        self.sample_rate = 16000 

    def classify_audio(self, audio_data):
        # audio_data should be a normalized float32 array
        scores, embeddings, spectrogram = self.model(audio_data)
        return scores