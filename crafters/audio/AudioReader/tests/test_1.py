from .. import AudioReader


def test_1():
    import librosa
    audio_file_path = librosa.util.example_audio_file()

    crafter = AudioReader()
    crafted_doc = crafter.craft(audio_file_path, 0)

    signal = crafted_doc['blob']
    assert signal.shape == (2, 1355168)
