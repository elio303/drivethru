import pyaudio


class AudioPlayer:
    def __init__(self, py_audio: pyaudio.PyAudio, sample_rate: int) -> None:
        self._player = py_audio.open(
            format=pyaudio.paInt16, channels=1, rate=sample_rate, output=True
        )

    def queue_sound(self, audio: bytes) -> None:
        self._player.write(audio)
