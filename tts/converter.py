from audio.player import AudioPlayer
from openai import OpenAI


class TTSConverter:
    def __init__(
        self, openai: OpenAI, audio_player: AudioPlayer, model: str, voice: str
    ) -> None:
        self._client = openai.audio.speech
        self._audio_player = audio_player
        self._model = model
        self._voice = voice

    def text_to_speech(self, text: str) -> None:
        with self._client.with_streaming_response.create(
            model=self._model,
            voice=self._voice,
            input=text,
            response_format="wav",
        ) as response:
            for chunk in response.iter_bytes(chunk_size=1024):
                self._audio_player.queue_sound(chunk)
