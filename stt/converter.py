from audio.recorder import AudioRecorder
from openai import OpenAI


class STTConverter:
    def __init__(
        self, client: OpenAI, audio_recorder: AudioRecorder, model: str
    ) -> None:
        self._client = client.audio.transcriptions
        self._audio_recorder = audio_recorder
        self._model = model

    def speech_to_text(self) -> str:
        audio_bytes = self._audio_recorder.record()

        return str(
            self._client.create(
                model=self._model,
                file=audio_bytes,
            ).text
        )
