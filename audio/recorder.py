import numpy as np
import pydub as pd
import sounddevice as sd
import speech_recognition as sr

from io import BytesIO
from typing import Final

SOUND_FORMAT: Final[str] = "mp3"
SOUND_RAW_DATA_TYPE: Final[str] = "int16"


class AudioRecorder:
    def __init__(self, wait_s: int, sound_threshold: int, sample_rate: int):
        self._wait_s = wait_s
        self._sound_threshold = sound_threshold
        self._sample_rate = sample_rate

    def record(self) -> BytesIO:
        initial_audio = self._record_with_defaults(self._wait_s)

        while True:
            audio_batch = self._record_with_defaults(1)
            combined_audio = np.concatenate((initial_audio, audio_batch))

            if np.max(audio_batch) < self._sound_threshold:
                break

        audio_data = sr.AudioData(
            combined_audio.flatten().tobytes(), self._sample_rate, 2
        )

        audio_segment = pd.AudioSegment(
            audio_data.get_raw_data(),
            sample_width=audio_data.sample_width,
            frame_rate=audio_data.sample_rate,
            channels=1,
        )

        buffer = BytesIO()
        buffer.name = f"file.{SOUND_FORMAT}"
        audio_segment.export(buffer, format=SOUND_FORMAT)

        return buffer

    def _record_with_defaults(self, wait_s: int) -> np.typing.NDArray[np.float64]:
        audio: np.typing.NDArray[np.float64] = sd.rec(
            int(wait_s * self._sample_rate),
            samplerate=self._sample_rate,
            channels=1,
            dtype=SOUND_RAW_DATA_TYPE,
        )

        sd.wait()

        return audio
