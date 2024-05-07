import numpy as np
import numpy.typing
import pyaudio
import sounddevice as sd
import speech_recognition as sr
from openai import OpenAI
from openai.types.beta.threads import TextContentBlock
from typing import Final, cast

RECORDING_SAMPLE_RATE: Final[int] = 44100
PLAYBACK_SAMPLE_RATE: Final[int] = 24000
SILENCE_THRESHOLD: Final[int] = 1500
WAIT_FOR_SOUND_SEC: Final[int] = 5
LLM_CLIENT_ATTENDANT_CONTEXT: Final[str] = (
    "You are an AI chatbot designed to assist customers\
    in placing orders at a Starbucks drive through.\
    A customer has just pulled in to the drive through.\
    You must take into account the following constraints:\
    1. Each drink must have one of three sizes, tall, grande, or venti.\
    2. Each drink can have extra toppings for a cost.\
    3. A customer cannot order more than 10 items.\
    4. You must provide the customer with their order total when they are finished ordering.\
    Do not provide any additional commentary or suggestions beyond what is requested.\
    Do not speak for the user."
)
LLM_CLIENT_PROGRAM_CONTEXT: Final[str] = (
    "You are an AI assistant designed to generate an order history\
    in a json format with product name, size, extras, and price for each item.\
    You must take into account the following constraints:\
    1. Each drink must have one of three sizes, tall, grande, or venti.\
    2. Each drink can have extra toppings for a cost.\
    3. A customer cannot order more than 10 items.\
    4. You cannot provide any additional commentary or suggestions beyond what is requested.\
    5. You cannot speak for the user.\
    6. You can only produce the json order history and nothing else."
)

API_KEY = "sk-proj-f23rt0ggTQSRCkaF42K1T3BlbkFJkONquEQE07srMUAYcO3u"
ORG_ID = "org-juvKipYAYRMMLBRSMICCfD32"
PROJ_ID = "proj_7ha00Sn8B48Upyo3f0VFjUE8"
MODEL = "gpt-3.5-turbo-1106"
ENV_LEVEL = "DEBUG"


class Logger:
    @staticmethod
    def debug(message: str) -> None:
        if ENV_LEVEL == "DEBUG":
            print(message, end="", flush=True)


class AudioRecorder:
    @staticmethod
    def record() -> BytesIO:
        initial_audio = AudioRecorder._record_with_defaults(WAIT_FOR_SOUND_SEC)

        while True:
            audio_batch = AudioRecorder._record_with_defaults(1)
            combined_audio = np.concatenate((initial_audio, audio_batch))

            if np.max(audio_batch) < SILENCE_THRESHOLD:
                break

        return sr.AudioData(
            combined_audio.flatten().tobytes(), RECORDING_SAMPLE_RATE, 2
        )

    @staticmethod
    def _record_with_defaults(wait_time_seconds: int) -> np.typing.NDArray[np.float64]:
        audio: np.typing.NDArray[np.float64] = sd.rec(
            int(wait_time_seconds * RECORDING_SAMPLE_RATE),
            samplerate=RECORDING_SAMPLE_RATE,
            channels=1,
            dtype="int16",
        )

        sd.wait()

        return audio


class AudioPlayer:
    def __init__(self, py_audio: pyaudio.PyAudio) -> None:
        self._player = py_audio.open(
            format=pyaudio.paInt16, channels=1, rate=PLAYBACK_SAMPLE_RATE, output=True
        )

    def queue_sound(self, audio: bytes) -> None:
        self._player.write(audio)


class TTSProvider:
    def __init__(self, openai: OpenAI, audio_player: AudioPlayer) -> None:
        self._client = openai
        self._audio_player = audio_player

    def text_to_speech(self, text: str) -> None:
        with self._client.audio.speech.with_streaming_response.create(
            model="tts-1-hd",
            voice="alloy",
            input=text,
            response_format="wav",
        ) as response:
            for chunk in response.iter_bytes(chunk_size=1024):
                self._audio_player.queue_sound(chunk)


class STTProvider:
    def __init__(self, recognizer: sr.Recognizer):
        self._google_audio_recognizer = recognizer

    def speech_to_text(self) -> str:
        # TODO: Google may revoke this functionality at any time.
        # Move to an implementation that doesn't rely on a Web API
        audio_bytes = AudioRecorder.record()

        text = ""
        try:
            text += self._google_audio_recognizer.recognize_google(audio_bytes)
        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))

        return text


class MockAssistant:
    def __init__(
        self, openai: OpenAI, name: str, instructions: str, tts_provider: TTSProvider
    ) -> None:
        self._tts_provider = tts_provider

    def prompt(self) -> str:
        return "Welcome to Starbucks! What can I get started for you today?"


class Assistant:
    def __init__(
        self, openai: OpenAI, name: str, instructions: str, tts_provider: TTSProvider
    ) -> None:
        client = OpenAI(
            api_key=API_KEY,
            organization=ORG_ID,
            project=PROJ_ID,
        ).beta
        client = openai.beta
        self._client = client.threads
        self._thread = self._client.create()
        self._assistant = client.assistants.create(
            name=name,
            instructions=instructions,
            model=MODEL,
        )
        self._tts_provider = tts_provider

    def prompt(self, message: str) -> str:
        self._client.messages.create(
            thread_id=self._thread.id,
            role="user",
            content=message,
        )

        run = self._client.runs.create_and_poll(
            thread_id=self._thread.id,
            assistant_id=self._assistant.id,
            instructions=self._assistant.instructions,
        )

        if run.status == "completed":
            messages = self._client.messages.list(thread_id=self._thread.id)
            message_content = cast(TextContentBlock, messages.data[0].content[0])
            return str(message_content.text.value)
        else:
            return "Please speak to a customer service agent, as our AI assistant is currently unavailable."


class Main:
    def __init__(self) -> None:
        self._py_audio = pyaudio.PyAudio()
        self._audio_player = AudioPlayer(self._py_audio)
        self._client = OpenAI(
            api_key=API_KEY,
            organization=ORG_ID,
            project=PROJ_ID,
        )
        self._recognizer = sr.Recognizer()
        self._stt_provider = STTProvider(self._recognizer)
        self._tts_provider = TTSProvider(self._client, self._audio_player)
        self._assistant = Assistant(
            self._client,
            "Starbucks Attendant",
            LLM_CLIENT_ATTENDANT_CONTEXT,
            self._tts_provider,
        )

    def run(self) -> None:
        user_prompt = "Hello."

        while True:
            response = self._assistant.prompt(user_prompt)
            Logger.debug(f"\nassistant > {response}")
            self._tts_provider.text_to_speech(response)

            user_prompt = self._stt_provider.speech_to_text()
            Logger.debug(f"\ncustomer > {user_prompt}")


Main().run()
