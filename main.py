import numpy as np
import numpy.typing
import os
import pyaudio
import pydub
import sounddevice as sd
import speech_recognition as sr
import sys
from dotenv import load_dotenv
from io import BytesIO
from openai import OpenAI
from openai.types.beta.assistant import Assistant
from openai.types.beta.threads import TextContentBlock
from typing import Final, Literal, TypedDict, cast

LLM_CLIENT_PROGRAM_CONTEXT: Final[str] = (
    "You are an AI assistant designed to generate an order history\
    in a json format with product name, size, extras, and price for each item.\n\
    You must take into account the following constraints:\n\
    1. Each drink must have one of three sizes, tall, grande, or venti.\n\
    2. Each drink can have extra toppings for a cost.\n\
    3. A customer cannot order more than 10 items.\n\
    4. You cannot provide any additional commentary or suggestions beyond what is requested.\n\
    5. You cannot speak for the user.\n\
    6. You can only produce the json order history and nothing else.\n"
)

ASSISTANT_ID: Final[str] = "asst_4cCB3pkqgBEwggBywlwh7bvW"
ASSISTANT_NAME: Final[str] = "Starbucks Drive-thru Attendant"
ASSISTANT_MODEL: Final[str] = "gpt-3.5-turbo-1106"
ASSISTANT_CONTEXT: Final[str] = (
    "You are an AI chatbot designed to assist customers\
    in placing orders at a Starbucks drive through.\n\
    A customer has just pulled in to the drive through.\n\
    You must take into account the following constraints:\n\
    1. Each drink must have one of three sizes, tall, grande, or venti.\n\
    2. Each drink can have extra toppings for a cost.\n\
    3. A customer cannot order more than 10 items.\n\
    4. You must provide the customer with their order total when they are finished ordering.\n\
    Do not provide any additional commentary or suggestions beyond what is requested.\n\
    Do not speak for the user.\n"
)

TTS_MODEL: Literal["tts-1", "tts-1-hd"] = "tts-1-hd"
TTS_VOICE: Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"] = "onyx"
TTS_SAMPLE_RATE: Final[int] = 24000

STT_MODEL: Final[str] = "whisper-1"
STT_SAMPLE_RATE: Final[int] = 44100
STT_SILENCE_THRESHOLD: Final[int] = 1500
STT_WAIT_SEC: Final[int] = 5

ENV_LEVEL: Literal["DEBUG", "DEV", "PRODUCTION"] = "DEBUG"


class Logger:
    @staticmethod
    def debug(message: str) -> None:
        if ENV_LEVEL == "DEBUG":
            print(message, end="", flush=True)


class AudioRecorder:
    @staticmethod
    def record() -> BytesIO:
        initial_audio = AudioRecorder._record_with_defaults(STT_WAIT_SEC)

        while True:
            audio_batch = AudioRecorder._record_with_defaults(1)
            combined_audio = np.concatenate((initial_audio, audio_batch))

            if np.max(audio_batch) < STT_SILENCE_THRESHOLD:
                break
        
        audio_data = sr.AudioData(
            combined_audio.flatten().tobytes(), STT_SAMPLE_RATE, 2
        )

        audio_segment = pydub.AudioSegment(
            audio_data.get_raw_data(),
            sample_width = audio_data.sample_width,
            frame_rate = audio_data.sample_rate,
            channels = 1,
        )

        buffer = BytesIO()
        buffer.name = "file.mp3"
        audio_segment.export(buffer, format="mp3")

        return buffer

    @staticmethod
    def _record_with_defaults(wait_time_seconds: int) -> np.typing.NDArray[np.float64]:
        audio: np.typing.NDArray[np.float64] = sd.rec(
            int(wait_time_seconds * STT_SAMPLE_RATE),
            samplerate=STT_SAMPLE_RATE,
            channels=1,
            dtype="int16",
        )

        sd.wait()

        return audio


class AudioPlayer:
    def __init__(self, py_audio: pyaudio.PyAudio) -> None:
        self._player = py_audio.open(
            format=pyaudio.paInt16, channels=1, rate=TTS_SAMPLE_RATE, output=True
        )

    def queue_sound(self, audio: bytes) -> None:
        self._player.write(audio)


class TTSProvider:
    def __init__(self, openai: OpenAI, audio_player: AudioPlayer) -> None:
        self._client = openai.audio.speech
        self._audio_player = audio_player

    def text_to_speech(self, text: str) -> None:
        with self._client.with_streaming_response.create(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            input=text,
            response_format="wav",
        ) as response:
            for chunk in response.iter_bytes(chunk_size=1024):
                self._audio_player.queue_sound(chunk)


class STTProvider:
    def __init__(self, client: OpenAI):
        self._client = client.audio.transcriptions

    def speech_to_text(self) -> str:
        audio_bytes = AudioRecorder.record()

        return self._client.create(
            model=STT_MODEL, 
            file=audio_bytes,
        ).text

class MockAssistant:
    def __init__(self, _0, _1, _2) -> None:
        pass

    def prompt(self, _) -> str:
        return "Welcome to Starbucks! What can I get started for you today?"

class AssistantFactory:
    def __init__(self, openai: OpenAI):
        self._client = openai.beta.assistants
    
    def create(self, name: str, instructions: str, model: str):
        return self._client.create(
            name=name,
            instructions=instructions,
            model=model,
        )

    def fetch(self, assistant_id: str):
        return self._client.retrieve(
            assistant_id=assistant_id,
        )

class AssistantClient:
    def __init__(
            self, tts_provider: TTSProvider, openai: OpenAI, assistant: Assistant
    ) -> None:
        self._client = openai.beta.threads
        self._tts_provider = tts_provider
        self._assistant = assistant
        self._thread = self._client.create()

    def prompt(self, message: str) -> str:
        self._client.messages.create(
            thread_id=self._thread.id,
            role="user",
            content=message,
        )

        run = self._client.runs.create_and_poll(
            thread_id=self._thread.id,
            assistant_id=self._assistant.id,
        )

        if run.status == "completed":
            messages = self._client.messages.list(thread_id=self._thread.id)
            message_content = cast(TextContentBlock, messages.data[0].content[0])
            return str(message_content.text.value)
        else:
            return "Please speak to a customer service agent, as our AI assistant is currently unavailable."


class OpenAICredentials(TypedDict):
    api_key: str
    organization: str
    project: str

class CommandLineArguments:
    def __init__(self) -> None:
        Arguments = TypedDict(
            "Arguments",
            {
                "env-path": str,
            },
        )

        self.arguments = cast(
            Arguments,
            {
                key.lstrip("-"): value
                for (key, value) in (argument.split("=") for argument in sys.argv[1:])
            },
        )

    def open_ai_credentials(self) -> OpenAICredentials:
        if self.arguments.get("env-path") is None:
            # TODO: error logging
            raise RuntimeError('ERROR: "--env-path=PATH" argument not supplied.')

        load_dotenv(dotenv_path=self.arguments["env-path"])

        return {
            "api_key": os.environ["OPENAI_API_KEY"],
            "organization": os.environ["OPENAI_ORG_ID"],
            "project": os.environ["OPENAI_PROJ_ID"],
        }


class Main:
    def __init__(self) -> None:
        self._command_line_arguments = CommandLineArguments()
        self._py_audio = pyaudio.PyAudio()
        self._audio_player = AudioPlayer(self._py_audio)
        self._client = OpenAI(**self._command_line_arguments.open_ai_credentials())
        self._stt_provider = STTProvider(self._client)
        self._tts_provider = TTSProvider(self._client, self._audio_player)
        
        self._assistant = AssistantClient(
            self._tts_provider,
            self._client,
            AssistantFactory(self._client).fetch(assistant_id=ASSISTANT_ID),
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
