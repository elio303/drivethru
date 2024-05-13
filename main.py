import numpy.typing
import pyaudio

from openai import OpenAI

from assistant.prompter import AssistantPrompter
from assistant.factory import AssistantFactory
from audio.player import AudioPlayer
from audio.recorder import AudioRecorder
from config.provider import ConfigProvider
from stt.converter import STTConverter
from tools.logger import Logger
from tools.command_line_arguments import CommandLineArguments
from tts.converter import TTSConverter


class Main:
    def __init__(self) -> None:
        command_line_arguments = CommandLineArguments()
        environment = command_line_arguments.environment()
        config = ConfigProvider.load("config.yaml")

        audio_recorder = AudioRecorder(
            config.stt.wait_time, config.stt.silence_threshold, config.stt.sample_rate
        )
        py_audio = pyaudio.PyAudio()
        audio_player = AudioPlayer(py_audio, config.tts.sample_rate)
        client = OpenAI(**command_line_arguments.open_ai_credentials())
        assistant = AssistantFactory(client).fetch(assistant_id=config.assistant.id)

        self._logger = Logger(environment)
        self._stt_provider = STTConverter(client, audio_recorder, config.stt.model)
        self._tts_provider = TTSConverter(
            client, audio_player, config.tts.model, config.tts.voice
        )

        self._assistant = AssistantPrompter(self._tts_provider, client, assistant)

    def run(self) -> None:
        user_prompt = "Hello."

        while True:
            response = self._assistant.prompt(user_prompt)
            self._logger.debug(f"\nassistant > {response}")
            self._tts_provider.text_to_speech(response)

            user_prompt = self._stt_provider.speech_to_text()
            self._logger.debug(f"\ncustomer > {user_prompt}")


Main().run()
