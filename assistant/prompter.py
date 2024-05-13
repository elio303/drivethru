from openai import OpenAI
from openai.types.beta.assistant import Assistant
from openai.types.beta.threads import TextContentBlock
from typing import cast

from tts.converter import TTSConverter


class MockAssistantPrompter:
    def __init__(self, _0: TTSConverter, _1: OpenAI, _2: Assistant) -> None:
        pass

    def prompt(self, _: str) -> str:
        return "Welcome to Starbucks! What can I get started for you today?"


class AssistantPrompter:
    def __init__(
        self, tts_provider: TTSConverter, openai: OpenAI, assistant: Assistant
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
