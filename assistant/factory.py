from openai import OpenAI
from openai.types.beta.assistant import Assistant


class AssistantFactory:
    def __init__(self, openai: OpenAI) -> None:
        self._client = openai.beta.assistants

    def create(self, name: str, instructions: str, model: str) -> Assistant:
        return self._client.create(
            name=name,
            instructions=instructions,
            model=model,
        )

    def fetch(self, assistant_id: str) -> Assistant:
        return self._client.retrieve(
            assistant_id=assistant_id,
        )
