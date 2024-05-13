import os
import sys

from dotenv import load_dotenv
from typing import Literal, TypedDict, cast

type Environment = Literal["DEV", "PROD"]


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

        if self.arguments.get("env-path") is None:
            # TODO: error logging
            raise RuntimeError('ERROR: "--env-path=PATH" argument not supplied.')

        load_dotenv(dotenv_path=self.arguments["env-path"])

    def environment(self) -> Environment:
        return cast(Environment, os.environ["ENVIRONMENT"])

    def open_ai_credentials(self) -> OpenAICredentials:
        return {
            "api_key": os.environ["OPENAI_API_KEY"],
            "organization": os.environ["OPENAI_ORG_ID"],
            "project": os.environ["OPENAI_PROJ_ID"],
        }
