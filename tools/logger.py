from command_line_arguments import Environment


class Logger:
    def __init__(self, env: Environment):
        self._env = env

    def debug(self, message: str) -> None:
        if self._env == "DEV":
            print(message, end="", flush=True)
