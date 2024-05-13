import yaml

from types import SimpleNamespace


class ConfigProvider:
    @staticmethod
    def load(path: str) -> SimpleNamespace:
        with open(path) as file:
            dictionary = yaml.load(file, Loader=yaml.FullLoader)
            return SimpleNamespace(**dictionary)
