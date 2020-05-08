from io import StringIO
from typing import Any, Callable, Dict, Optional


class InMemoryFile(StringIO):
    def __init__(self, value: str, save_value: Callable, file: str) -> None:
        super().__init__(value)
        self.save_value = save_value
        self.file = file

    def close(self) -> None:
        self.save_value(self.getvalue(), self.file)
        super().close()


class OpenMock:
    """
    Mocks the `open` built-in function. A call to the instance of OpenMock returns an in-memory file which is
    readable and writable. The actual in-memory file implementation should call the passed `save_value` method
    to save the file content in the cache when the file is being closed to preserve the file content.
    """

    def __init__(self, values: Dict[str, str] = {}) -> None:
        self.values = values

    def __call__(self, file, *args, **kwargs):
        value = self.values.get(file)
        return InMemoryFile(value, self.save_value, file)

    def save_value(self, value: str, file: str) -> None:
        self.values[file] = value
