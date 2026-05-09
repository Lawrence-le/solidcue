
from abc import ABC, abstractmethod
from typing import Dict, List


class BaseProvider(ABC):
    @abstractmethod
    def generate(self, messages: list[dict]) -> str:
        """Send chat messages to LLM and return normalized response."""
        pass
