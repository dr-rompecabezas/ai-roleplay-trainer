#!/usr/bin/env python3
"""
LLM Provider Abstractions
Provides a common interface for different LLM APIs (OpenAI, Anthropic, etc.)
"""

import os
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    @abstractmethod
    def make_call(self, messages: list[dict[str, str]], max_tokens: int = 1000) -> str:
        """Make a call to the LLM API

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text response
        """
        pass


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider"""

    def __init__(self):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = "claude-sonnet-4-20250514"

    def make_call(self, messages: list[dict[str, str]], max_tokens: int = 1000) -> str:
        """Make a call to Anthropic's API"""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=messages,
            )
            if response and hasattr(response, "content") and response.content:
                return response.content[0].text
            else:
                return "API Error: No content returned."
        except Exception as e:
            return f"Error making Anthropic API call: {str(e)}"


class OpenAIProvider(LLMProvider):
    """OpenAI API provider using the correct API syntax"""

    def __init__(self):
        from openai import OpenAI

        self.client = OpenAI()
        self.model = "gpt-5"

    def make_call(self, messages: list[dict[str, str]], max_tokens: int = 1000) -> str:
        """Make a call to OpenAI's API using the responses endpoint"""
        try:
            # Convert conversation to a single input string for the responses API
            # This is a simplified approach - in production you might want more sophisticated conversion
            conversation_text = self._messages_to_text(messages)

            response = self.client.responses.create(
                model=self.model, input=conversation_text
            )

            if hasattr(response, "output_text"):
                return response.output_text
            else:
                return "API Error: No output_text in response."

        except Exception as e:
            return f"Error making OpenAI API call: {str(e)}"

    def _messages_to_text(self, messages: list[dict[str, str]]) -> str:
        """Convert message history to a single text input for OpenAI responses API"""
        text_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                text_parts.append(f"Human: {content}")
            elif role == "assistant":
                text_parts.append(f"Assistant: {content}")

        return "\n\n".join(text_parts)


def create_provider(provider_name: str = None) -> LLMProvider:
    """Factory function to create LLM providers

    Args:
        provider_name: Name of provider ('openai' or 'anthropic').
                      If None, uses LLM_PROVIDER env var or defaults to 'anthropic'

    Returns:
        Configured LLM provider instance
    """
    if provider_name is None:
        provider_name = os.getenv("LLM_PROVIDER", "anthropic")

    provider_name = provider_name.lower().strip()

    if provider_name == "openai":
        return OpenAIProvider()
    elif provider_name == "anthropic":
        return AnthropicProvider()
    else:
        raise ValueError(
            f"Unknown provider: {provider_name}. Supported: 'openai', 'anthropic'"
        )
