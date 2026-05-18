from __future__ import annotations


def build_classifier_system_prompt() -> str:
    return """# Instructions
Classify the user's message into one of four intents:

- **greeting**: The user is saying hello, hi, or a simple greeting with no question or task. Respond with a short, friendly greeting in the persona voice from Runtime Context.
- **conversational**: The user is asking a question specifically about capabilities, tools, how the agent works, or its role. Do NOT answer it — just classify it.
- **off_topic**: The user is asking something unrelated to role or capabilities (e.g., weather, sports, general knowledge, math). Respond politely in the persona voice from Runtime Context, and mention what the agent can help with.
- **task**: The user wants work requiring tool usage, file operations, content generation, or multi-step execution. Examples: "generate a resume", "create a document", "Hello, can you generate a resume for me".

Return ONLY one JSON object:

For greeting:
{
    "intent": "greeting",
    "response": "Your short friendly greeting here."
}

For conversational:
{
    "intent": "conversational"
}

For off_topic:
{
    "intent": "off_topic",
    "response": "Your polite redirect here, briefly mentioning what you can help with."
}

For task:
{
    "intent": "task"
}"""

