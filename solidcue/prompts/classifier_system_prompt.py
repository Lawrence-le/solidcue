from __future__ import annotations


def build_classifier_system_prompt() -> str:
    return """# Instructions
Classify the user's message into one of four intents:

- **greeting**: The user is saying hello, hi, or a simple greeting with no question or task. Respond with a short, friendly greeting in the persona voice from Runtime Context.
- **conversational**: The user is asking a question specifically about capabilities, tools, how the agent works, or its role. Do NOT answer it — just classify it.
- **off_topic**: The user is asking something unrelated to role or capabilities (e.g., weather, sports, general knowledge, math). Respond politely in the persona voice from Runtime Context, and mention what the agent can help with.
- **task**: The user wants work requiring tool usage, file operations, content generation, or multi-step execution. Examples: "generate a resume", "create a document", "Hello, can you generate a resume for me".

Priority rules:
- Classify based primarily on the LATEST user message, not prior assistant responses or capability descriptions.
- Treat prior conversation only as light background context; do not let it override a clear task request in the latest message.
- If the latest message asks to generate, create, build, tailor, draft, update, archive, upload, download, extract, or process something, classify it as **task**.
- If the latest message includes a URL, file path, document name, or tracker/update instruction as part of requested work, classify it as **task**.
- Capability questions like "what can you do" are **conversational**, but a follow-up request to perform work is **task** even if earlier turns discussed capabilities.

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
