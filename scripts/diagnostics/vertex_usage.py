#!/usr/bin/env python3
"""Minimal Vertex AI connectivity diagnostic.

Credentials and project selection come exclusively from environment variables.
"""

import os

from google import genai
from google.genai import types
from google.oauth2 import service_account


def main() -> int:
    credential_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
    model = os.environ.get("KIKIFAST_VERTEX_MODEL", "gemini-2.5-flash")
    if not credential_path or not project:
        raise SystemExit(
            "Set GOOGLE_APPLICATION_CREDENTIALS and GOOGLE_CLOUD_PROJECT first."
        )

    credentials = service_account.Credentials.from_service_account_file(
        credential_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    client = genai.Client(
        vertexai=True,
        project=project,
        location=location,
        credentials=credentials,
    )
    config = types.GenerateContentConfig(temperature=0.2, max_output_tokens=128)
    for chunk in client.models.generate_content_stream(
        model=model,
        contents="Reply with: Vertex AI connection successful.",
        config=config,
    ):
        if chunk.text:
            print(chunk.text, end="")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
