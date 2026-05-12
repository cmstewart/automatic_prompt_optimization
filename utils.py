
import string

import openai
from openai import OpenAI

client = OpenAI(max_retries=5)


class DailyRateLimitError(Exception):
    """Raised when the daily request limit is exhausted."""
    pass


def parse_sectioned_prompt(s):

    result = {}
    current_header = None

    for line in s.split('\n'):
        line = line.strip()

        if line.startswith('# '):
            # first word without punctuation
            current_header = line[2:].strip().lower().split()[0]
            current_header = current_header.translate(str.maketrans('', '', string.punctuation))
            result[current_header] = ''
        elif current_header is not None:
            result[current_header] += line + '\n'

    return result


def chatgpt(prompt, temperature=0.3, n=1, top_p=1, stop=None, max_tokens=1024,
                  presence_penalty=0, frequency_penalty=0, logit_bias={}, timeout=60):
    """Call the OpenAI chat completions API and return a list of response strings."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            n=n,
            top_p=top_p,
            stop=stop,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            timeout=timeout,
        )
        return [choice.message.content for choice in response.choices]
    except openai.RateLimitError as e:
        msg = str(e)
        if "RPD" in msg or "requests per day" in msg:
            raise DailyRateLimitError(msg) from e
        print(f"Warning: OpenAI rate limit (non-daily): {e}")
        return [""]
    except (openai.APIError, openai.APIConnectionError) as e:
        print(f"Warning: OpenAI API call failed: {e}")
        return [""]


def wrap_prompt(prompt: str) -> str:
    """
    Guarantee that the returned string contains both `{question}` and
    `{context}` placeholders.  If the original already has them, it is
    returned unchanged; otherwise we prepend a minimal header.
    """
    if "{question}" in prompt and "{context}" in prompt:
        return prompt          # nothing to do

    header = (
        "Question: {question}\n"
        "Context:\n{context}\n\n"
    )
    return header + prompt.strip()
