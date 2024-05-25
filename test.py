import asyncio
import gzip
import json

import modal
from openai import AsyncOpenAI
from tqdm.auto import tqdm

WORKSPACE = modal.config._profile

client = AsyncOpenAI(
    base_url=f"https://{WORKSPACE}--llama3-vllm-serve.modal.run/v1",
    api_key="super-secret-token",
)

sem = asyncio.Semaphore(100)
model = "llama3-8b-instruct"


async def summarize(text):
    async with sem:
        try:
            return await client.completions.create(
                model=model,
                prompt="Summarize the following text in a single sentence:\n\n"
                + text
                + "\n\nHere is a one line summary of the text:",
                max_tokens=128,
                top_p=0.95,
                temperature=0.3,
            )
        except Exception as e:
            print(f"Error: {e}")
            return


async def summarizev2(text):
    async with sem:
        try:
            return await client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": "Summarize the following text in a single sentence:\n\n"
                        + text,
                    },
                ],
                max_tokens=128,
                top_p=0.95,
                temperature=0.3,
            )
        except Exception as e:
            print(f"Error: {e}")
            return


async def main(texts):
    inp_tokens = 0
    out_tokens = 0
    summaries = []
    tasks = [summarizev2(text) for text in texts]
    for task in tqdm(asyncio.as_completed(tasks)):
        try:
            result = await task
            inp_tokens += result.usage.prompt_tokens
            out_tokens += result.usage.completion_tokens
            text = result.choices[0].message.content.strip()
            text = text.removeprefix("Here is a summary of the text in a single sentence:")
            summaries.append(text.strip())
            # summaries.append(result.choices[0].text.strip())
        except Exception as e:
            print(f"Error: {e}")
            summaries.append("Error")

    return summaries, inp_tokens, out_tokens


if __name__ == "__main__":
    with gzip.open("transaction-text.jsonl.gz", "rt") as f:
        data = [json.loads(line) for line in f]
        texts = [d["text"] for d in data[:100]]

    summaries, inp_tokens, out_tokens = asyncio.run(main(texts))
    print(f"Input tokens: {inp_tokens}")
    print(f"Output tokens: {out_tokens}")
    print("Total tokens:", inp_tokens + out_tokens)
    print(*["- " + i.replace("\n", " ") for i in summaries], sep="\n")
