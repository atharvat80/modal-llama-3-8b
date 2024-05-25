# Run an OpenAI-Compatible vLLM Server
from pathlib import Path

import modal
import modal.runner

MODEL_NAME = "ISTA-DASLab/Meta-Llama-3-8B-Instruct-AQLM-2Bit-1x16"
MODEL_DIR = f"/models/{MODEL_NAME}"
SERVED_NAME = "llama3-8b-instruct"
MINUTES = 60
N_GPU = 1
TOKEN = "super-secret-token"


# Set up the container image
def download_model_to_image(model_dir, model_name):
    import os
    from huggingface_hub import snapshot_download

    os.makedirs(model_dir, exist_ok=True)
    snapshot_download(
        model_name,
        local_dir=model_dir,
        ignore_patterns=["*.pt", "*.bin"],
    )


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm", "huggingface_hub", "hf-transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        download_model_to_image,
        timeout=20 * MINUTES,
        kwargs={"model_dir": MODEL_DIR, "model_name": MODEL_NAME},
    )
)

app = modal.App("llama3-vllm")
local_template_path = Path(__file__).parent / "template_llama3.jinja"


@app.function(
    image=image,
    gpu=modal.gpu.T4(),
    allow_concurrent_inputs=100,
    concurrency_limit=120, # 2 minutes of idle time before shutting down
    enable_memory_snapshot=True,
)
@modal.asgi_app()
def serve():
    import fastapi
    import vllm.entrypoints.openai.api_server as api_server
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
    from vllm.entrypoints.openai.serving_completion import (
        OpenAIServingCompletion,
    )
    from vllm.usage.usage_lib import UsageContext

    app = api_server.app

    # security: CORS middleware for external requests
    app.add_middleware(
        fastapi.middleware.cors.CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # security: auth middleware
    @app.middleware("http")
    async def authentication(request: fastapi.Request, call_next):
        if not request.url.path.startswith("/v1"):
            return await call_next(request)
        if request.headers.get("Authorization") != "Bearer " + TOKEN:
            return fastapi.responses.JSONResponse(
                content={"error": "Unauthorized"}, status_code=401
            )
        return await call_next(request)

    engine_args = AsyncEngineArgs(
        model=MODEL_DIR,
        served_model_name=SERVED_NAME,
        tensor_parallel_size=N_GPU,
        gpu_memory_utilization=0.9,
        enforce_eager=False,
        quantization="aqlm",
        seed=0,
    )

    engine = AsyncLLMEngine.from_engine_args(
        engine_args, 
        usage_context=UsageContext.OPENAI_API_SERVER,
    )

    api_server.openai_serving_chat = OpenAIServingChat(
        engine,
        served_model_names=[SERVED_NAME],
        response_role="assistant",
        # chat_template="chat_template.jinja",
    )
    api_server.openai_serving_completion = OpenAIServingCompletion(
        engine, served_model_names=[SERVED_NAME],
    )

    return app


if __name__ == "__main__":
    modal.runner.deploy_app(app)
