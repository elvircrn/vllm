# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse, Response

import vllm.envs as envs
from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

logger = init_logger(__name__)


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


router = APIRouter()


@router.post("/hot_reload")
async def hot_reload(raw_request: Request):
    """Hot-reload vLLM code from a git branch without reloading model weights.

    Pauses the scheduler, aborts in-flight requests, clears KV cache,
    pulls new code from git, resets torch.compile and CUDA graphs,
    re-warms up the model, and resumes serving.

    Query parameters:
        branch: Git branch/ref to checkout (required)
        remote: Git remote name (default: "origin")
        vllm_source_dir: Path to vLLM source on worker nodes
                         (default: "/opt/vllm-source")
        modules: Comma-separated list of module prefixes to reload
                 (default: "vllm.compilation,vllm.v1.worker.gpu.cudagraph")

    Example:
        POST /hot_reload?branch=my-feature&remote=fork
    """
    branch = raw_request.query_params.get("branch")
    if not branch:
        return JSONResponse(
            status_code=400,
            content={"error": "branch query parameter is required"},
        )
    remote = raw_request.query_params.get("remote", "origin")
    vllm_source_dir = raw_request.query_params.get(
        "vllm_source_dir", "/opt/vllm-source"
    )
    modules_str = raw_request.query_params.get("modules", "")
    module_prefixes = (
        [m.strip() for m in modules_str.split(",") if m.strip()]
        if modules_str
        else None
    )

    logger.info(
        "Hot reload requested: branch=%s, remote=%s, source=%s, modules=%s",
        branch,
        remote,
        vllm_source_dir,
        module_prefixes,
    )

    try:
        await engine_client(raw_request).hot_reload(
            branch=branch,
            remote=remote,
            vllm_source_dir=vllm_source_dir,
            module_prefixes=module_prefixes,
        )
        return Response(status_code=200)
    except Exception as e:
        logger.exception("Hot reload failed")
        return JSONResponse(
            status_code=500,
            content={"error": f"Hot reload failed: {str(e)}"},
        )


def attach_router(app: FastAPI):
    if not envs.VLLM_SERVER_DEV_MODE:
        return

    app.include_router(router)
