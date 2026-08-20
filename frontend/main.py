"""FastAPI proxy for TerrainComm Agent (Agent Runtime, A2A protocol)."""

import os
import uuid

import google.auth
import google.auth.transport.requests
import httpx
from a2a.client import ClientConfig, create_client
from a2a.types import Message, Part, Role, SendMessageRequest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from google.protobuf.json_format import MessageToDict

RESOURCE = os.environ.get(
    "AGENT_ENGINE_RESOURCE_NAME",
    "projects/432975831710/locations/us-east1/reasoningEngines/5082259402929471488",
)
AGENT_DIRECTORY = os.environ.get("AGENT_DIRECTORY", "app")
LOCATION = RESOURCE.split("/locations/")[1].split("/")[0]

A2A_BASE = (
    f"https://{LOCATION}-aiplatform.googleapis.com/reasoningEngines/v1/"
    f"{RESOURCE}/api/a2a/{AGENT_DIRECTORY}"
)

_A2UI_MIME = "application/json+a2ui"

_creds, _ = google.auth.default(
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)


def _auth_headers() -> dict[str, str]:
    _creds.refresh(google.auth.transport.requests.Request())
    return {
        "Authorization": f"Bearer {_creds.token}",
        "Content-Type": "application/json",
    }


app = FastAPI()


@app.exception_handler(Exception)
async def _json_errors(request: Request, exc: Exception):
    return JSONResponse(
        status_code=200,
        content={
            "parts": [{"kind": "text", "text": f"Error: {type(exc).__name__}: {exc}"}]
        },
    )


_contexts: dict[str, str] = {}


def _extract_parts(parts_list: list) -> list[dict]:
    out = []
    for p in parts_list:
        root = p.get("data") if isinstance(p.get("data"), dict) else p
        meta = root.get("metadata") or {}
        mime = meta.get("mimeType") if isinstance(meta, dict) else None

        if mime == _A2UI_MIME:
            data = root.get("data")
            if data:
                out.append({"kind": "a2ui", "data": data})
        elif "text" in root and root["text"]:
            out.append({"kind": "text", "text": root["text"]})
        elif "text" in p and p["text"]:
            out.append({"kind": "text", "text": p["text"]})
        elif "data" in root and root["data"]:
            out.append({"kind": "text", "text": str(root["data"])})
    return out


@app.post("/chat")
async def chat(req: Request):
    body = await req.json()
    message = body.get("message", "")
    user_id = body.get("user_id") or "web-user"
    parts: list[dict] = []

    headers = _auth_headers()
    async with httpx.AsyncClient(headers=headers, timeout=120) as client:
        config = ClientConfig(httpx_client=client)
        a2a_client = await create_client(
            A2A_BASE, client_config=config, resolver_http_kwargs={"headers": headers}
        )

        msg_kwargs = {
            "message_id": str(uuid.uuid4()),
            "role": Role.ROLE_USER,
            "parts": [Part(text=message)],
        }
        if _contexts.get(user_id):
            msg_kwargs["context_id"] = _contexts[user_id]

        msg = Message(**msg_kwargs)
        send_req = SendMessageRequest(message=msg)

        last_task = None
        got_artifact_update = False

        async for event in a2a_client.send_message(send_req):
            d = MessageToDict(event)
            if "task" in d:
                last_task = d["task"]
                if d["task"].get("contextId"):
                    _contexts[user_id] = d["task"]["contextId"]
            if "artifactUpdate" in d:
                got_artifact_update = True
                art = d["artifactUpdate"].get("artifact", {})
                parts.extend(_extract_parts(art.get("parts", [])))

        if not got_artifact_update and last_task is not None:
            for artifact in last_task.get("artifacts", []):
                parts.extend(_extract_parts(artifact.get("parts", [])))

    if not parts:
        parts = [{"kind": "text", "text": "(The agent didn't return a reply.)"}]
    return JSONResponse({"parts": parts})


STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
