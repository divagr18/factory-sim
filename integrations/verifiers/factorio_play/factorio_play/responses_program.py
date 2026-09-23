# /// script
# requires-python = ">=3.11"
# dependencies = ["openai>=1.66", "mcp>=1.24.0,<2", "httpx", "tenacity"]
# ///
"""factorio-play's agent program: the model plays through the OpenAI Responses API.

verifiers' built-in `null` harness talks Chat Completions. Some reasoning models
take function tools only on `/responses` there, so this program runs the same
loop over `client.responses.create`: reasoning items are carried forward turn to
turn, and each tool result goes back as a `function_call_output`. Requests go to
verifiers' interception endpoint (its Responses dialect), which forwards them
upstream and applies the run's sampling settings.

The MCP plumbing (`mcp_session`, `with_retry`, `connect_mcp`, `call_mcp`) is
adapted from verifiers' `harnesses/null/program.py`:

    Copyright (c) 2026 Prime Intellect. MIT License.
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files, to deal in the Software
    without restriction, subject to including this notice in all copies or
    substantial portions. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
    ANY KIND.
"""

import argparse
import asyncio
import json
from contextlib import AsyncExitStack, asynccontextmanager, suppress
from pathlib import Path

import httpx
from openai import AsyncOpenAI
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential_jitter

MCP_CALL_ATTEMPTS = 6
MCP_TIMEOUT = 600.0


@asynccontextmanager
async def mcp_session(spec: dict):
    """A fresh streamable-HTTP session to an MCP server, opened and closed in this task."""
    from mcp import ClientSession
    from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

    stack = AsyncExitStack()
    try:
        http_client = await stack.enter_async_context(
            create_mcp_http_client(
                headers=spec.get("headers") or None,
                timeout=httpx.Timeout(spec.get("timeout", MCP_TIMEOUT), connect=5.0),
            )
        )
        read, write, *_ = await stack.enter_async_context(
            streamable_http_client(spec["url"], http_client=http_client)
        )
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        yield session
    finally:
        with suppress(Exception):
            await stack.aclose()


async def with_retry(call):
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(MCP_CALL_ATTEMPTS),
        wait=wait_exponential_jitter(initial=0.5, max=30),
        reraise=True,
    ):
        with attempt:
            return await call()


async def connect_mcp(config: dict) -> tuple[list[dict], dict, dict]:
    """Each server's tools as Responses function tools, plus name -> (server, tool)."""
    tools: list[dict] = []
    dispatch: dict[str, tuple] = {}
    servers: dict[str, dict] = {}
    for name, spec in config.get("mcpServers", {}).items():
        servers[name] = spec

        async def list_tools(spec: dict = spec):
            async with mcp_session(spec) as session:
                return (await session.list_tools()).tools

        for tool in await with_retry(list_tools):
            full = f"{name}_{tool.name}" if name else tool.name
            if full in dispatch:
                raise ValueError(f"duplicate tool name {full!r} across servers")
            tools.append(
                {
                    "type": "function",
                    "name": full,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema,
                }
            )
            dispatch[full] = (name, tool.name)
    return tools, dispatch, servers


async def call_mcp(servers: dict, dispatch: dict, name: str, arguments: dict) -> str:
    server_name, raw = dispatch[name]

    async def call():
        async with mcp_session(servers[server_name]) as session:
            return await session.call_tool(raw, arguments)

    result = await with_retry(call)
    parts = [b.text if b.type == "text" else str(b) for b in result.content]
    return "\n".join(parts) if parts else ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--initial-messages-file", default="")
    parser.add_argument("--mcp-config", default="")
    return parser.parse_args()


def _call_output(call_id: str, output: str) -> dict:
    return {"type": "function_call_output", "call_id": call_id, "output": output}


async def main() -> None:
    args = parse_args()
    items: list[dict] = []
    if args.initial_messages_file:
        path = Path(args.initial_messages_file)
        payload = path.read_bytes()
        path.unlink()
        items.extend(json.loads(payload))
    elif args.prompt:
        items.append({"role": "user", "content": args.prompt})
    client = AsyncOpenAI(
        base_url=args.base_url, api_key=args.api_key, timeout=httpx.Timeout(None, connect=5.0)
    )
    config = json.loads(args.mcp_config or "{}")
    if config.get("mcpServers"):
        async with asyncio.timeout(60):
            tools, dispatch, servers = await connect_mcp(config)
    else:
        tools, dispatch, servers = [], {}, {}
    request: dict = {"model": args.model}
    if args.system_prompt:
        request["instructions"] = args.system_prompt
    if tools:
        request["tools"] = tools
    while True:
        response = await client.responses.create(**request, input=items)
        # Every output item goes back as input next turn, reasoning included, so the
        # model keeps its chain of thought across tool calls.
        output = [item.model_dump(exclude_none=True) for item in response.output]
        items.extend(output)
        calls = [item for item in output if item.get("type") == "function_call"]
        if not calls:
            break
        for call in calls:
            name, call_id = call["name"], call["call_id"]
            try:
                tool_args = json.loads(call.get("arguments") or "{}")
            except json.JSONDecodeError as e:
                items.append(_call_output(call_id, f"error: invalid JSON in arguments ({e})"))
                continue
            if not isinstance(tool_args, dict):
                items.append(_call_output(call_id, "error: arguments must be a JSON object"))
                continue
            if name in dispatch:
                content = await call_mcp(servers, dispatch, name, tool_args)
            else:
                content = f"error: unknown tool {name!r}"
            items.append(_call_output(call_id, content))


if __name__ == "__main__":
    asyncio.run(main())
