import asyncio
from copy import deepcopy
import json

import pytest
from fastapi.responses import JSONResponse, StreamingResponse

from src.converter.anti_truncation import AntiTruncationStreamProcessor


DONE = b"data: [DONE]\n\n"


def chunk(parts=(), finish=None, **metadata):
    candidate = {"content": {"role": "model", "parts": list(parts)}}
    if finish:
        candidate["finishReason"] = finish
    return f"data: {json.dumps({'candidates': [candidate], **metadata})}\n\n".encode()


def payload():
    return {"request": {"contents": [{"role": "user", "parts": [{"text": "hello"}]}]}}


def collect(attempts, **options):
    requests = []

    async def request(body):
        requests.append(deepcopy(body))
        current = attempts[min(len(requests) - 1, len(attempts) - 1)]
        if isinstance(current, JSONResponse):
            return current

        async def generate():
            for item in current:
                if isinstance(item, Exception):
                    raise item
                yield item

        return StreamingResponse(generate())

    processor = AntiTruncationStreamProcessor(request, payload(), **options)

    async def run():
        return [item async for item in processor.process_stream()]

    return asyncio.run(run()), requests


def texts(output):
    result = []
    for item in output:
        if item == DONE:
            continue
        data = json.loads(item.decode()[6:])
        for candidate in data.get("candidates", []):
            result.extend(
                part["text"]
                for part in candidate.get("content", {}).get("parts", [])
                if "text" in part and not part.get("thought")
            )
    return "".join(result)


def test_plain_text_is_yielded_before_upstream_advances():
    first = chunk([{"text": "hello"}])

    async def request(_):
        async def generate():
            yield first
            raise AssertionError("upstream advanced before caller received its first chunk")

        return StreamingResponse(generate())

    async def run():
        stream = AntiTruncationStreamProcessor(request, payload()).process_stream()
        try:
            assert await anext(stream) == first
        finally:
            await stream.aclose()

    asyncio.run(run())


@pytest.mark.parametrize(
    "parts", [[{"text": "hello"}], [{"functionCall": {"name": "lookup", "args": {}}}]]
)
def test_real_output_without_synthetic_call_is_not_retried(parts):
    first = chunk(parts)
    output, requests = collect([[first, DONE]])
    assert len(requests) == 1
    assert output == [first, DONE]


def test_tool_finish_and_usage_are_preserved_once():
    first = chunk([{"functionCall": {"name": "lookup", "args": {}}}])
    finish = chunk([{"text": ""}], "STOP", usageMetadata={"totalTokenCount": 5})
    output, requests = collect([[first, finish, DONE]])
    assert len(requests) == 1
    assert output == [first, finish, DONE]


@pytest.mark.parametrize("with_synthetic", [False, True])
def test_interrupted_tool_turn_is_not_replayed(with_synthetic):
    parts = [{"functionCall": {"name": "lookup", "args": {}}}]
    if with_synthetic:
        parts.insert(0, {"functionCall": {"name": "emit_answer", "args": {"content": "answer"}}})
    tool_chunk = chunk(parts)
    output, requests = collect(
        [[tool_chunk, RuntimeError("offline interruption")], [tool_chunk, DONE]]
    )
    assert len(requests) == 1
    data = [json.loads(item.decode()[6:]) for item in output if item != DONE]
    calls = [
        part["functionCall"]
        for item in data
        for candidate in item.get("candidates", [])
        for part in candidate.get("content", {}).get("parts", [])
        if "functionCall" in part
    ]
    assert calls == [{"name": "lookup", "args": {}}]
    assert data[-1]["error"]["code"] == 500
    assert "offline interruption" in data[-1]["error"]["message"]
    assert output[-1] == DONE


@pytest.mark.parametrize("trailing_text", ["", "duplicate"])
def test_synthetic_finish_strips_text_and_keeps_usage(trailing_text):
    first = chunk([{"functionCall": {"name": "emit_answer", "args": {"content": "answer"}}}])
    finish = chunk([{"text": trailing_text}], "STOP", usageMetadata={"totalTokenCount": 5})
    output, requests = collect([[first, finish, DONE]])
    assert len(requests) == 1
    assert texts(output) == "answer"
    final = json.loads(output[-2].decode()[6:])
    assert final["candidates"][0]["finishReason"] == "STOP"
    assert final["usageMetadata"]["totalTokenCount"] == 5


def test_separate_usage_chunk_survives_synthetic_answer():
    first = chunk([{"functionCall": {"name": "emit_answer", "args": {"content": "answer"}}}])
    usage = b'data: {"usageMetadata":{"totalTokenCount":5}}\n\n'
    output, _ = collect([[first, usage, DONE]])
    assert usage in output


def test_thought_only_continuation_uses_prompt_without_empty_prefill():
    thought = chunk([{"text": "thinking", "thought": True}])
    output, requests = collect(
        [[thought, DONE], [chunk([{"text": "answer"}]), DONE]], enable_prefill_mode=True
    )
    assert len(requests) == 2
    assert thought in output
    assert texts(output) == "answer"
    assert requests[1]["request"]["contents"][-1]["role"] == "user"
    assert len(requests[1]["request"]["contents"]) == 2


def test_interrupted_text_enters_continuation_history_without_replay():
    output, requests = collect(
        [
            [chunk([{"text": "partial"}]), RuntimeError("offline interruption")],
            [chunk([{"text": " remainder"}]), DONE],
        ]
    )
    assert len(requests) == 2
    assert texts(output) == "partial remainder"
    assert requests[1]["request"]["contents"][-2] == {
        "role": "model",
        "parts": [{"text": "partial"}],
    }


def test_exhausted_interruption_does_not_repeat_already_streamed_text():
    output, requests = collect(
        [[chunk([{"text": "partial"}]), RuntimeError("offline interruption")]], max_attempts=1
    )
    assert len(requests) == 1
    assert texts(output) == "partial"
    assert output[-1] == DONE


@pytest.mark.parametrize(
    "parts", [[{"text": "answer"}], [{"functionCall": {"name": "lookup", "args": {}}}]]
)
def test_non_streaming_real_output_is_not_retried(parts):
    response = JSONResponse({"candidates": [{"content": {"parts": parts}}]})
    output, requests = collect([response])
    assert len(requests) == 1
    assert json.loads(output[0].decode()[6:])["candidates"][0]["content"]["parts"] == parts


def test_upstream_error_is_not_retried():
    response = JSONResponse({"error": {"message": "offline failure"}}, status_code=429)
    output, requests = collect([response])
    assert len(requests) == 1
    assert json.loads(output[0].decode()[6:])["error"]["message"] == "offline failure"


def test_continuation_does_not_mutate_or_accumulate_original_history():
    original = payload()
    saved = deepcopy(original)
    processor = AntiTruncationStreamProcessor(None, original)
    processor._append_content("partial")
    processor.current_attempt = 2
    second = processor._build_current_payload()
    processor.current_attempt = 3
    third = processor._build_current_payload()
    assert second == third
    assert len(second["request"]["contents"]) == 3
    assert original == saved
    assert processor.base_payload == saved
