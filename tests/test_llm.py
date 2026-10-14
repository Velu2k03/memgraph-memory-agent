import pytest

from memgraph.llm import FakeLLM, LLMError, llm_from_env, strip_code_fences


def test_strip_code_fences_json():
    assert strip_code_fences('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_strip_code_fences_plain():
    assert strip_code_fences("hello") == "hello"


def test_strip_code_fences_unclosed():
    assert strip_code_fences('```\n{"a": 1}') == '{"a": 1}'


def test_fake_llm_pops_in_order():
    llm = FakeLLM(["one", "two"])
    assert llm.generate("p1") == "one"
    assert llm.generate("p2") == "two"
    assert len(llm.calls) == 2
    assert llm.calls[0]["prompt"] == "p1"


def test_fake_llm_empty_raises():
    with pytest.raises(LLMError):
        FakeLLM([]).generate("x")


def test_llm_from_env_prefers_gemini(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from memgraph.llm import GeminiLLM

    assert isinstance(llm_from_env(), GeminiLLM)


def test_llm_from_env_falls_back_to_openai(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    from memgraph.llm import OpenAICompatLLM

    assert isinstance(llm_from_env(), OpenAICompatLLM)


def test_llm_from_env_raises_without_keys(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LLMError):
        llm_from_env()
