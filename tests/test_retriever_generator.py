import pytest

from rag.config import FALLBACK_ANSWER
from rag.exceptions import EmptyQueryError, GenerationError, RAGError
from rag.generator import AnswerGenerator
from rag.pipeline import compute_confidence
from rag.retriever import RetrievedChunk, Retriever, build_page_filter


class FakeEmbedder:
    def embed_query(self, query):
        return [0.0, 1.0]


class FakeStore:
    def __init__(self, data):
        self.data, self.calls = data, []

    def query(self, namespace, vector, top_k, metadata_filter=None):
        self.calls.append((namespace, top_k, metadata_filter))
        return self.data.get(namespace, [])[:top_k]


def match(i, score, page=1, doc="d.pdf"):
    return {"id": i, "score": score,
            "metadata": {"chunk_id": i, "doc_name": doc, "page": page, "text": f"text {i}"}}


def test_empty_query_raises():
    r = Retriever(FakeStore({}), FakeEmbedder())
    for q in ("", "   ", None):
        with pytest.raises(EmptyQueryError):
            r.retrieve(q, ["ns"])


def test_no_documents_selected_raises():
    with pytest.raises(RAGError):
        Retriever(FakeStore({}), FakeEmbedder()).retrieve("hi", [])


def test_threshold_filters_and_best_score_reported():
    store = FakeStore({"ns": [match("a", 0.8), match("b", 0.35), match("c", 0.1)]})
    res = Retriever(store, FakeEmbedder()).retrieve("q", ["ns"], top_k=3, threshold=0.3)
    assert [h.chunk_id for h in res.hits] == ["a", "b"]
    assert res.best_score == 0.8


def test_everything_below_threshold_gives_no_hits():
    store = FakeStore({"ns": [match("a", 0.2)]})
    res = Retriever(store, FakeEmbedder()).retrieve("q", ["ns"], threshold=0.5)
    assert res.hits == [] and res.best_score == 0.2


def test_multi_namespace_results_are_merged_and_sorted():
    store = FakeStore({"n1": [match("a", 0.6)], "n2": [match("b", 0.9, doc="e.pdf")]})
    res = Retriever(store, FakeEmbedder()).retrieve("q", ["n1", "n2"], top_k=2, threshold=0.0)
    assert [h.chunk_id for h in res.hits] == ["b", "a"]
    assert [c[0] for c in store.calls] == ["n1", "n2"]


def test_page_filter_is_passed_to_store():
    store = FakeStore({"ns": []})
    Retriever(store, FakeEmbedder()).retrieve("q", ["ns"], page_range=(5, 2))
    assert store.calls[0][2] == {"page": {"$gte": 2, "$lte": 5}}
    assert build_page_filter(None) is None


class FakeClient:
    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.called = reply, error, False
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.called = True
        if self.error:
            raise self.error

        class Msg:  # minimal response shape
            content = self.reply

        class Choice:
            message = Msg()

        class Resp:
            choices = [Choice()]

        return Resp()


HITS = [RetrievedChunk("c1", "d.pdf", 3, "The sky is blue.", 0.7)]


def test_no_hits_returns_fallback_without_calling_llm():
    client = FakeClient("should never be used")
    out = AnswerGenerator("k", "m", client=client).generate("q", [])
    assert out.text == FALLBACK_ANSWER and not out.answered and not client.called


def test_llm_fallback_is_normalised():
    client = FakeClient(f'  "{FALLBACK_ANSWER}" ')
    out = AnswerGenerator("k", "m", client=client).generate("q", HITS)
    assert out.text == FALLBACK_ANSWER and not out.answered


def test_answer_with_citations_parsed():
    client = FakeClient("The sky is blue [1]. Also [7] is ignored.")
    out = AnswerGenerator("k", "m", client=client).generate("q", HITS)
    assert out.answered and out.cited == [1]


def test_llm_failure_is_wrapped():
    client = FakeClient(error=RuntimeError("boom"))
    with pytest.raises(GenerationError):
        AnswerGenerator("k", "m", client=client).generate("q", HITS)


def test_confidence_labels():
    mk = lambda s: [RetrievedChunk("c", "d", 1, "t", s)]
    assert compute_confidence([]) == (None, "N/A")
    assert compute_confidence(mk(0.7)) == (70.0, "High")
    assert compute_confidence(mk(0.45))[1] == "Medium"
    assert compute_confidence(mk(0.31))[1] == "Low"
