from copy import deepcopy

import numpy as np
import pytest

from research.kuki.data import FRAMES, group_duplicates, prepare_document, split_groups
from research.kuki.windows import window_ranges


def raw():
    return dict(
        doc_id="a",
        lang="ru",
        source="s",
        block="x",
        published_at="2026",
        title="A",
        content="Алиса защищала Бориса.",
        n_annotators=1,
        annotators=[1],
        annotations={
            "1": {
                "frames": [],
                "spans": [
                    dict(layer="l1_roles", label="PROTAGONIST", start=0, end=5, text="Алиса"),
                    dict(layer="l1_roles", label="INNOCENT", start=0, end=5, text="Алиса"),
                    dict(layer="l3_persuasion", label="Call", start=6, end=14, text="защищала"),
                    dict(layer="l3_persuasion", label="Call", start=6, end=14, text="защищала"),
                    dict(layer="l1_roles", label="Coded Phrase", start=15, end=21, text="Бориса"),
                ],
            }
        },
    )


def test_policy_preserves_source_and_does_not_guess():
    source = raw()
    before = deepcopy(source)
    d, log = prepare_document(source)
    assert source == before
    assert d["text"] == source["content"]
    answer = d["annotations"][0]
    assert answer["l1_roles"] == [] and answer["l2"] == [0] * 14
    assert len(answer["l3_persuasion"]) == 1
    assert answer["l3_persuasion"][0]["source_span_indices"] == [2, 3]
    assert {e["kind"] for e in log} == {
        "empty_l2_observed_negative",
        "conflicting_l1_roles",
        "duplicate_span",
        "invalid_layer_label",
    }


def test_invalid_offsets_are_fatal():
    source = raw()
    source["annotations"]["1"]["spans"][0]["end"] = 6
    with pytest.raises(ValueError, match="mismatch"):
        prepare_document(source)


@pytest.mark.parametrize("length", [1, 1022, 1023, 1916, 10000])
def test_windows_cover_once_after_weighting(length):
    ranges = window_ranges(length, 1022, 128)
    coverage = np.zeros(length)
    for start, end in ranges:
        coverage[start:end] += 1
    assert (coverage >= 1).all()
    normalized = np.zeros(length)
    for start, end in ranges:
        normalized[start:end] += 1 / coverage[start:end]
    assert np.allclose(normalized, 1)
    assert ranges[-1][1] == length


def test_copy_groups_never_cross_split():
    docs = []
    for i in range(50):
        d, _ = prepare_document(raw())
        d["doc_id"] = str(i)
        d["text"] = f"Новость номер {i}"
        d["source"] = str(i % 3)
        d["annotations"][0]["l2"][i % len(FRAMES)] = 1
        docs.append(d)
    docs[1]["text"] = docs[0]["text"]
    groups, pairs, _ = group_duplicates(docs)
    assert len(pairs) == 1
    first = split_groups(docs, groups)
    assert first == split_groups(docs, groups)
    assert first["0"] == first["1"]
    assert len(first) == 50 and {s for s, g in first.values()} == {"train", "dev", "test"}


def test_actual_tokenizer_keeps_specials_and_original_offsets():
    transformers = pytest.importorskip("transformers")
    from research.kuki.windows import tokenize_document

    try:
        tok = transformers.AutoTokenizer.from_pretrained(
            "deepvk/RuModernBERT-base",
            revision="8aee6f818db9070b635a549580dc64dcfa03e39a",
            local_files_only=True,
        )
    except OSError:
        pytest.skip("Pinned tokenizer is not cached; run ST0 tokenizer audit first")
    text = "Алиса защищала Бориса. " * 30
    encoded = tokenize_document(tok, text, max_length=32, overlap=8)
    assert len(encoded["windows"]) > 1
    assert sum(sum(w["weights"]) for w in encoded["windows"]) == pytest.approx(encoded["tokens"])
    for window in encoded["windows"]:
        assert window["input_ids"][0] == tok.cls_token_id
        assert window["input_ids"][-1] == tok.sep_token_id
        assert window["weights"][0] == window["weights"][-1] == 0
