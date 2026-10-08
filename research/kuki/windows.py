"""Cover every content token once in aggregation, retaining global offsets."""

from collections import Counter


def window_ranges(length, content_size, overlap):
    if not (0 <= overlap < content_size):
        raise ValueError("Need 0 <= overlap < content window size")
    if length <= 0:
        raise ValueError("Empty token sequence")
    ranges = []
    start = 0
    while start < length:
        end = min(start + content_size, length)
        ranges.append((start, end))
        if end == length:
            break
        start = end - overlap
    return ranges


def tokenize_document(tokenizer, text, max_length=1024, overlap=128):
    encoded = tokenizer(
        text, add_special_tokens=False, return_offsets_mapping=True, truncation=False, verbose=False
    )
    ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    special = tokenizer.num_special_tokens_to_add(pair=False)
    ranges = window_ranges(len(ids), max_length - special, overlap)
    coverage = Counter(i for start, end in ranges for i in range(start, end))
    # Tokenizers >=0.23.1 may stop reading text early with truncation=True,
    # leaving incomplete overflow (upstream issue huggingface/tokenizers#2091).
    # Split the already complete encoding instead, then add model specials.
    full = encoded.encodings[0]
    full.truncate(max_length - special, stride=overlap, direction="right")
    pieces = [full, *full.overflowing]
    assert len(pieces) == len(ranges)
    windows = []
    for k, (start, end) in enumerate(ranges):
        piece = tokenizer.backend_tokenizer.post_process(pieces[k], add_special_tokens=True)
        token_ids = piece.ids
        mask = piece.special_tokens_mask
        assert [t for t, special in zip(token_ids, mask, strict=True) if not special] == ids[
            start:end
        ]
        assert [
            tuple(o) for o, special in zip(piece.offsets, mask, strict=True) if not special
        ] == [tuple(o) for o in offsets[start:end]]
        weights = []
        index = start
        for flag in mask:
            weights.append(0.0 if flag else 1.0 / coverage[index])
            if not flag:
                index += 1
        assert index == end and len(weights) == len(token_ids) <= max_length
        windows.append(dict(input_ids=token_ids, weights=weights, start=start, end=end))
    assert abs(sum(sum(w["weights"]) for w in windows) - len(ids)) < 1e-6
    return dict(windows=windows, offsets=offsets, tokens=len(ids))
