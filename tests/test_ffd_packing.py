import numpy as np
from haag_vq.methods.ffd_packing import ffd_layout, ffd_encode, ffd_decode


def test_layout_each_dim_within_one_byte():
    rng = np.random.default_rng(0)
    b = rng.integers(0, 9, size=300)
    byte_idx, bit_off, n_bytes = ffd_layout(b)
    for d in range(len(b)):
        if b[d] == 0:
            assert byte_idx[d] == -1
        else:
            assert 0 <= bit_off[d] and bit_off[d] + b[d] <= 8  # fits in one byte


def test_byte_count_between_dense_and_naive():
    rng = np.random.default_rng(1)
    b = rng.integers(1, 9, size=500)
    _, _, n_bytes = ffd_layout(b)
    dense = int(np.ceil(b.sum() / 8))
    naive = int(np.sum(np.ceil(b / 8)))  # one byte per dim (all b<=8 -> = count)
    assert dense <= n_bytes <= naive


def test_no_two_dims_overlap_in_a_byte():
    rng = np.random.default_rng(2)
    b = rng.integers(0, 9, size=200)
    byte_idx, bit_off, n_bytes = ffd_layout(b)
    # for each byte, the placed [offset, offset+width) intervals must be disjoint
    for bi in range(n_bytes):
        intervals = sorted((bit_off[d], bit_off[d] + b[d]) for d in range(len(b)) if byte_idx[d] == bi)
        for (s1, e1), (s2, e2) in zip(intervals, intervals[1:]):
            assert e1 <= s2


def test_encode_decode_roundtrip():
    rng = np.random.default_rng(3)
    D, N = 256, 1000
    b = rng.integers(0, 9, size=D)
    codes = np.zeros((N, D), dtype=np.int64)
    for d in range(D):
        if b[d] > 0:
            codes[:, d] = rng.integers(0, 1 << int(b[d]), size=N)
    byte_idx, bit_off, n_bytes = ffd_layout(b)
    packed = ffd_encode(codes, b, byte_idx, bit_off, n_bytes)
    assert packed.shape == (N, n_bytes) and packed.dtype == np.uint8
    out = ffd_decode(packed, b, byte_idx, bit_off, D)
    np.testing.assert_array_equal(out, codes)


def test_all_widths_1_to_8_roundtrip():
    # stress each width
    for width in range(1, 9):
        D, N = 40, 50
        b = np.full(D, width)
        rng = np.random.default_rng(width)
        codes = rng.integers(0, 1 << width, size=(N, D)).astype(np.int64)
        bi, bo, nb = ffd_layout(b)
        out = ffd_decode(ffd_encode(codes, b, bi, bo, nb), b, bi, bo, D)
        np.testing.assert_array_equal(out, codes)
