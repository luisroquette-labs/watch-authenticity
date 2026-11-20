from dataset.media_reuse import is_known_official_media_reuse


def test_known_official_media_reuse_checks_raw_and_derivative_hashes():
    clean = "0" * 64
    assert is_known_official_media_reuse(
        "f6552812a866a41d8285ef2b5997aa66b9370414ea26bd5e3cb55faf0f112c98", clean
    )
    assert is_known_official_media_reuse(
        clean, "d5d939207ffe0192d12e403d5cc9d26be72e498c4df4bab97b186c3a5df01c24"
    )
    assert is_known_official_media_reuse(
        "5ad455e8b1e4b19eeded1417fb59a312e929b2a81f94885bc4e0849e1951bc1a", clean
    )
    assert not is_known_official_media_reuse(clean, clean)
