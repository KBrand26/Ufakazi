from ufakazi.providers import Provider, Response


def test_response_defaults_to_no_logprob():
    """A provider without logprob support yields a None choice_logprob."""
    r = Response(text="A")
    assert r.choice_logprob is None
    assert r.text == "A"


def test_provider_protocol_is_importable():
    assert Provider is not None
