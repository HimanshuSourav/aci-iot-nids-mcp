from schemas import schema_for


def test_schema_for_rejects_unknown() -> None:
    try:
        schema_for("tinyml")
    except ValueError as exc:
        assert "tinyml" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_schema_for_full_cic_has_example() -> None:
    info = schema_for("full-cic")
    assert info["name"] == "full_cic"
    assert "Flow Duration" in info["example"]
