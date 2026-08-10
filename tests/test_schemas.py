import pytest
from jsonschema import validate, ValidationError

from afip.schemas.module_schemas import MODULE_SCHEMA


@pytest.mark.parametrize("module", list(MODULE_SCHEMA.keys()))
def test_schema_has_required_fields(module):
    schema = MODULE_SCHEMA[module]["input_schema"]
    assert "required" in schema
    assert set(schema["required"]).issubset(set(schema["properties"].keys()))


def test_m1_sentiment_valid_instance_passes():
    schema = MODULE_SCHEMA["M1_SENTIMENT"]["input_schema"]
    instance = {
        "signal": 0.5, "direction": "POSITIVE", "alert_fired": True,
        "top_entities": ["NVDA"], "confidence": 0.8,
    }
    validate(instance=instance, schema=schema)  # should not raise


def test_m1_sentiment_invalid_direction_fails():
    schema = MODULE_SCHEMA["M1_SENTIMENT"]["input_schema"]
    instance = {
        "signal": 0.5, "direction": "BULLISH",  # not in enum
        "alert_fired": True, "top_entities": [], "confidence": 0.8,
    }
    with pytest.raises(ValidationError):
        validate(instance=instance, schema=schema)


def test_m1_sentiment_out_of_range_signal_fails():
    schema = MODULE_SCHEMA["M1_SENTIMENT"]["input_schema"]
    instance = {
        "signal": 1.5,  # out of [-1, 1]
        "direction": "POSITIVE", "alert_fired": True,
        "top_entities": [], "confidence": 0.8,
    }
    with pytest.raises(ValidationError):
        validate(instance=instance, schema=schema)


def test_m2_earnings_missing_required_field_fails():
    schema = MODULE_SCHEMA["M2_EARNINGS"]["input_schema"]
    instance = {
        "revenue_gaap": 100.0, "eps_non_gaap": 1.0,
        "guidance_low": 90.0, "guidance_high": 110.0,
        # missing "summary" and "full_document_processed"
    }
    with pytest.raises(ValidationError):
        validate(instance=instance, schema=schema)


def test_m4_compliance_valid_instance_passes():
    schema = MODULE_SCHEMA["M4_COMPLIANCE"]["input_schema"]
    instance = {
        "findings": [{"severity": "HIGH", "description": "x", "regulation": "Reg S-K"}],
        "attorney_review_flag": True,
    }
    validate(instance=instance, schema=schema)
