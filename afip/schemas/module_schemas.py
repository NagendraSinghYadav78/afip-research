"""
MODULE_SCHEMA definitions.

Paper reference: Section 5.1 "Shared Algorithm Conventions and Module Schema
Definitions" — "The MODULE_SCHEMA directory maps each module name to its
Anthropic tool definition. These schemas enforce structured output and
replace free-text generation with schema-constrained JSON."

Each entry below is a real JSON Schema (draft-07 compatible, and directly
usable as an Anthropic `tools[].input_schema`) for one of the four AFIP
modules (M1-M4). `jsonschema.validate` is used at runtime by
afip.algorithms.master_orchestration.ValidateSchema to reproduce
Algorithm 1, Step 4/5.
"""

M1_SENTIMENT_SCHEMA = {
    "name": "M1_SENTIMENT",
    "description": "Structured real-time equity sentiment signal.",
    "input_schema": {
        "type": "object",
        "properties": {
            "signal": {"type": "number", "minimum": -1.0, "maximum": 1.0,
                       "description": "EWMA sentiment signal S_t"},
            "direction": {"type": "string", "enum": ["POSITIVE", "NEGATIVE", "NEUTRAL"]},
            "alert_fired": {"type": "boolean"},
            "top_entities": {"type": "array", "items": {"type": "string"}, "maxItems": 10},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "required": ["signal", "direction", "alert_fired", "top_entities", "confidence"],
        "additionalProperties": False,
    },
}

M2_EARNINGS_SCHEMA = {
    "name": "M2_EARNINGS",
    "description": "Structured earnings-call / report summarization output.",
    "input_schema": {
        "type": "object",
        "properties": {
            "revenue_gaap": {"type": "number"},
            "eps_non_gaap": {"type": "number"},
            "guidance_low": {"type": "number"},
            "guidance_high": {"type": "number"},
            "summary": {"type": "string", "maxLength": 2000},
            "full_document_processed": {"type": "boolean"},
            "key_risks": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["revenue_gaap", "eps_non_gaap", "guidance_low", "guidance_high",
                      "summary", "full_document_processed"],
        "additionalProperties": False,
    },
}

M3_RISK_SCHEMA = {
    "name": "M3_RISK",
    "description": "Structured portfolio risk assessment output.",
    "input_schema": {
        "type": "object",
        "properties": {
            "var99": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "cvar99": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "beta": {"type": "number"},
            "sharpe": {"type": "number"},
            "ips_violations": {"type": "array", "items": {"type": "string"}},
            "rebalancing_options": {"type": "array", "items": {"type": "string"}},
            "advisor_review_required": {"type": "boolean"},
        },
        "required": ["var99", "cvar99", "beta", "sharpe", "ips_violations",
                      "advisor_review_required"],
        "additionalProperties": False,
    },
}

M4_COMPLIANCE_SCHEMA = {
    "name": "M4_COMPLIANCE",
    "description": "Structured regulatory compliance document review output.",
    "input_schema": {
        "type": "object",
        "properties": {
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "severity": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
                        "description": {"type": "string"},
                        "regulation": {"type": "string"},
                    },
                    "required": ["severity", "description"],
                },
            },
            "attorney_review_flag": {"type": "boolean"},
        },
        "required": ["findings", "attorney_review_flag"],
        "additionalProperties": False,
    },
}

MODULE_SCHEMA = {
    "M1_SENTIMENT": M1_SENTIMENT_SCHEMA,
    "M2_EARNINGS": M2_EARNINGS_SCHEMA,
    "M3_RISK": M3_RISK_SCHEMA,
    "M4_COMPLIANCE": M4_COMPLIANCE_SCHEMA,
}
