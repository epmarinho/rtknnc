"""Offline validator for the keyword subset used by this foundation's schemas.

No remote resolution or optional dependencies. Unknown assertion keywords fail
closed; this is not a general implementation of JSON Schema draft 2020-12.
"""

import math
import re
from datetime import datetime
from pathlib import Path

from .results import canonical_json
from .validation import strict_json_loads

ROOT = Path(__file__).resolve().parent
KEYWORDS = {
    "$schema", "$comment", "title", "description", "default", "$ref", "type", "const",
    "required", "properties", "additionalProperties", "propertyNames", "items",
    "minItems", "uniqueItems", "minimum", "maximum", "minLength", "maxLength", "pattern", "format",
}


def validate_schema(value, path=ROOT / "result.schema.json"):
    path = Path(path).resolve()
    if ROOT not in path.parents:
        raise ValueError("Schema must remain under modern_baselines")
    schema = strict_json_loads(path.read_text())
    _check(value, schema, path, "$")


def _check(value, schema, source, location):
    def fail(message):
        raise ValueError(f"Schema {location}: {message}")

    if set(schema) - KEYWORDS:
        fail("unsupported schema keyword")
    if "$ref" in schema:
        filename, separator, fragment = schema["$ref"].partition("#")
        referenced = (source.parent / filename).resolve() if filename else source
        if ROOT not in referenced.parents or ":" in filename:
            fail("only local schema references are permitted")
        target = strict_json_loads(referenced.read_text())
        if separator and fragment:
            if not fragment.startswith("/"):
                fail("unsupported schema fragment")
            for token in fragment[1:].split("/"):
                target = target[token.replace("~1", "/").replace("~0", "~")]
        _check(value, target, referenced, location)
    kind = schema.get("type")
    number = type(value) in (int, float) and math.isfinite(value)
    matches = {
        "object": isinstance(value, dict), "array": isinstance(value, list),
        "string": isinstance(value, str), "boolean": type(value) is bool,
        "number": number, "integer": number and int(value) == value,
    }
    if kind is not None and not matches.get(kind, False):
        fail(f"expected {kind}")
    if "const" in schema:
        constant = schema["const"]
        if value != constant or (type(value) is bool) != (type(constant) is bool):
            fail("constant mismatch")
    if isinstance(value, dict):
        if not set(schema.get("required", ())).issubset(value):
            fail("missing required fields")
        properties = schema.get("properties", {})
        for key, member in value.items():
            if not isinstance(key, str):
                fail("object keys must be strings")
            if "propertyNames" in schema:
                _check(key, schema["propertyNames"], source, location + ".<key>")
            if key in properties:
                _check(member, properties[key], source, location + "." + key)
            elif schema.get("additionalProperties") is False:
                fail(f"unexpected field {key}")
            elif isinstance(schema.get("additionalProperties"), dict):
                _check(member, schema["additionalProperties"], source, location + "." + key)
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            fail("too few items")
        if schema.get("uniqueItems") and len({canonical_json(item) for item in value}) != len(value):
            fail("duplicate items")
        if "items" in schema:
            for index, item in enumerate(value):
                _check(item, schema["items"], source, f"{location}[{index}]")
    if number:
        if value < schema.get("minimum", -math.inf) or value > schema.get("maximum", math.inf):
            fail("number outside bounds")
    if isinstance(value, str):
        if not schema.get("minLength", 0) <= len(value) <= schema.get("maxLength", math.inf):
            fail("string length outside bounds")
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            fail("pattern mismatch")
        if "format" in schema:
            if schema["format"] != "date-time":
                fail("unsupported format")
            try:
                stamp = datetime.fromisoformat(value)
                if stamp.tzinfo is None or "T" not in value:
                    fail("require date-time with timezone")
            except ValueError:
                fail("invalid date-time")
