"""Small shared input checks; no optional packages or remote schema resolution."""

import json
import math
import re


def strict_json_loads(text):
    def pairs_to_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"Nonfinite JSON number: {value}")
        return number

    return json.loads(text, object_pairs_hook=pairs_to_object, parse_constant=reject_constant,
                      parse_float=finite_float)


def require_keys(value, keys, context):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"Invalid {context} fields")


def require_text(value, context):
    if not isinstance(value, str) or not value:
        raise ValueError(f"Require nonempty {context} string")


def require_sha256(value, context):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"Invalid {context} SHA-256")
