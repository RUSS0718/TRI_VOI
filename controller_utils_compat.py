import json
import re
from typing import Any, Dict


def parse_json_safe(text: str) -> Dict[str, Any]:
    """Extract first JSON object from model output without throwing."""
    if not text:
        return {}
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return {}
