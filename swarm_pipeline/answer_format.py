"""Canonicalize valid scalar JSON answers without changing response journals."""
import math

def canonical_answer(value):
    if isinstance(value,str):return value
    if type(value) is int:return str(value)
    if type(value) is float and math.isfinite(value):return str(value)
    raise ValueError('Answer must be a string or finite JSON number')
