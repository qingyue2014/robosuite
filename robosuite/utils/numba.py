"""
Numba utils.
"""
import robosuite.macros as macros

if macros.ENABLE_NUMBA:
    import numba


def jit_decorator(func):
    if macros.ENABLE_NUMBA:
        return numba.jit(nopython=True, cache=macros.CACHE_NUMBA)(func)
    return func
