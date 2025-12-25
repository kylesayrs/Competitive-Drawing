import inspect
import torch


@torch.no_grad()
def smoothed_one_hot(
    target: int,
    num_classes: int,
    smoothing: float = 0.1,
) -> torch.Tensor:
    assert 0.0 <= smoothing <= 1.0
    assert num_classes > 1

    out = torch.full((num_classes, ), smoothing / (num_classes - 1))
    out[target] = 1.0 - smoothing

    return out


def collect_func_args(frame=None):
    if frame is None:
        frame = inspect.currentframe()
        if frame is None:
            return {}

        caller = frame.f_back
        if caller is None:
            return {}

        frame = caller

    # Get the code object for the function running in that frame
    code = frame.f_code

    # Names of positional/keyword parameters (not including *args/**kwargs yet)
    arg_names = list(code.co_varnames[:code.co_argcount])

    # Python 3 has separate count for keyword-only args
    kwonly_count = getattr(code, "co_kwonlyargcount", 0)
    kwonly_names = list(code.co_varnames[code.co_argcount : code.co_argcount + kwonly_count])

    # *args and **kwargs names, if present
    i = code.co_argcount + kwonly_count
    varargs_name = code.co_varnames[i] if (code.co_flags & inspect.CO_VARARGS) else None
    if varargs_name:
        i += 1
    varkw_name = code.co_varnames[i] if (code.co_flags & inspect.CO_VARKEYWORDS) else None

    loc = frame.f_locals

    out = {}

    # Regular positional-or-keyword args
    for name in arg_names:
        if name in loc:
            out[name] = loc[name]

    # Keyword-only args
    for name in kwonly_names:
        if name in loc:
            out[name] = loc[name]

    # *args
    if varargs_name and varargs_name in loc:
        out[varargs_name] = loc[varargs_name]

    # **kwargs
    if varkw_name and varkw_name in loc:
        out[varkw_name] = loc[varkw_name]

    return out
