def on_startup(*args, **kwargs):
    from .nuke_startup import on_startup as _on_startup

    return _on_startup(*args, **kwargs)
