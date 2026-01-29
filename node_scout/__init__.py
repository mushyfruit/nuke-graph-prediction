"""NodeScout package.

This package is primarily intended to run inside Nuke, but we keep imports
lightweight so that tooling (tests, packaging, CLI helpers) can import
`node_scout` without requiring Nuke's Python modules.
"""

from .core.model import constants as model_cnst


def on_startup(*args, **kwargs):
    """Entry point for Nuke's startup hook.

    Import is deferred so importing this package outside of Nuke doesn't crash.
    """

    from .nuke_startup import on_startup as _on_startup

    return _on_startup(*args, **kwargs)
