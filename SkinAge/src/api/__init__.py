"""SkinAge API package."""


def create_app(*args, **kwargs):
    """Lazy import to avoid pulling in fastapi/yaml for dashboard-only use."""
    from .app import create_app as _create_app
    return _create_app(*args, **kwargs)


__all__ = ["create_app"]
