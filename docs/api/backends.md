# Tracking Backends

ragpill talks to a tracking backend through a small set of protocols, so the
execute / evaluate / upload layers are backend-agnostic. MLflow is the default
(install `ragpill[mlflow]`); other backends are selected with
[`configure_backend`](#configure_backend).

```python
from ragpill.backends import configure_backend
from ragpill.backends.phoenix_backend import PhoenixBackend

configure_backend(PhoenixBackend)  # use Arize Phoenix instead of MLflow
```

!!! note
    `ragpill[phoenix]` and `ragpill[mlflow]` are **not co-installable** in one
    environment — Phoenix pins a newer OpenTelemetry than mlflow-skinny. Pick one
    backend per environment. See `plans/phoenix-backend-findings.md`.

## Backend

::: ragpill.backends.Backend
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## SpanHandle

::: ragpill.backends.SpanHandle
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## configure_backend

::: ragpill.backends.configure_backend
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## get_backend

::: ragpill.backends.get_backend
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## reset_backend

::: ragpill.backends.reset_backend
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## MLflowBackend

::: ragpill.backends.mlflow_backend.MLflowBackend
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## PhoenixBackend

::: ragpill.backends.phoenix_backend.PhoenixBackend
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## LangfuseBackend

::: ragpill.backends.langfuse_backend.LangfuseBackend
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3
