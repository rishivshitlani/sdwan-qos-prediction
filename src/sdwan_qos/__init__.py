"""Shared package for the SD-WAN QoS prediction project.

Phase 1 of the refactor: this package holds the single source of truth for
project-wide constants (``sdwan_qos.config``). The existing scripts in ``src/``
import from here instead of keeping their own copies.
"""
