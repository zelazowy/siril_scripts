"""Engine-independent local plate solving. No Siril, Qt, or project dependencies."""
from .api import SolveOptions, SolveResult, solve_image, register_backend, Diagnostic, resolve_options, validate_backend
__version__ = '0.1.0'
