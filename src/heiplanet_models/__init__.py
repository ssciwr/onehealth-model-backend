from . import computation_graph
from . import utils
from . import Jmodel

# Import the version from the package metadata or provide a fallback
# if that is not possible
try:
    from importlib.metadata import version

    __version__ = version("heiplanet_models")
except ImportError:
    __version__ = "unknown"

# Optional: Define what gets imported with "from heiplanet_models import *"
__all__ = ["Jmodel", "utils", "computation_graph", "__version__"]
