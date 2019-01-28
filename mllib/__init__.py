import logging

from . import bsgd # noqa
from . import metrics # noqa
from . import model_selection # noqa
from . import preprocessing # noqa
from . import utils # noqa

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()

logger.addHandler(handler)
logger.setLevel(logging.INFO)
