import logging

from . import bsgd # noqa
from . import metrics # noqa
from . import model_selection # noqa
from . import preprocessing # noqa
from . import utils # noqa

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname)1.1s %(asctime)s] %(message)s')

handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
