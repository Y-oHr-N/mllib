import logging

from . import base # noqa
from . import compose # noqa
from . import ensemble # noqa
from . import kernel_model # noqa
from . import metrics # noqa
from . import model_selection # noqa
from . import preprocessing # noqa
from . import utils # noqa
from . import visualization # noqa

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname)1.1s %(asctime)s] %(message)s')

handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
