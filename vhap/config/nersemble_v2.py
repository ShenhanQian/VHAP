# 
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual 
# property and proprietary rights in and to this software and related documentation. 
# Any commercial use, reproduction, disclosure or distribution of this software and 
# related documentation without an express license agreement from Toyota Motor Europe NV/SA 
# is strictly prohibited.
#


from dataclasses import dataclass
import tyro

from vhap.config.nersemble import NersembleDataConfig, NersembleTrackingConfig
from vhap.util.log import get_logger
logger = get_logger(__name__)


@dataclass()
class NersembleV2DataConfig(NersembleDataConfig):
    _target: str = "vhap.data.nersemble_v2_dataset.NeRSembleV2Dataset"


@dataclass()
class NersembleV2TrackingConfig(NersembleTrackingConfig):
    data: NersembleV2DataConfig


if __name__ == "__main__":
    config = tyro.cli(NersembleV2TrackingConfig)
    print(tyro.to_yaml(config))