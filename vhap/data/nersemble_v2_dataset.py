# 
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual 
# property and proprietary rights in and to this software and related documentation. 
# Any commercial use, reproduction, disclosure or distribution of this software and 
# related documentation without an express license agreement from Toyota Motor Europe NV/SA 
# is strictly prohibited.
#


import json
import numpy as np
import colour
from vhap.data.nersemble_dataset import NeRSembleDataset
from vhap.util.log import get_logger
from vhap.util.color_correction import color_correction_Cheung2004_precomputed


logger = get_logger(__name__)


class NeRSembleV2Dataset(NeRSembleDataset):
    """ 
    Folder layout for NeRSembleV2 dataset:

        <root_folder>/
        |---<subject>/
            |---calibration/
            |       |---camera_params.json
            |       |---color_calibration.json
            |
            |---sequences/
                |---<sequence>/
                    |---images/
                    |   |---cam_<camera_id>_<timestep_id>.jpg
                    |
                    |---alpha_maps/
                    |   |---cam_<camera_id>_<timestep_id>.png
                    |
                    |---landmark2d/
                            |---face-alignment/
                            |    |---<camera_id>.npz
                            |
                            |---STAR/
                                |---<camera_id>.npz
                
    """
    
    def match_sequences(self):
        logger.info(f"Subject: {self.cfg.subject}, sequence: {self.cfg.sequence}")
        return list(filter(lambda x: x.is_dir(), (self.cfg.root_folder / self.cfg.subject / 'sequences').glob(f"{self.cfg.sequence}*")))
    
    def load_camera_params(self):
        super().load_camera_params(self.cfg.root_folder / self.cfg.subject / "calibration" / "camera_params.json")
    
    def load_color_correction(self):
        if self.cfg.use_color_correction:
            color_correction_path = self.cfg.root_folder / self.cfg.subject / 'calibration' / f'color_calibration.json'
            self.color_correction = {serial: np.array(ccm) for serial, ccm in json.load(open(color_correction_path)).items()}

    def apply_color_correction(self, item):
        if self.cfg.use_color_correction:
            rgb = item["rgb"] / 255
            image_linear = colour.cctf_decoding(rgb)
            ccm = self.color_correction[item["camera_id"]]
            image_corrected = color_correction_Cheung2004_precomputed(image_linear, ccm)
            image_corrected = colour.cctf_encoding(image_corrected)
            item["rgb"] = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        return item


if __name__ == "__main__":
    import tyro
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    from vhap.data.nersemble_v2_dataset import NeRSembleV2Dataset
    from vhap.config.nersemble_v2 import NersembleV2DataConfig
    from vhap.config.base import import_module

    cfg = tyro.cli(NersembleV2DataConfig)
    cfg.use_landmark = False
    dataset = import_module(cfg._target)(
        cfg=cfg,
        img_to_tensor=False,
    )

    print(len(dataset))

    sample = dataset[0]
    print(sample.keys())
    print(sample["rgb"].shape)

    dataloader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=1)
    for item in tqdm(dataloader):
        pass
