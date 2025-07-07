# 
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual 
# property and proprietary rights in and to this software and related documentation. 
# Any commercial use, reproduction, disclosure or distribution of this software and 
# related documentation without an express license agreement from Toyota Motor Europe NV/SA 
# is strictly prohibited.
#


import os
import sys
from collections import defaultdict
from typing import Literal
from tqdm import tqdm
import face_alignment
import numpy as np
from torch.utils.data import DataLoader
import dlib
from joblib import Parallel, delayed

from vhap.util.log import get_logger, tqdm_joblib
logger = get_logger(__name__)


class LandmarkDetectorFA:

    IMAGE_FILE_NAME = "image_0000.png"
    LMK_FILE_NAME = "keypoints_static_0000.json"

    def __init__(
        self,
        face_detector:Literal["sfd", "blazeface"]="sfd",
    ):
        """
        Creates dataset_path where all results are stored
        :param video_path: path to video file
        :param dataset_path: path to results directory
        """

        logger.info("Initialize FaceAlignment module...")
        # 68 facial landmark detector

        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_HALF_D, 
            face_detector=face_detector,
            flip_input=True, 
            device="cuda"
        )

    def detect_single_image(self, img):
        bbox = self.fa.face_detector.detect_from_image(img)

        if len(bbox) == 0:
            lmks = np.zeros([68, 3]) - 1  # set to -1 when landmarks is inavailable

        else:
            if len(bbox) > 1:
                # if multiple boxes detected, use the one with highest confidence
                bbox = [bbox[np.argmax(np.array(bbox)[:, -1])]]

            lmks = self.fa.get_landmarks_from_image(img, detected_faces=bbox)[0]
            lmks = np.concatenate([lmks, np.ones_like(lmks[:, :1])], axis=1)

            if (lmks[:, :2] == -1).sum() > 0:
                lmks[:, 2:] = 0.0
            else:
                lmks[:, 2:] = 1.0

            h, w = img.shape[:2]
            lmks[:, 0] /= w
            lmks[:, 1] /= h
            bbox[0][[0, 2]] /= w
            bbox[0][[1, 3]] /= h
        return bbox, lmks

def detect_dataset(dataset):
    """
    Annotates each frame with 68 facial landmarks
    :return: dict mapping frame number to landmarks numpy array and the same thing for bboxes
    """
    detector = LandmarkDetectorFA()

    landmarks = {}
    bboxes = {}
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    for item in tqdm(dataloader):
        timestep_id = item["timestep_id"][0]
        camera_id = item["camera_id"][0]
        img = item["rgb"][0].numpy()
        logger.info(
            f"Annotate facial landmarks for timestep: {timestep_id}, camera: {camera_id}"
        )
        
        bbox, lmks = detector.detect_single_image(img)
        if len(bbox) == 0:
            logger.error(
                f"No bbox found for frame: {timestep_id}, camera: {camera_id}. Setting landmarks to all -1."
            )

        if camera_id not in landmarks:
            landmarks[camera_id] = {}
        if camera_id not in bboxes:
            bboxes[camera_id] = {}
        landmarks[camera_id][timestep_id] = lmks
        bboxes[camera_id][timestep_id] = bbox[0] if len(bbox) > 0 else np.zeros(5) - 1
    return landmarks, bboxes


def detect_dataset_chunk(dataset, chunk_idx, num_chunks):
    detector = LandmarkDetectorFA()

    landmarks = {}
    bboxes = {}
    item_inds = list(range(chunk_idx, len(dataset), num_chunks))
    for item_idx in tqdm(item_inds):
        item = dataset[item_idx]
        if item is None:
            bbox = np.zeros(5) - 1
            lmks = np.zeros([68, 3]) - 1
            continue

        timestep_id = item["timestep_id"]
        camera_id = item["camera_id"]
        img = item["rgb"]
        logger.info(
            f"Annotate facial landmarks for timestep: {timestep_id}, camera: {camera_id}"
        )
        sys.stdout.flush()
        
        bbox, lmks = detector.detect_single_image(img)
        if len(bbox) == 0:
            logger.error(
                f"No bbox found for frame: {timestep_id}, camera: {camera_id}. Setting landmarks to all -1."
            )
            sys.stdout.flush()

        if camera_id not in landmarks:
            landmarks[camera_id] = {}
        if camera_id not in bboxes:
            bboxes[camera_id] = {}
        landmarks[camera_id][timestep_id] = lmks
        bboxes[camera_id][timestep_id] = bbox[0] if len(bbox) > 0 else np.zeros(5) - 1
    return landmarks, bboxes


def annotate_landmarks(dataset, n_jobs=1):
    """
    Annotates each frame with landmarks for face. Assumes frames have been extracted
    :return:
    """
    os.umask(0o002)

    if n_jobs > 1:
        with tqdm_joblib(tqdm(desc="Progress", total=len(dataset))) as progress_bar:
            out = Parallel(n_jobs=n_jobs)(
                delayed(detect_dataset_chunk)(dataset, chunk_idx, n_jobs) for chunk_idx in range(n_jobs)
            )
        
        landmarks = defaultdict(dict)
        bboxes = defaultdict(dict)
        for landmarks_chunk, bboxes_chunk in out:
            for camera_id in landmarks_chunk.keys():
                landmarks[camera_id].update(landmarks_chunk[camera_id])
                bboxes[camera_id].update(bboxes_chunk[camera_id])
    else:
        landmarks, bboxes = detect_dataset(dataset)

    # construct final npz
    for camera_id, lmk_face_camera in landmarks.items():
        bounding_box = []
        face_landmark_2d = []
        for timestep_id in sorted(lmk_face_camera.keys()):
            bounding_box.append(bboxes[camera_id][timestep_id][None])
            face_landmark_2d.append(landmarks[camera_id][timestep_id][None])

        lmk_dict = {
            "bounding_box": bounding_box,
            "face_landmark_2d": face_landmark_2d,
        }

        for k, v in lmk_dict.items():
            if len(v) > 0:
                lmk_dict[k] = np.concatenate(v, axis=0)
    out_path = dataset.get_property_path(
        "landmark2d/face-alignment", camera_id=camera_id
    )
    logger.info(f"Saving landmarks to: {out_path}")
    if not out_path.parent.exists():
        out_path.parent.mkdir(parents=True)
    np.savez(out_path, **lmk_dict)


if __name__ == "__main__":
    import tyro
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    from vhap.config.base import DataConfig, import_module

    cfg = tyro.cli(DataConfig)
    dataset = import_module(cfg._target)(
        cfg=cfg,
        img_to_tensor=False,
    )
    dataset.items = dataset.items[:2]

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    detector = LandmarkDetectorFA()
    detector.annotate_landmarks(dataloader)
