from pathlib import Path
from tqdm import tqdm
from typing import Literal, Optional, List
import tyro
import ffmpeg
from PIL import Image
import torch
from vhap.data.image_folder_dataset import ImageFolderDataset
from torch.utils.data import DataLoader
from BackgroundMattingV2.model import MattingRefine
from BackgroundMattingV2.asset import get_weights_path


def video2frames(video_path: Path, image_dir: Path, keep_video_name: bool=False, target_fps: int=30, n_downsample: int=1):
    print(f'Converting video {video_path} to frames with downsample scale {n_downsample}')
    if not image_dir.exists():
        image_dir.mkdir(parents=True)
    
    file_path_stem = video_path.stem + '_' if keep_video_name else ''

    probe = ffmpeg.probe(str(video_path))
    
    video_fps = int(probe['streams'][0]['r_frame_rate'].split('/')[0])
    if  video_fps ==0:
        video_fps = int(probe['streams'][0]['avg_frame_rate'].split('/')[0])
        if video_fps == 0:
            # nb_frames / duration
            video_fps = int(probe['streams'][0]['nb_frames']) / float(probe['streams'][0]['duration'])
            if video_fps == 0:
                raise ValueError('Cannot get valid video fps')

    num_frames = int(probe['streams'][0]['nb_frames'])
    video = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    W = int(video['width'])
    H = int(video['height'])
    w = W // n_downsample
    h = H // n_downsample
    print(f'[Video]  FPS: {video_fps} | number of frames: {num_frames} | resolution: {W}x{H}')
    print(f'[Target] FPS: {target_fps} | number of frames: {round(num_frames * target_fps / int(video_fps))} | resolution: {w}x{h}')

    (ffmpeg
    .input(str(video_path))
    .filter('fps', fps=f'{target_fps}')
    .filter('scale', width=w, height=h)
    .output(
        str(image_dir / f'{file_path_stem}%06d.jpg'),
        start_number=0,
        qscale=1,  # lower values mean higher quality (1 is the best, 31 is the worst).
    )
    .overwrite_output()
    .run(quiet=True)
    )

def robust_video_matting(image_dir: Path, N_warmup: Optional[int]=10):
    print(f'Running robust video matting on images in {image_dir}')
    # model = torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3").cuda()
    model = torch.hub.load("PeterL1n/RobustVideoMatting", "resnet50").cuda()

    dataset = ImageFolderDataset(image_folder=image_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    # bgr = torch.tensor([.47, 1, .6]).view(3, 1, 1).cuda()  # Green background.
    rec = [None] * 4  # Initial recurrent states.
    downsample_ratio = 0.5  #(for videos in 512x512)
    # downsample_ratio = 0.125  #(for videos in 3k)
    for item in tqdm(dataloader):
        rgb = item['rgb']
        rgb = rgb.permute(0, 3, 1, 2).float().cuda() / 255
        with torch.no_grad():
            while N_warmup:
                # use the first frame to warm up the recurrent states as if the video 
                # has N_warmup identical frames at the beginning. This trick effectively
                # removes the artifacts for the first frame.
                fgr, pha, *rec = model(rgb, *rec, downsample_ratio)  # Cycle the recurrent states.
                N_warmup -= 1

            fgr, pha, *rec = model(rgb, *rec, downsample_ratio)  # Cycle the recurrent states.
            # fgr, pha, *rec = model(rgb, *rec)  # Cycle the recurrent states.
            # fg = fgr * pha + bgr * (1 - pha)

        alpha = (pha[0, 0] * 255).cpu().numpy()
        alpha = Image.fromarray(alpha.astype('uint8'))
        alpha_path = item['image_path'][0].replace('images', 'alpha_maps')
        if not Path(alpha_path).parent.exists():
            Path(alpha_path).parent.mkdir(parents=True)
        alpha.save(alpha_path)

def background_matting_v2(
        image_dir: Path,
        background_folder: Path=Path('../../BACKGROUND'),
        model_backbone: Literal['resnet101', 'resnet50', 'mobilenetv2']='resnet101',
        model_backbone_scale: float=0.25,
        model_refine_mode: Literal['full', 'sampling', 'thresholding']='thresholding',
        model_refine_sample_pixels: int=80_000,
        model_refine_threshold: float=0.01,
        model_refine_kernel_size: int=3,
    ):
    model = MattingRefine(
        model_backbone,
        model_backbone_scale,
        model_refine_mode,
        model_refine_sample_pixels,
        model_refine_threshold,
        model_refine_kernel_size
    )

    weights_path = get_weights_path(model_backbone)

    model = model.cuda().eval()
    model.load_state_dict(torch.load(weights_path, map_location='cuda'), strict=False)

    dataset = ImageFolderDataset(
        image_folder=image_dir, 
        background_folder=background_folder,
        background_fname2camId=lambda x: x.split('.')[0].split('_')[1],  # image_00001.jpg -> 00001
        image_fname2camId=lambda x: x.split('.')[0].split('_')[1],       # cam_00001.jpg -> 00001
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    for item in tqdm(dataloader):
        src = item['rgb']
        bgr = item['background']
        src = src.permute(0, 3, 1, 2).float().cuda() / 255
        bgr = bgr.permute(0, 3, 1, 2).float().cuda() / 255

        with torch.no_grad():
            pha, fgr, _, _, err, ref = model(src, bgr)

        alpha = (pha[0, 0] * 255).cpu().numpy()
        alpha = Image.fromarray(alpha.astype('uint8'))
        alpha_path = item['image_path'][0].replace('images', 'alpha_maps')
        if not Path(alpha_path).parent.exists():
            Path(alpha_path).parent.mkdir(parents=True)
        alpha.save(alpha_path)

def downsample_frames(image_dir: Path, n_downsample: int):
    print(f'Downsample frames in {image_dir} by {n_downsample}')
    assert n_downsample in [2, 4, 8]

    image_paths = sorted(list(image_dir.glob('*.jpg')))
    for i, image_path in tqdm(enumerate(image_paths), total=len(image_paths)):
        # downasample the resolution of images
        img = Image.open(image_path)
        W, H = img.size
        img = img.resize((W // n_downsample, H // n_downsample))
        img.save(image_path)
    
def main(
        input: Path, 
        target_fps: int=25, 
        downsample_scales: List[int]=[],
        matting_method: Optional[Literal['robust_video_matting', 'background_matting_v2']]=None,
        background_folder: Path=Path('../../BACKGROUND'),
    ):
    # prepare path
    if input.suffix in ['.mov', '.mp4']:
        print(f'Processing video file: {input}')
        videos = [input]
        image_dir = input.parent / input.stem / 'images'
    else:
        if not input.exists():
            matched_folders = list(input.parent.glob(f"{input.name}*"))
            if len(matched_folders) == 0:
                raise FileNotFoundError(f"Cannot find the directory: {input}")
            elif len(matched_folders) == 1:
                input = matched_folders[0]
            else:
                raise FileNotFoundError(f"Found multiple matched folders: {matched_folders}")
        print(f'Processing directory: {input}')
        videos = list(input.glob('cam_*.mp4'))
        image_dir = input / 'images'

    # extract frames
    for i, video_path in enumerate(videos):
        print(f'[{i}/{len(videos)}] Processing video file: {video_path}')

        for n_downsample in [1] + downsample_scales:
            image_dir_ = image_dir if n_downsample == 1 else Path(str(image_dir) + f'_{n_downsample}')
            video2frames(video_path, image_dir_, keep_video_name=len(videos) > 1, target_fps=target_fps, n_downsample=n_downsample)
        
    # foreground matting
    if matting_method == 'robust_video_matting':
        robust_video_matting(image_dir)
    elif matting_method == 'background_matting_v2':
        background_matting_v2(image_dir, background_folder=background_folder)


if __name__ == '__main__':
    tyro.cli(main)