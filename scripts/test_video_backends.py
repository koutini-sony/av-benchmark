import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from av_bench.data.video_dataset import (
    _IMAGEBIND_FPS,
    _SYNC_FPS,
    StreamingMediaDecoder,
    VideoDataset,
)


def _to_tchw(video: torch.Tensor) -> torch.Tensor:
    if video.ndim != 4:
        raise ValueError(f'Expected 4D video tensor, got shape={tuple(video.shape)}')
    if video.shape[1] == 3:
        return video
    if video.shape[0] == 3:
        return video.permute(1, 0, 2, 3)
    if video.shape[-1] == 3:
        return video.permute(0, 3, 1, 2)
    raise ValueError(f'Could not infer channel dimension for tensor shape={tuple(video.shape)}')


def _save_video_frames(video: torch.Tensor, out_dir: Path, prefix: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    video_tchw = _to_tchw(video).detach().cpu()
    for i in range(video_tchw.shape[0]):
        frame = video_tchw[i].to(torch.float32)
        if frame.max() > 1.0 or frame.min() < 0.0:
            frame = frame / 255.0
        frame = frame.clamp(0.0, 1.0)
        save_image(frame, out_dir / f'{prefix}_{i:04d}.png')


def main():
    parser = argparse.ArgumentParser(description='Compare torio and pyav frame extraction outputs.')
    parser.add_argument('video_path', type=Path, help='Path to a video file.')
    parser.add_argument('--backend', choices=['pyav', 'torio', 'both'], default='both')
    parser.add_argument('--duration-sec', type=float, default=8.0)
    parser.add_argument('--out-dir', type=Path, default=Path('backend_decode_test'))
    args = parser.parse_args()

    if not args.video_path.exists():
        raise FileNotFoundError(f'Video file does not exist: {args.video_path}')

    dataset = VideoDataset([args.video_path], duration_sec=args.duration_sec)
    run_out_dir = args.out_dir / args.video_path.stem
    print(f'video={args.video_path}')
    print(f'duration_sec={args.duration_sec}')

    if args.backend in ('torio', 'both'):
        if StreamingMediaDecoder is None:
            print('torio: unavailable (skipped)')
        else:
            torio_ib, torio_sync = dataset._sample_with_torio(args.video_path)
            torio_ib_dir = run_out_dir / 'torio_ib'
            torio_sync_dir = run_out_dir / 'torio_sync'
            _save_video_frames(torio_ib, torio_ib_dir, 'frame')
            _save_video_frames(torio_sync, torio_sync_dir, 'frame')
            print(f'torio_ib.shape={tuple(torio_ib.shape)}')
            print(f'torio_sync.shape={tuple(torio_sync.shape)}')
            print(f'torio_ib.saved_dir={torio_ib_dir}')
            print(f'torio_sync.saved_dir={torio_sync_dir}')

    if args.backend in ('pyav', 'both'):
        pyav_ib = dataset._sample_with_pyav(args.video_path, _IMAGEBIND_FPS, dataset.ib_expected_length)
        pyav_sync = dataset._sample_with_pyav(args.video_path, _SYNC_FPS, dataset.sync_expected_length)
        pyav_ib_dir = run_out_dir / 'pyav_ib'
        pyav_sync_dir = run_out_dir / 'pyav_sync'
        _save_video_frames(pyav_ib, pyav_ib_dir, 'frame')
        _save_video_frames(pyav_sync, pyav_sync_dir, 'frame')
        print(f'pyav_ib.shape={tuple(pyav_ib.shape)}')
        print(f'pyav_sync.shape={tuple(pyav_sync.shape)}')
        print(f'pyav_ib.saved_dir={pyav_ib_dir}')
        print(f'pyav_sync.saved_dir={pyav_sync_dir}')


if __name__ == '__main__':
    main()
