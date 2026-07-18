import argparse
import glob
import os
import re

import cv2
import torch

from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact


def build_rrdb(num_feat: int, num_block: int, num_grow_ch: int, scale: int) -> RRDBNet:
    return RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=num_feat,
        num_block=num_block,
        num_grow_ch=num_grow_ch,
        scale=scale
    )


def normalize_state_dict(ckpt):
    """Ambil state_dict dari berbagai format checkpoint."""
    if isinstance(ckpt, dict):
        for key in (
            "params_ema",
            "params",
            "state_dict",
            "model_state_dict",
            "net_g",
            "generator",
            "model",
        ):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break

    if not isinstance(ckpt, dict):
        raise TypeError("Checkpoint bukan format dict/state_dict yang dikenali.")

    cleaned = {}
    for k, v in ckpt.items():
        if isinstance(k, str) and k.startswith("module."):
            k = k[len("module."):]
        cleaned[k] = v
    return cleaned


def infer_rrdb_config_from_checkpoint(model_path: str):
    """
    Coba infer konfigurasi RRDBNet dari checkpoint.
    Target utamanya: model custom berbasis RRDBNet seperti UltraSharp / Remacri / NKMD.
    """
    ckpt = torch.load(model_path, map_location="cpu")
    sd = normalize_state_dict(ckpt)

    # num_feat dari conv pertama
    conv_first_key = None
    for k in sd.keys():
        if k.endswith("conv_first.weight"):
            conv_first_key = k
            break
    if conv_first_key is None:
        raise ValueError("Tidak menemukan key 'conv_first.weight'. Ini kemungkinan bukan RRDBNet.")

    conv_first_w = sd[conv_first_key]
    num_feat = conv_first_w.shape[0]

    # num_block dari indeks RRDB_trunk
    trunk_indices = set()
    trunk_pattern = re.compile(r"(?:RRDB_trunk|body)\.(\d+)\.")
    for k in sd.keys():
        m = trunk_pattern.search(k)
        if m:
            trunk_indices.add(int(m.group(1)))
    if not trunk_indices:
        raise ValueError("Tidak menemukan blok RRDB trunk/body. Ini kemungkinan bukan RRDBNet.")
    num_block = max(trunk_indices) + 1

    # num_grow_ch dari conv1 pertama di dalam RRDB block
    grow_key = None
    for k in sd.keys():
        if ("RRDB_trunk" in k or "body" in k) and k.endswith("conv1.weight"):
            grow_key = k
            break
    if grow_key is None:
        raise ValueError("Tidak menemukan key conv1 di dalam RRDB block.")
    num_grow_ch = sd[grow_key].shape[0]

    # scale dari jumlah tahap upsampling
    up_indices = set()
    up_pattern = re.compile(r"(?:upconv|conv_up|upsample|upsampler)\.?_?(\d+)")
    for k in sd.keys():
        m = up_pattern.search(k)
        if m:
            up_indices.add(int(m.group(1)))

    # RRDBNet biasanya:
    # x2 -> upconv1
    # x4 -> upconv1 + upconv2
    # x8 -> upconv1 + upconv2 + upconv3
    if up_indices:
        scale = 2 ** max(up_indices)
    else:
        # fallback aman untuk RRDB custom umum
        scale = 4

    model = build_rrdb(
        num_feat=num_feat,
        num_block=num_block,
        num_grow_ch=num_grow_ch,
        scale=scale
    )

    return model, scale, {
        "num_feat": num_feat,
        "num_block": num_block,
        "num_grow_ch": num_grow_ch,
        "scale": scale,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='inputs', help='Input image or folder')
    parser.add_argument(
        '-n',
        '--model_name',
        type=str,
        default='RealESRGAN_x4plus',
        help=(
            'Model names: RealESRGAN_x4plus | RealESRNet_x4plus | RealESRGAN_x4plus_anime_6B | '
            'RealESRGAN_x2plus | realesr-animevideov3 | realesr-general-x4v3 | '
            'UltraSharp | Foolhardy Remacri | NKMD | auto'
        )
    )
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder')
    parser.add_argument(
        '-dn',
        '--denoise_strength',
        type=float,
        default=0.5,
        help=(
            'Denoise strength. 0 for weak denoise (keep noise), 1 for strong denoise ability. '
            'Only used for the realesr-general-x4v3 model'
        )
    )
    parser.add_argument('-s', '--outscale', type=float, default=4, help='The final upsampling scale of the image')
    parser.add_argument('--model_path', type=str, default=None, help='[Option] Model path. Usually, you do not need to specify it')
    parser.add_argument('--suffix', type=str, default='out', help='Suffix of the restored image')
    parser.add_argument('-t', '--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
    parser.add_argument('--face_enhance', action='store_true', help='Use GFPGAN to enhance face')
    parser.add_argument('--fp32', action='store_true', help='Use fp32 precision during inference. Default: fp16 (half precision).')
    parser.add_argument('--alpha_upsampler', type=str, default='realesrgan', help='The upsampler for the alpha channels. Options: realesrgan | bicubic')
    parser.add_argument('--ext', type=str, default='auto', help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs')
    parser.add_argument('-g', '--gpu-id', type=int, default=None, help='gpu device to use (default=None) can be 0,1,2 for multi-gpu')

    args = parser.parse_args()
    model_key = args.model_name.split('.')[0].lower()

    # Registry model resmi + custom yang sudah diketahui
    model_registry = {
        "realesrgan_x4plus": {
            "builder": lambda: build_rrdb(num_feat=64, num_block=23, num_grow_ch=32, scale=4),
            "netscale": 4,
            "urls": ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"],
            "filename": "RealESRGAN_x4plus.pth",
        },
        "realesrnet_x4plus": {
            "builder": lambda: build_rrdb(num_feat=64, num_block=23, num_grow_ch=32, scale=4),
            "netscale": 4,
            "urls": ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth"],
            "filename": "RealESRNet_x4plus.pth",
        },
        "realesrgan_x4plus_anime_6b": {
            "builder": lambda: build_rrdb(num_feat=64, num_block=6, num_grow_ch=32, scale=4),
            "netscale": 4,
            "urls": ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"],
            "filename": "RealESRGAN_x4plus_anime_6B.pth",
        },
        "realesrgan_x2plus": {
            "builder": lambda: build_rrdb(num_feat=64, num_block=23, num_grow_ch=32, scale=2),
            "netscale": 2,
            "urls": ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"],
            "filename": "RealESRGAN_x2plus.pth",
        },
        "realesr-animevideov3": {
            "builder": lambda: SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu'),
            "netscale": 4,
            "urls": ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth"],
            "filename": "realesr-animevideov3.pth",
        },
        "realesr-general-x4v3": {
            "builder": lambda: SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu'),
            "netscale": 4,
            "urls": [
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
            ],
            "filename": "realesr-general-x4v3.pth",
        },

        # Custom RRDB x4
        "ultrasharp": {
            "builder": lambda: build_rrdb(num_feat=64, num_block=23, num_grow_ch=32, scale=4),
            "netscale": 4,
            "urls": ["https://huggingface.co/lokCX/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth"],
            "filename": "4x-UltraSharp.pth",
        },
        "foolhardy-remacri": {
            "builder": lambda: build_rrdb(num_feat=64, num_block=23, num_grow_ch=32, scale=4),
            "netscale": 4,
            "urls": ["https://huggingface.co/FacehugmanIII/4x_foolhardy_Remacri/resolve/main/4x_Foolhardy_Remacri.pth"],
            "filename": "4x_Foolhardy_Remacri.pth",
        },
        "nkmd": {
            "builder": lambda: build_rrdb(num_feat=64, num_block=23, num_grow_ch=32, scale=4),
            "netscale": 4,
            "urls": ["https://huggingface.co/art0123/Models_collection/resolve/main/upscale_models/4x_NMKD-Superscale-SP_178000_G.pth"],
            "filename": "4x_NMKD-Superscale-SP_178000_G.pth",
        },
    }

    # resolve alias sederhana
    alias_map = {
        "4x-ultrasharp": "ultrasharp",
        "4x_ultrasharp": "ultrasharp",
        "remacri": "foolhardy-remacri",
        "foolhardy remacri": "foolhardy-remacri",
        "4x-foolhardy-remacri": "foolhardy-remacri",
        "4x_foolhardy_remacri": "foolhardy-remacri",
        "nmkd": "nkmd",
    }
    model_key = alias_map.get(model_key, model_key)

    # AUTO: kalau mau paksa baca langsung dari .pth tanpa hardcode architecture
    if model_key == "auto":
        if not args.model_path:
            raise ValueError('Kalau --model_name auto, --model_path wajib diisi.')
        model, netscale, inferred = infer_rrdb_config_from_checkpoint(args.model_path)
        model_path = args.model_path
        print(f'✅ Auto-detect RRDB: scale={inferred["scale"]}, blocks={inferred["num_block"]}, '
              f'feat={inferred["num_feat"]}, grow_ch={inferred["num_grow_ch"]}')

    elif model_key in model_registry:
        cfg = model_registry[model_key]
        model = cfg["builder"]()
        netscale = cfg["netscale"]

        if args.model_path is not None:
            model_path = args.model_path
        else:
            model_path = os.path.join('weights', cfg["filename"])
            if not os.path.isfile(model_path):
                ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
                for url in cfg["urls"]:
                    model_path = load_file_from_url(
                        url=url,
                        model_dir=os.path.join(ROOT_DIR, 'weights'),
                        progress=True,
                        file_name=None
                    )
    else:
        # fallback: kalau model_name unknown tapi user kasih model_path, coba auto-detect checkpoint
        if not args.model_path:
            raise ValueError(
                f'MODEL "{args.model_name}" tidak valid dan --model_path belum diisi. '
                'Pakai model resmi/custom yang dikenali, atau set --model_name auto.'
            )
        model, netscale, inferred = infer_rrdb_config_from_checkpoint(args.model_path)
        model_path = args.model_path
        print(f'✅ Auto-detect RRDB fallback: scale={inferred["scale"]}, blocks={inferred["num_block"]}, '
              f'feat={inferred["num_feat"]}, grow_ch={inferred["num_grow_ch"]}')

    # DNI only for realesr-general-x4v3
    dni_weight = None
    if model_key == 'realesr-general-x4v3' and args.denoise_strength != 1:
        wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
        model_path = [model_path, wdn_model_path]
        dni_weight = [args.denoise_strength, 1 - args.denoise_strength]

    print(f'✅ Model: {args.model_name} | Netscale: {netscale}')
    print(f'✅ Model path: {model_path}')

    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=not args.fp32,
        gpu_id=args.gpu_id
    )

    if args.face_enhance:
        from gfpgan import GFPGANer
        face_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=args.outscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler
        )

    os.makedirs(args.output, exist_ok=True)

    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*')))

    for idx, path in enumerate(paths):
        imgname, extension = os.path.splitext(os.path.basename(path))
        print('Testing', idx, imgname)

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f'Warning: gagal membaca {path}')
            continue

        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = 'RGBA'
        else:
            img_mode = None

        try:
            if args.face_enhance:
                _, _, output = face_enhancer.enhance(
                    img, has_aligned=False, only_center_face=False, paste_back=True
                )
            else:
                output, _ = upsampler.enhance(img, outscale=args.outscale)
        except RuntimeError as error:
            print('Error', error)
            print('Kalau CUDA OOM, coba kecilkan --tile.')
        else:
            if args.ext == 'auto':
                extension = extension[1:]
            else:
                extension = args.ext

            if img_mode == 'RGBA':
                extension = 'png'

            if args.suffix == '':
                save_path = os.path.join(args.output, f'{imgname}.{extension}')
            else:
                save_path = os.path.join(args.output, f'{imgname}_{args.suffix}.{extension}')

            cv2.imwrite(save_path, output)
            print(f'✅ Saved: {save_path}')


if __name__ == '__main__':
    main()