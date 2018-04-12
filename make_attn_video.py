import numpy as np
import skimage.color
import skimage.transform
import skvideo.io
import sys
from PIL import Image

def overlay(bg, fg):
    """
    Overlay attention over the video frame
    """
    src_rgb = fg[..., :3].astype(np.float32) / 255.
    src_alpha = fg[..., 3].astype(np.float32) / 255.
    dst_rgb = bg[..., :3].astype(np.float32) / 255.
    dst_alpha = bg[..., 3].astype(np.float32) / 255.

    out_alpha = src_alpha + dst_alpha * (1. - src_alpha)
    out_rgb = (src_rgb * src_alpha[..., None] + dst_rgb * dst_alpha[..., None] * \
              (1. - src_alpha[..., None])) / out_alpha[..., None]

    out = np.zeros_like(bg)
    out[..., :3] = out_rgb * 255
    out[..., 3] = out_alpha * 255

    return out

vid_path = sys.argv[1]
attn_mask_path = sys.argv[2]
attn_height = int(sys.argv[3])
attn_width = int(sys.argv[4])
output_path = sys.argv[5]

vreader = skvideo.io.vreader(vid_path)
# vwriter = skvideo.io.FFmpegWriter(
#     output_path, inputdict={'-pix_fmt': 'yuv420p', '-vcodec': 'libx264'})
vwriter = skvideo.io.FFmpegWriter(output_path)
metadata = skvideo.io.ffprobe(vid_path)['video']
num_frames = int(metadata['@nb_frames'])
vid_height, vid_width = int(metadata['@height']), int(metadata['@width'])
upscale_ratio = vid_height / attn_height
attn_mask = np.load(attn_mask_path).reshape(-1, attn_height, attn_width)
assert num_frames == len(attn_mask)

alpha_layer = np.ones((vid_height, vid_width, 1)) * 255
for i, img in enumerate(vreader):
    im_h, im_w, im_c = img.shape
    # Process image for attention mask.
    img = np.dstack((img, alpha_layer))

    alpha_img = skimage.transform.pyramid_expand(
        attn_mask[i], upscale=upscale_ratio, sigma=20)
    alpha_img = alpha_img * 255.0 / np.max(alpha_img)
    alpha_img = skimage.color.gray2rgb(alpha_img)
    alpha_img = np.dstack((alpha_img, 0.8 * alpha_layer)) #rgba

    # Merge images.
    masked_frame = overlay(img, alpha_img)
    np.delete(masked_frame, -1)
    vwriter.writeFrame(masked_frame)
    if i == 30:
        break
