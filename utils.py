from common.transformations.camera import normalize
from common.transformations.model import medmodel_intrinsics
import common.transformations.orientation as orient
import numpy as np
import math
import os
import cv2
#from tools.lib.logreader import LogReader


FULL_FRAME_SIZE = (1164, 874)
W, H = FULL_FRAME_SIZE[0], FULL_FRAME_SIZE[1]
eon_focal_length = FOCAL = 910.0

# aka 'K' aka camera_frame_from_view_frame
eon_intrinsics = np.array([
    [FOCAL,   0.,   W/2.],
    [0.,  FOCAL,  H/2.],
    [0.,    0.,     1.]])


def printf(*args, **kwargs):
    print(flush=True, *args, **kwargs)


def transform_img(base_img,
                  augment_trans=np.array([0, 0, 0]),
                  augment_eulers=np.array([0, 0, 0]),
                  from_intr=eon_intrinsics,
                  to_intr=eon_intrinsics,
                  output_size=None,
                  pretransform=None,
                  top_hacks=False,
                  yuv=False,
                  alpha=1.0,
                  beta=0,
                  blur=0):
    # import cv2  # pylint: disable=import-error
    cv2.setNumThreads(1)

    if yuv:
        base_img = cv2.cvtColor(base_img, cv2.COLOR_YUV2RGB_I420)

    size = base_img.shape[:2]
    if not output_size:
        output_size = size[::-1]

    cy = from_intr[1, 2]

    def get_M(h=1.22):
        quadrangle = np.array([[0, cy + 20],
                            [size[1]-1, cy + 20],
                            [0, size[0]-1],
                            [size[1]-1, size[0]-1]], dtype=np.float32)
        quadrangle_norm = np.hstack((normalize(quadrangle, intrinsics=from_intr), np.ones((4, 1))))
        quadrangle_world = np.column_stack((h*quadrangle_norm[:, 0]/quadrangle_norm[:, 1],
                                            h*np.ones(4),
                                            h/quadrangle_norm[:, 1]))
        rot = orient.rot_from_euler(augment_eulers)
        to_extrinsics = np.hstack((rot.T, -augment_trans[:, None]))
        to_KE = to_intr.dot(to_extrinsics)
        warped_quadrangle_full = np.einsum('jk,ik->ij', to_KE, np.hstack((quadrangle_world, np.ones((4, 1)))))
        warped_quadrangle = np.column_stack((warped_quadrangle_full[:, 0]/warped_quadrangle_full[:, 2],
                                            warped_quadrangle_full[:, 1]/warped_quadrangle_full[:, 2])).astype(np.float32)
        M = cv2.getPerspectiveTransform(quadrangle, warped_quadrangle.astype(np.float32))
        return M

    M = get_M()
    if pretransform is not None:
        M = M.dot(pretransform)
    augmented_rgb = cv2.warpPerspective(base_img, M, output_size, borderMode=cv2.BORDER_REPLICATE)

    if top_hacks:
        cyy = int(math.ceil(to_intr[1, 2]))
        M = get_M(1000)
        if pretransform is not None:
            M = M.dot(pretransform)
        augmented_rgb[:cyy] = cv2.warpPerspective(base_img, M, (output_size[0], cyy), borderMode=cv2.BORDER_REPLICATE)

    # brightness and contrast augment
    # augmented_rgb = np.clip((float(alpha)*augmented_rgb + beta), 0, 255).astype(np.uint8)

    # print('after clip:', augmented_rgb.shape, augmented_rgb.dtype)
    # gaussian blur
    if blur > 0:
        augmented_rgb = cv2.GaussianBlur(augmented_rgb, (blur*2+1, blur*2+1), cv2.BORDER_DEFAULT)

    if yuv:
        augmented_img = cv2.cvtColor(augmented_rgb, cv2.COLOR_RGB2YUV_I420)
    else:
        augmented_img = augmented_rgb

    return augmented_img


def reshape_yuv(frames):
    H = (frames.shape[1]*2)//3
    W = frames.shape[2]
    in_img1 = np.zeros((frames.shape[0], 6, H//2, W//2), dtype=np.uint8)

    in_img1[:, 0] = frames[:, 0:H:2, 0::2]
    in_img1[:, 1] = frames[:, 1:H:2, 0::2]
    in_img1[:, 2] = frames[:, 0:H:2, 1::2]
    in_img1[:, 3] = frames[:, 1:H:2, 1::2]
    in_img1[:, 4] = frames[:, H:H+H//4].reshape((-1, H//2, W//2))
    in_img1[:, 5] = frames[:, H+H//4:H+H//2].reshape((-1, H//2, W//2))
    return in_img1


def load_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    yuv_frames = []
    index = 0
    while cap.isOpened():
        index += 1
        ret, frame = cap.read()
        if not ret:
            break

        yuv_frames.append(bgr_to_yuv(frame))
        if index == 20:
            return yuv_frames

    return yuv_frames


def load_calibration(segment_path):
    logs_file = os.path.join(segment_path, 'rlog.bz2')
    lr = LogReader(logs_file)
    liveCalibration = [m.liveCalibration for m in lr if m.which() == 'liveCalibration']  # probably not 1200, but 240
    return liveCalibration


def bgr_to_yuv(img_bgr):
    img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV_I420)
    assert img_yuv.shape == ((874*3//2, 1164))
    return img_yuv


def bgr_to_rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def yuv_to_rgb(yuv):
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_I420)


def rgb_to_yuv(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV_I420)


def transform_frames(frames):
    imgs_med_model = np.zeros((len(frames), 384, 512), dtype=np.uint8)
    for i, img in enumerate(frames):
        imgs_med_model[i] = transform_img(img, 
                                          from_intr=eon_intrinsics,
                                          to_intr=medmodel_intrinsics, 
                                          yuv=True,
                                          output_size=(512, 256))

    reshaped = reshape_yuv(imgs_med_model)

    return reshaped


def get_train_imgs(path_to_segment, video_file='fcamera.hevc', gt_file='ground_truths.npz'):
    '''Return pre-processed (not-yet-stacked) frames from a segment video.

    return: (N, 6, 128, 256)

    `N` is determined by the number of ground truth poses for the segment.

    TODO: Should we add -1? Since the first ground truth pose is computed 
    before we have 2 frames — so we should discard the first ground truth.
    Not sure about how ground truth poses are aligned with frames tho.
    '''

    input_video = os.path.join(path_to_segment, video_file)
    ground_truths_file = os.path.join(path_to_segment, gt_file)

    #if not os.path.exists(ground_truths_file):
    #    raise FileNotFoundError('Segment ground truths NOT FOUND: {}'.format(path_to_segment))

    #ground_truths = np.load(ground_truths_file)
    #n_inputs_necessary = ground_truths['plan'].shape[0] # TODO: should we add -1 here?

    yuv_frames = load_frames(input_video)
    prepared_frames = transform_frames(yuv_frames) # NOTE: should NOT be normalized

    # example of how to get stacked frames
    #
    # train_imgs = np.zeros((n_inputs_necessary, 12, 128, 256))
    # for i in range(n_inputs_necessary):
    #     stacked_frames = np.vstack(prepared_frames[i:i+2])[None] # (12, 128, 256)
    #     train_imgs[i] = stacked_frames

    return prepared_frames#[:n_inputs_necessary]
