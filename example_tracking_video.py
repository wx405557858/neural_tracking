import cv2
from keras.models import load_model
import numpy as np


def load_video(filename):
    cap = cv2.VideoCapture(filename)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames


def draw_flow(img, flow, xx0, yy0, K=5, img_raw=None):
    if img_raw is None:
        img_ = cv2.resize(img, (img.shape[1] * K, img.shape[0] * K))
    else:
        img_ = cv2.resize(img_raw, (img.shape[1] * K, img.shape[0] * K))

    scale = 4
    color = (0, 255, 255)

    for i in range(flow.shape[0]):
        for j in range(flow.shape[1]):
            pt1 = (xx0[i, j] * K, yy0[i, j] * K)
            pt2 = (flow[i, j, 0] * K, flow[i, j, 1] * K)
            pt3 = (
                pt2[0] + (pt2[0] - pt1[0]) * scale,
                pt2[1] + (pt2[1] - pt1[1]) * scale,
            )
            pt1 = (int(round(pt1[0])), int(round(pt1[1])))
            pt3 = (int(round(pt3[0])), int(round(pt3[1])))
            cv2.arrowedLine(img_, pt1, pt3, color, 2, 8, 0, 0.4)

    return img_


def tracking(frames, model):
    im0 = frames[0]
    im0_blur = cv2.GaussianBlur(im0, (int(63), int(63)), 0)
    diff0 = None
    pred0 = None

    for im in frames:
        diff = (im * 1.0 - im0_blur) / 255 * 2 + 0.5
        diff_center = diff

        diff_center_small = cv2.resize(
            diff_center, (112, 80), interpolation=cv2.INTER_AREA
        )

        if diff0 is None:
            diff0 = diff.copy()
            diff_center0 = diff_center.copy()
            diff_center0_small = diff_center_small.copy()

        pred = model.predict(
            np.array([np.dstack([diff_center0_small - 0.5, diff_center_small - 0.5])])
        )[0]

        if pred0 is None:
            pred0 = pred.copy()
            xx0 = pred0[:, :, 0]
            yy0 = pred0[:, :, 1]

        flow = pred[:, :]

        diff_center = (diff_center - 0.5) * 2 + 0.5
        display_img = draw_flow(
            diff_center_small, flow, xx0, yy0, K=4, img_raw=diff_center
        )

        display_img = np.clip(display_img, 0, 1)
        display_img = (display_img * 255).astype(np.uint8)

        cv2.imshow("frame", diff_center_small)
        cv2.imshow("nn", display_img)

        c = cv2.waitKey(1)
        if c & 0xFF == ord("q"):
            break


frames = load_video("data/marker_example.mov")
model = load_model("models/weights.h5")

tracking(frames, model)
