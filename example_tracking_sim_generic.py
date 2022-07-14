import argparse
import cv2
import numpy as np
import time
import math

from keras.models import load_model

N = 6
M = 8

padding = 7
interval = 7
W = 64
H = 48 * 2


traj = []
moving = False
rotation = False

K = 10
mkr_rng = 0.0


x = np.arange(0, W, 1)
y = np.arange(0, H, 1)
xx, yy = np.meshgrid(y, x)

img_blur = (np.random.random((15, 15, 3)) * 0.9) + 0.1
frame0_blur = cv2.resize(img_blur, (H, W))


crazy = False


def shear(center_x, center_y, sigma, shear_x, shear_y, xx, yy):
    g = np.exp(-(((xx - center_x) ** 2 + (yy - center_y) ** 2)) / (2.0 * sigma ** 2))
    # rng = 0.05+np.random.random()*0.95
    # g[g>g.max()*rng] = g[g>g.max()*rng].mean()

    dx = shear_x * g
    dy = shear_y * g
    if crazy == False:
        thres = 2 * interval
        mag = (dx ** 2 + dy ** 2) ** 0.5
        mask = mag > thres
        dx[mask] = dx[mask] / mag[mask] * thres
        dy[mask] = dy[mask] / mag[mask] * thres

    xx_ = xx + dx
    yy_ = yy + dy
    return xx_, yy_


def twist(center_x, center_y, sigma, theta, xx, yy):
    g = np.exp(-(((xx - center_x) ** 2 + (yy - center_y) ** 2)) / (2.0 * sigma ** 2))
    # rng = 0.05+np.random.random()*0.95
    # g[g>g.max()*rng] = g[g>g.max()*rng].mean()
    dx = xx - center_x
    dy = yy - center_y

    rotx = dx * np.cos(theta) - dy * np.sin(theta)
    roty = dx * np.sin(theta) + dy * np.cos(theta)

    xx_ = xx + (rotx - dx) * g
    yy_ = yy + (roty - dy) * g
    return xx_, yy_


xx0 = xx.copy()
yy0 = yy.copy()

wx = xx0.copy()
wy = yy0.copy()

changing_x, changing_y = 0, 0


def generate(xx, yy):
    # img = np.ones((W, H, 3))
    img = frame0_blur.copy()

    for i in range(N):
        for j in range(M):
            r = int(yy[i, j])
            c = int(xx[i, j])
            if r >= W or r < 0 or c >= H or c < 0:
                continue
            shape = img[r - 1 : r + 2, c - 1 : c + 2, :].shape

            img[r - 1 : r + 2, c - 1 : c + 2, :] = (
                frame0_blur[r - 1 : r + 2, c - 1 : c + 2, :] * mkr_rng
            )
            img[r, c, :] = frame0_blur[r, c, :] * -2

    img = cv2.GaussianBlur(img, (3, 3), 0)
    print("IMG SHAPE", img.shape)
    # img = cv2.resize(img, (H, W))
    img[img < 0] = 0.0
    img[img > 1] = 1.0
    img = img[:W, :H]
    print("IMG SHAPE", img.shape)
    return img


def draw_dense_flow(img, flow, scale=1.0, K=5):
    img_ = cv2.resize(img, (img.shape[1] * K, img.shape[0] * K))
    row, col = img.shape[0], img.shape[1]
    print("FLOW SHAPE", flow.shape, "IMG SHAPE", img.shape)
    for i in range(2, row - 2, 4):
        for j in range(2, col - 2, 4):
            d = (flow[i, j] * scale * K).astype(int)
            cv2.arrowedLine(
                img_,
                (int(j * K + K // 2), int(i * K + K // 2)),
                (int(j * K + d[0] + K // 2), int(i * K + d[1] + K // 2)),
                (0, 255, 255),
                2,
                8,
                0,
                0.4,
            )
    return img_


def contrain(xx, yy):
    dx = xx - xx0
    dy = yy - yy0
    if crazy == False:
        thres = 2 * interval
        mag = (dx ** 2 + dy ** 2) ** 0.5
        mask = mag > thres
        dx[mask] = dx[mask] / mag[mask] * thres
        dy[mask] = dy[mask] / mag[mask] * thres
    xx = xx0 + dx
    yy = yy0 + dy
    return xx, yy


def cross_product(Ax, Ay, Bx, By, Cx, Cy):
    len1 = ((Bx - Ax) ** 2 + (By - Ay) ** 2) ** 0.5
    len2 = ((Cx - Ax) ** 2 + (Cy - Ay) ** 2) ** 0.5
    return ((Bx - Ax) * (Cy - Ay) - (By - Ay) * (Cx - Ax)) / (len1 * len2 + 1e-6)


def motion_callback(event, x, y, flags, param):
    global traj, moving, xx, yy, wx, wy, rotation, changing_x, changing_y

    x, y = x / K, y / K

    if event == cv2.EVENT_LBUTTONDOWN:
        traj.append([x, y])

        wx = xx0.copy()
        wy = yy0.copy()

        rotation = False
        moving = True

    elif event == cv2.EVENT_LBUTTONUP:
        traj = []
        moving = False

    elif event == cv2.EVENT_MOUSEMOVE:
        if moving == True:
            traj.append([x, y])
            sigma = 10
            # sigma = 20
            if rotation == False:
                xx, yy = shear(
                    traj[0][0],
                    traj[0][1],
                    sigma,
                    x - traj[0][0],
                    y - traj[0][1],
                    wx,
                    wy,
                )
            else:
                sigma = 20
                theta = math.asin(
                    cross_product(traj[0][0], traj[0][1], changing_x, changing_y, x, y)
                )
                theta = max(min(theta, 50 / 180.0 * math.pi), -50 / 180.0 * math.pi)
                xx, yy = twist(traj[0][0], traj[0][1], sigma, theta, wx, wy)
            if crazy == False:
                xx, yy = contrain(xx, yy)


cv2.namedWindow("image")
cv2.setMouseCallback("image", motion_callback)

flag_first = False

model = load_model("models/weights_generic.h5")

svid = 0

xind = (np.random.random(N * M) * W).astype(np.int)
yind = (np.random.random(N * M) * H).astype(np.int)


# T = 4
interval_x = W / (N + 1)
interval_y = H / (M + 1)

x = np.arange(interval_x, W, interval_x)[:N]
y = np.arange(interval_y, H, interval_y)[:M]
xind, yind = np.meshgrid(x, y)
xind = (xind.reshape([1, -1])[0]).astype(np.int)
yind = (yind.reshape([1, -1])[0]).astype(np.int)


xx_marker, yy_marker = xx[xind, yind].reshape([N, M]), yy[xind, yind].reshape([N, M])
img0 = generate(xx_marker, yy_marker)


while True:
    xx_marker_, yy_marker_ = xx[xind, yind].reshape([N, M]), yy[xind, yind].reshape(
        [N, M]
    )

    img = generate(xx_marker_, yy_marker_)
    # img += np.random.random(img.shape) * 0.1
    st = time.time()
    pred = model.predict(
        np.array(
            [
                np.dstack(
                    [
                        img0 - 0.5,
                        img - 0.5,
                        np.reshape(xx, [W, H, 1]),
                        np.reshape(yy, [W, H, 1]),
                    ]
                )
            ]
        )
    )[0][0]
    print(time.time() - st)

    # pred += 0.5

    if flag_first == False:
        flag_first = True
        img0 = img.copy()

        xx = xx0.copy()
        yy = yy0.copy()

        pred0 = pred.copy()
        continue

    pred = pred - pred0 + np.dstack([xx0, yy0])

    # absolute
    x = np.arange(0, W, 1)
    y = np.arange(0, H, 1)
    xx0, yy0 = np.meshgrid(y, x)
    flow = pred[:, :] - np.dstack([xx0, yy0])
    display_img = draw_dense_flow(img, flow, scale=1, K=10)

    cv2.imshow("image", display_img)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("r"):
        xx = xx0.copy()
        yy = yy0.copy()

    if key == ord("s"):
        rotation = rotation ^ True
        if len(traj) > 0:
            changing_x = traj[-1][0]
            changing_y = traj[-1][1]
        if rotation == False:
            traj = []
        wx, wy = xx.copy(), yy.copy()

    elif key == ord("q"):
        break
    elif key == ord("c"):
        img_blur = (np.random.random((3, 3, 3)) * 0.7) + 0.3
        frame0_blur = cv2.resize(img_blur, (H, W))

        xx_marker, yy_marker = xx0[xind, yind].reshape([N, M]), yy0[xind, yind].reshape(
            [N, M]
        )
        img0 = generate(xx_marker, yy_marker)

    elif key == ord("p"):
        cv2.imwrite("im{}.jpg".format(svid), display_img * 255)
        svid += 1

    elif key == ord("k"):
        crazy = crazy ^ True

    elif key == ord("d"):
        N = np.random.randint(6, 10)
        M = np.random.randint(6, 14)

        interval_x = W / (N + 1)
        interval_y = H / (M + 1)

        x = np.arange(interval_x, W, interval_x)[:N]
        y = np.arange(interval_y, H, interval_y)[:M]

        xind, yind = np.meshgrid(x, y)
        xind = (xind.reshape([1, -1])[0]).astype(np.int)
        yind = (yind.reshape([1, -1])[0]).astype(np.int)

        xx_marker, yy_marker = xx0[xind, yind].reshape([N, M]), yy0[xind, yind].reshape(
            [N, M]
        )
        img0 = generate(xx_marker, yy_marker)

    elif key == ord("z"):
        mkr_rng = mkr_rng - 0.5
        if mkr_rng < 0:
            mkr_rng = 1

        xx_marker, yy_marker = xx0[xind, yind].reshape([N, M]), yy0[xind, yind].reshape(
            [N, M]
        )
        img0 = generate(xx_marker, yy_marker)
        # flag_first = False

# close all open windows
cv2.destroyAllWindows()
