import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

IMG_WIDTH = 1920
IMG_HEIGHT = 1080
#IMG_WIDTH = 1152
#IMG_HEIGHT = 648
SINGLE_IMG_SIZE = int(IMG_WIDTH * IMG_HEIGHT)
IMG_SIZE = int(IMG_WIDTH * IMG_HEIGHT * 3 / 2)

Y_WIDTH = IMG_WIDTH
Y_HEIGHT = IMG_HEIGHT
Y_SIZE = int(Y_WIDTH * Y_HEIGHT)

U_V_WIDTH = int(IMG_WIDTH / 2)
U_V_HEIGHT = int(IMG_HEIGHT / 2)
U_V_SIZE = int(U_V_WIDTH * U_V_HEIGHT)

way = 2
option = 0

def from_I420(yuv_data, frames):
    Y = np.zeros((frames, IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
    U = np.zeros((frames, U_V_HEIGHT, U_V_WIDTH), dtype=np.uint8)
    V = np.zeros((frames, U_V_HEIGHT, U_V_WIDTH), dtype=np.uint8)

    for frame_idx in range(0, frames):
        y_start = frame_idx * IMG_SIZE
        u_start = y_start + Y_SIZE
        v_start = u_start + U_V_SIZE
        v_end = v_start + U_V_SIZE

        Y[frame_idx, :, :] = yuv_data[y_start : u_start].reshape((Y_HEIGHT, Y_WIDTH))
        U[frame_idx, :, :] = yuv_data[u_start : v_start].reshape((U_V_HEIGHT, U_V_WIDTH))
        V[frame_idx, :, :] = yuv_data[v_start : v_end].reshape((U_V_HEIGHT, U_V_WIDTH))
    return Y, U, V

def from_YV12(yuv_data, frames):
    Y = np.zeros((frames, IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
    U = np.zeros((frames, U_V_HEIGHT, U_V_WIDTH), dtype=np.uint8)
    V = np.zeros((frames, U_V_HEIGHT, U_V_WIDTH), dtype=np.uint8)

    for frame_idx in range(0, frames):
        y_start = frame_idx * IMG_SIZE
        v_start = y_start + Y_SIZE
        u_start = v_start + U_V_SIZE
        u_end = u_start + U_V_SIZE

        Y[frame_idx, :, :] = yuv_data[y_start : v_start].reshape((Y_HEIGHT, Y_WIDTH))
        V[frame_idx, :, :] = yuv_data[v_start : u_start].reshape((U_V_HEIGHT, U_V_WIDTH))
        U[frame_idx, :, :] = yuv_data[u_start : u_end].reshape((U_V_HEIGHT, U_V_WIDTH))
    return Y, U, V


def from_NV12(yuv_data, frames):
    Y = np.zeros((frames, IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
    U = np.zeros((frames, U_V_HEIGHT, U_V_WIDTH), dtype=np.uint8)
    V = np.zeros((frames, U_V_HEIGHT, U_V_WIDTH), dtype=np.uint8)

    for frame_idx in range(0, frames):
        y_start = frame_idx * IMG_SIZE
        u_v_start = y_start + Y_SIZE
        u_v_end = u_v_start + (U_V_SIZE * 2)

        Y[frame_idx, :, :] = yuv_data[y_start : u_v_start].reshape((Y_HEIGHT, Y_WIDTH))
        U_V = yuv_data[u_v_start : u_v_end].reshape((U_V_SIZE, 2))
        U[frame_idx, :, :] = U_V[:, 0].reshape((U_V_HEIGHT, U_V_WIDTH))
        V[frame_idx, :, :] = U_V[:, 1].reshape((U_V_HEIGHT, U_V_WIDTH))
    return Y, U, V


def from_NV21(yuv_data, frames):
    Y = np.zeros((frames, IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
    U = np.zeros((frames, U_V_HEIGHT, U_V_WIDTH), dtype=np.uint8)
    V = np.zeros((frames, U_V_HEIGHT, U_V_WIDTH), dtype=np.uint8)

    for frame_idx in range(0, frames):
        y_start = frame_idx * IMG_SIZE
        u_v_start = y_start + Y_SIZE
        u_v_end = u_v_start + (U_V_SIZE * 2)

        Y[frame_idx, :, :] = yuv_data[y_start : u_v_start].reshape((Y_HEIGHT, Y_WIDTH))
        U_V = yuv_data[u_v_start : u_v_end].reshape((U_V_SIZE, 2))
        V[frame_idx, :, :] = U_V[:, 0].reshape((U_V_HEIGHT, U_V_WIDTH))
        U[frame_idx, :, :] = U_V[:, 1].reshape((U_V_HEIGHT, U_V_WIDTH))
    return Y, U, V

def np_yuv2rgb(Y,U,V):
    bgr_data = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
    # V = np.repeat(V, 2, 0)
    # V = np.repeat(V, 2, 1)
    # U = np.repeat(U, 2, 0)
    # U = np.repeat(U, 2, 1)

    if way == 0:
        c = (Y-np.array([16])) * 298
        d = U - np.array([128])
        e = V - np.array([128])

        r = (c + 409 * e + 128) // 256
        g = (c - 100 * d - 208 * e + 128) // 256
        b = (c + 516 * d + 128) // 256
    elif way == 1:
        r = (Y + 1.14 * V)//1
        g = (Y - 0.39 * U - 0.58 * V)//1
        b = (Y + 2.03 * U)//1
    elif way == 2:
        r = (Y - 0.000007154783816076815 * U + 1.4019975662231445 * V - np.array([179.45477266423404]))//1
        g = (Y - 0.3441331386566162 * U - 0.7141380310058594 * V + np.array([135.45870971679688]))//1
        b = (Y + 1.7720025777816772 * U + 0.00001542569043522235 * V - np.array([226.8183044444304]))//1
    else:
        r = (Y)//1
        g = (Y)//1
        b = (Y)//1

    r = np.where(r < 0, 0, r)
    r = np.where(r > 255,255,r)

    g = np.where(g < 0, 0, g)
    g = np.where(g > 255,255,g)

    b = np.where(b < 0, 0, b)
    b = np.where(b > 255,255,b)

    bgr_data[:, :, 2] = r
    bgr_data[:, :, 1] = g
    bgr_data[:, :, 0] = b

    return bgr_data

def yuv2rgb(Y, U, V):
    bgr_data = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
    for h_idx in range(Y_HEIGHT):
        for w_idx in range(Y_WIDTH):
            y = Y[h_idx, w_idx]
            u = U[int(h_idx // 2), int(w_idx // 2)]
            v = V[int(h_idx // 2), int(w_idx // 2)]

            c = (y - 16) * 298
            d = u - 128
            e = v - 128

            r = (c + 409 * e + 128) // 256
            g = (c - 100 * d - 208 * e + 128) // 256
            b = (c + 516 * d + 128) // 256

            bgr_data[h_idx, w_idx, 2] = 0 if r < 0 else (255 if r > 255 else r)
            bgr_data[h_idx, w_idx, 1] = 0 if g < 0 else (255 if g > 255 else g)
            bgr_data[h_idx, w_idx, 0] = 0 if b < 0 else (255 if b > 255 else b)

    return bgr_data

def from_new_format(yuv_data, frames):
    Y = np.zeros((frames, IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
    U = np.zeros((frames, U_V_HEIGHT, U_V_WIDTH), dtype=np.uint8)
    V = np.zeros((frames, U_V_HEIGHT, U_V_WIDTH), dtype=np.uint8)

    for frame_idx in range(0, frames):
        y_start = frame_idx * IMG_SIZE
        u_start = y_start + Y_SIZE
        v_start = u_start + U_V_SIZE
        v_end = v_start + U_V_SIZE

        Y[frame_idx, :, :] = yuv_data[y_start : u_start].reshape((Y_HEIGHT, Y_WIDTH))
        U[frame_idx, :, :] = yuv_data[u_start : v_start].reshape((U_V_HEIGHT, U_V_WIDTH))
        V[frame_idx, :, :] = yuv_data[v_start : v_end].reshape((U_V_HEIGHT, U_V_WIDTH))
    return Y, U, V

if __name__ == '__main__':
    import time
    # SINGLE_IMG_SIZE
    yuv = "./test.yuv"
    # frames = int(os.path.getsize(yuv) / IMG_SIZE)
    frames = 1
    # print("file size : " + str(os.path.getsize(yuv)))
    # print("image size : " + str(IMG_SIZE))

    with open(yuv, "rb") as yuv_f:
        time1 = time.time()
        yuv_bytes = yuv_f.read()
        yuv_data = np.frombuffer(yuv_bytes, np.uint8)

        Y = np.zeros((frames, IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
        U = np.zeros((frames, IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
        V = np.zeros((frames, IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
        # U = np.zeros((frames, U_V_HEIGHT, U_V_WIDTH), dtype=np.uint8)
        # V = np.zeros((frames, U_V_HEIGHT, U_V_WIDTH), dtype=np.uint8)

        for frame_idx in range(0, frames):
            y_start = 0
            u_start = 3 * Y_SIZE

            yuv_data_head = yuv_data[y_start : u_start].reshape((Y_SIZE, 3))

            if option == 0:
                Y[frame_idx, :, :] = yuv_data_head[:,0].reshape((Y_HEIGHT, Y_WIDTH))
                U[frame_idx, :, :] = yuv_data_head[:,1].reshape((Y_HEIGHT, Y_WIDTH))
                V[frame_idx, :, :] = yuv_data_head[:,2].reshape((Y_HEIGHT, Y_WIDTH))
            elif option == 1:
                Y[frame_idx, :, :] = yuv_data_head[:,0].reshape((Y_HEIGHT, Y_WIDTH))
                V[frame_idx, :, :] = yuv_data_head[:,1].reshape((Y_HEIGHT, Y_WIDTH))
                U[frame_idx, :, :] = yuv_data_head[:,2].reshape((Y_HEIGHT, Y_WIDTH))
            elif option == 2:
                U[frame_idx, :, :] = yuv_data_head[:,0].reshape((Y_HEIGHT, Y_WIDTH))
                Y[frame_idx, :, :] = yuv_data_head[:,1].reshape((Y_HEIGHT, Y_WIDTH))
                V[frame_idx, :, :] = yuv_data_head[:,2].reshape((Y_HEIGHT, Y_WIDTH))
            elif option == 3:
                U[frame_idx, :, :] = yuv_data_head[:,0].reshape((Y_HEIGHT, Y_WIDTH))
                V[frame_idx, :, :] = yuv_data_head[:,1].reshape((Y_HEIGHT, Y_WIDTH))
                Y[frame_idx, :, :] = yuv_data_head[:,2].reshape((Y_HEIGHT, Y_WIDTH))
            elif option == 4:
                V[frame_idx, :, :] = yuv_data_head[:,0].reshape((Y_HEIGHT, Y_WIDTH))
                U[frame_idx, :, :] = yuv_data_head[:,1].reshape((Y_HEIGHT, Y_WIDTH))
                Y[frame_idx, :, :] = yuv_data_head[:,2].reshape((Y_HEIGHT, Y_WIDTH))
            else:
                V[frame_idx, :, :] = yuv_data_head[:,0].reshape((Y_HEIGHT, Y_WIDTH))
                Y[frame_idx, :, :] = yuv_data_head[:,1].reshape((Y_HEIGHT, Y_WIDTH))
                U[frame_idx, :, :] = yuv_data_head[:,2].reshape((Y_HEIGHT, Y_WIDTH))
            
            if (V[frame_idx, :, :] == Y[frame_idx, :, :]).all():
                print("V is the same as Y")
            elif (V[frame_idx, :, :] == U[frame_idx, :, :]).all(): 
                print("V is the same as U")
            elif (U[frame_idx, :, :] == Y[frame_idx, :, :]).all(): 
                print("Y is the same as U")
            
            windowname = "Y Matrix"
            cv2.namedWindow(windowname, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(windowname, int(0.5 * Y_WIDTH), int(0.5 * Y_HEIGHT))
            cv2.imshow(windowname , Y[frame_idx, :, :])
            # cv2.waitKey(0)
            
            windowname = "V Matrix"
            cv2.namedWindow(windowname, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(windowname, int(0.5 * Y_WIDTH), int(0.5 * Y_HEIGHT))
            cv2.imshow(windowname, V[frame_idx, :, :])
            # cv2.waitKey(0)
            
            windowname = "U Matrix"
            cv2.namedWindow(windowname, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(windowname, int(0.5 * Y_WIDTH), int(0.5 * Y_HEIGHT))
            cv2.imshow(windowname, U[frame_idx, :, :])
            cv2.waitKey(0)

            # plt.imshow(Y[frame_idx, :, :])
            # plt.title("Y")
            # plt.show()

            # plt.imshow(U[frame_idx, :, :])
            # plt.title("U")
            # plt.show()

            # plt.imshow(V[frame_idx, :, :])
            # plt.title("V")
            # plt.show()

        rgb_data = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
        for frame_idx in range(frames):
            # bgr_data = yuv2rgb(Y[frame_idx, :, :], U[frame_idx, :, :], V[frame_idx, :, :])            # for
            bgr_data = np_yuv2rgb(Y[frame_idx, :, :], U[frame_idx, :, :], V[frame_idx, :, :])           # numpy
            # time2 = time.time()
            # print(time2 - time1)
            if bgr_data is not None:
                # cv2.imwrite("frame_{}.jpg".format(frame_idx), bgr_data)
                frame_idx +=1

                # b, g, r = cv2.split(bgr_data)
                # img = cv2.merge([r, g, b])
                img = bgr_data
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                cv2.imshow("BGR", img)
                cv2.waitKey(0)

                plt.imshow(img)
                plt.title("BGR")
                plt.show()