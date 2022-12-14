import sys
import numpy as np
import cv2
import datetime

def cartoon_filter(img):
    h, w = img.shape[:2]
    img2 = cv2.resize(img, (w//2, h//2))

    blr = cv2.bilateralFilter(img2, -1, 20, 7)
    edge = 255 - cv2.Canny(img2, 80, 120)
    edge = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)

    dst = cv2.bitwise_and(blr, edge)
    dst = cv2.resize(dst, (w,h), interpolation = cv2.INTER_NEAREST)
    return dst

def pencil_sketch(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blr = cv2.GaussianBlur(gray, (0, 0), 3)

    dst = cv2.divide(gray, blr, scale = 255)
    return dst

def sharp(img):
    sharp = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    sharp_f = sharp[:,:,0].astype(np.float32)
    blr = cv2.GaussianBlur(sharp_f, (0, 0), 2.0)
    sharp[:,:,0] = np.clip(2. * sharp_f - blr, 0, 255).astype(np.uint8)

    dst = cv2.cvtColor(sharp, cv2.COLOR_YCrCb2BGR)
    return dst

def saturate_bright(p, num):
    pic = p.copy()
    pic = pic.astype('int32')
    pic = np.clip(pic + num, 0, 255)
    pic = pic.astype('uint8')
    return pic


def saturate_dark(p, num):
    pic = p.copy()
    pic = pic.astype('int32')
    pic = np.clip(pic - num, 0, 255)
    pic = pic.astype('uint8')
    return pic


def bright(x):
    b = cv2.getTrackbarPos('bright', 'frame')
    img2 = saturate_bright(img, b)
    cv2.imshow('frame', img2)


def dark(x):
    b = cv2.getTrackbarPos('dark', 'frame')
    img2 = saturate_dark(img, b)
    cv2.imshow('frame', img2)

cv2.namedWindow('frame')

cv2.createTrackbar('bright', 'frame', 0, 100, bright)
cv2.createTrackbar('dark', 'frame', 0, 100, dark)

now = datetime.datetime.now().strftime("%d_%H-%M-%S")

cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter("./" + str(now) + ".avi", fourcc, 20, (width, height), isColor=True)

if not cap.isOpened():
    print('video open failed!')
    sys.exit()

cam_mode = 0

while True:
        ret, frame = cap.read()

        if not ret:
            break
        
        key = cv2.waitKey(1)
        if cam_mode == 1:
            frame = cartoon_filter(frame)
        elif cam_mode == 2:
            frame = pencil_sketch(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif cam_mode == 3:
            frame = sharp(frame)

        cv2.imshow('frame', frame)
        
        if key == ord('s'):
            img_captured = cv2.imwrite('./' + str(now) + 'png', frame)
        
        if key == ord('r'):
            while True:
                ret_rec, frame_rec = cap.read()

                if ret_rec:
                    out.write(frame_rec)
                cv2.putText(frame_rec,"Recoding..",(0, 30),cv2.FONT_HERSHEY_COMPLEX,0.5,(50,50,255))
                img = cv2.imshow('frame', frame_rec)

                if cv2.waitKey(50) == ord("r"):
                    break
            out.release()
            cv2.waitKey(300)
            
        if key == 27:
            break
        if key == ord(' '):
            cam_mode += 1
            if cam_mode == 4:
                cam_mode = 0
out.release()
        
cap.release()

cv2.destroyAllWindows()
        