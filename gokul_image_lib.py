import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO


# EXP 1A - Read and Display Image
def read_display(filename):
    img = cv2.imread(filename)

    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# EXP 1B - Convert to Grayscale
def grayscale(filename):
    img = cv2.imread(filename)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Gray Image", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# EXP 1C - Resize Image
def resize_image(filename):
    img = cv2.imread(filename)

    resize = cv2.resize(img, (300, 300))

    cv2.imshow("Resized", resize)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# EXP 1D - Crop Image
def crop_image(filename):
    img = cv2.imread(filename)

    crop = img[50:200, 50:200]

    cv2.imshow("Cropped", crop)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# EXP 1E - Rotate Image
def rotate_image(filename):
    img = cv2.imread(filename)

    rows, cols = img.shape[:2]

    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)

    rotated = cv2.warpAffine(img, M, (cols, rows))

    cv2.imshow("Rotated", rotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# EXP 2A - Sobel Edge Detection
def sobel_edge(filename):
    img = cv2.imread(filename, 0)

    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1)

    cv2.imshow("Sobel X", sobelx)
    cv2.imshow("Sobel Y", sobely)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# EXP 2B - Canny Edge Detection
def canny_edge(filename):
    img = cv2.imread(filename, 0)

    edges = cv2.Canny(img, 100, 200)

    cv2.imshow("Canny Edge", edges)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# EXP 2C - Prewitt Edge Detection
def prewitt_edge(filename):
    img = cv2.imread(filename, 0)

    kernelx = np.array([[1, 0, -1],
                        [1, 0, -1],
                        [1, 0, -1]])

    kernely = np.array([[1, 1, 1],
                        [0, 0, 0],
                        [-1, -1, -1]])

    x = cv2.filter2D(img, -1, kernelx)
    y = cv2.filter2D(img, -1, kernely)

    cv2.imshow("Prewitt X", x)
    cv2.imshow("Prewitt Y", y)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# EXP 2D - Hough Line Detection
def hough_line(filename):
    img = cv2.imread(filename)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Lines", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# EXP 2E - Harris Corner Detection
def harris_corner(filename):
    img = cv2.imread(filename)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)

    corners = cv2.cornerHarris(gray, 2, 3, 0.04)

    img[corners > 0.01 * corners.max()] = [0, 0, 255]

    cv2.imshow("Corners", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# EXP 3 - Camera Calibration
def camera_calibration(filename):
    img = cv2.imread(filename)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (7, 7), None)

    if ret:
        cv2.drawChessboardCorners(img, (7, 7), corners, ret)

    cv2.imshow("Calibration", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# EXP 4 - Histogram Equalization
def histogram_equalization(filename):
    img = cv2.imread(filename, 0)

    equalized = cv2.equalizeHist(img)

    cv2.imshow("Original", img)
    cv2.imshow("Equalized", equalized)

    plt.subplot(1, 2, 1)
    plt.hist(img.ravel(), 256, [0, 256])
    plt.title("Original Histogram")

    plt.subplot(1, 2, 2)
    plt.hist(equalized.ravel(), 256, [0, 256])
    plt.title("Equalized Histogram")

    plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# EXP 5 - Skin Color Detection
def skin_detection(filename):
    img = cv2.imread(filename)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 48, 80])
    upper = np.array([20, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    skin = cv2.bitwise_and(img, img, mask=mask)

    cv2.imshow("Skin Detection", skin)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# EXP 6 - Image Warping
def image_warping(filename):
    img = cv2.imread(filename)

    rows, cols = img.shape[:2]

    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[10, 100], [200, 50], [100, 250]])

    M = cv2.getAffineTransform(pts1, pts2)

    warp = cv2.warpAffine(img, M, (cols, rows))

    cv2.imshow("Warped", warp)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# EXP 7 - Motion Tracking
def motion_tracking():
    cap = cv2.VideoCapture(0)

    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    while cap.isOpened():
        diff = cv2.absdiff(frame1, frame2)

        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)

        cv2.imshow("Motion", thresh)

        frame1 = frame2
        ret, frame2 = cap.read()

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# EXP 8 - Object Detection using YOLO
def yolo_detection(filename):
    model = YOLO("yolov8n.pt")

    results = model(filename)

    img = results[0].plot()

    cv2.imshow("YOLO Detection", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# EXP 9 - Stereo Vision and Depth Estimation
def stereo_depth(left_image, right_image):
    left = cv2.imread(left_image, 0)
    right = cv2.imread(right_image, 0)

    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

    disparity = stereo.compute(left, right)

    cv2.imshow("Depth Map", disparity)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# EXP 10 - Augmented Reality using Feature Matching
def feature_matching(first_image, second_image):
    img1 = cv2.imread(first_image, 0)
    img2 = cv2.imread(second_image, 0)

    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    matches = bf.match(des1, des2)

    result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None)

    cv2.imshow("Feature Matching", result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
