import cv2
import numpy as np

def getFeatures(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        ret, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV)

        # Find contours
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return None

        cnt = max(contours, key=cv2.contourArea)

        # Area and perimeter
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        # Color features
        R = np.mean(image[:, :, 2])
        G = np.mean(image[:, :, 1])
        B = np.mean(image[:, :, 0])

        # PCA/Eigen values
        data_pts = np.array(cnt, dtype=np.float64).reshape(-1, 2)
        mean = np.empty((0))
        mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)

        eig1 = eigenvalues[0][0]
        eig2 = eigenvalues[1][0]

        eccentricity = np.sqrt(1 - (eig2 / eig1))

        return [area, perimeter, R, B, G, eig1, eig2, eccentricity]

    except Exception as e:
        print(f"Error extracting features: {e}")
        return None
