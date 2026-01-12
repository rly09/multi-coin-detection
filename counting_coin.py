import cv2
import numpy as np

img = cv2.imread("images/20.jpg")
img = cv2.resize(img, (640, 800))
output = img.copy()

# ---------------- GRAYSCALE ----------------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("1 - Grayscale", gray)

# ---------------- NOISE REDUCTION ----------------
denoise = cv2.bilateralFilter(gray, 11, 17, 17)
cv2.imshow("2 - Noise Reduction", denoise)

# ---------------- GAUSSIAN BLUR ----------------
blur = cv2.GaussianBlur(denoise, (9, 9), 2)
cv2.imshow("3 - Gaussian Blur", blur)

# ---------------- CANNY EDGE ----------------
edges = cv2.Canny(blur, 80, 200)
cv2.imshow("4 - Canny Edge", edges)

# ---------------- HOUGH CIRCLE ----------------
circles = cv2.HoughCircles(
    blur,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=60,
    param1=120,
    param2=35,
    minRadius=20,
    maxRadius=120
)

count = 0

if circles is not None:
    circles = np.round(circles[0]).astype(int)

    circles = sorted(circles, key=lambda x: x[0])

    for i, (x, y, r) in enumerate(circles):
        count += 1

        cv2.circle(output, (x, y), r, (0, 255, 0), 3)
        cv2.circle(output, (x, y), 3, (0, 0, 255), -1)

        cv2.putText(
            output, str(i+1),
            (x-10, y+10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,(255,0,0),2
        )

# ---------------- FINAL OUTPUT ----------------
cv2.putText(
    output,
    f"Total Coins: {count}",
    (20,40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,(0,0,255),2
)

print("Number of coins:", count)

cv2.imshow("5 - Final Coin Detection", output)

cv2.waitKey(0)
cv2.destroyAllWindows()
