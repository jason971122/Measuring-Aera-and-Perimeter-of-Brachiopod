import cv2 as cv
import numpy as np
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import pandas as pd
import argparse
import imutils


def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# canny边缘检测
def canny_demo(image):
    t = 80
    canny_output = cv.Canny(image, t, t * 2)
    cv.imshow("canny_output", canny_output)
    cv.imwrite("./66.png", canny_output)
    return canny_output

# 读取图像
src = cv.imread("./7mm.jpg")
cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
cv.imshow("input", src)

# 调用
binary = canny_demo(src)
k = np.ones((3, 3), dtype=np.uint8)
binary = cv.morphologyEx(binary, cv.MORPH_DILATE, k)

# 轮廓发现
contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
pixelsPerMetric = None
areas_list = []
perimeter_list = []
for c in range(len(contours)):
    # 计算轮廓所包含的面积
    area = cv.contourArea(contours[c])
    # print(area)
    # 计算轮廓的周长
    perimeter = cv.arcLength(contours[c], True)
    # print(perimeter)0


    rect = cv.minAreaRect(contours[c])
    box2 = cv.BoxPoints(rect) if imutils.is_cv2() else cv.boxPoints(rect)
    box1 = np.array(box2)
    box = np.around(box1,2)

    cx, cy = rect[0]
    # box = cv.boxPoints(rect)
    # box = np.int0(box)
   # cv.drawContours(src,[box],0,(0,255,0),2)
    cv.circle(src, (np.int64(cx), np.int64(cy)), 2, (255, 0, 0), 2, 8, 0)
    cv.drawContours(src, contours, c, (0, 0, 255), 2, 8)
    # box1 = cv.minAreaRect(c)



    # order the points in the contour such that they appear
    # in top-left, top-right, bottom-right, and bottom-left
    # order, then draw the outline of the rotated bounding
    # box
    box = perspective.order_points(box)
    cv.drawContours(src, [box.astype("int")], -1, (0, 255, 0), 2)

    # loop over the srcinal points and draw them
    for (x, y) in box:
        cv.circle(src, (int(x), int(y)), 5, (0, 0, 255), -1)

    # unpack the ordered bounding box, then compute the midpoint
    # between the top-left and top-right coordinates, followed by
    # the midpoint between bottom-left and bottom-right coordinates
    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)

    # compute the midpoint between the top-left and top-right points,
    # followed by the midpoint between the top-righ and bottom-right
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    # draw the midpoints on the image
    cv.circle(src, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
    cv.circle(src, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
    cv.circle(src, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
    cv.circle(src, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

    # draw lines between the midpoints
    cv.line(src, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
             (255, 0, 255), 2)
    cv.line(src, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
             (255, 0, 255), 2)

    # compute the Euclidean distance between the midpoints
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

    # if the pixels per metric has not been initialized, then
    # compute it as the ratio of pixels to supplied metric
    # (in this case, inches)
    if pixelsPerMetric is None:
        width = 6.29
        # pixelsPerMetric = dB / args["width"]
        pixelsPerMetric = dB / width

    # compute the size of the object
    dimA = dA / pixelsPerMetric
    dimB = dB / pixelsPerMetric
    area = area / pixelsPerMetric**2
    perimeter = perimeter / pixelsPerMetric
    # print(area)
    # print(perimeter)
    areas_list.append(area)
    perimeter_list.append(perimeter)



    # draw the object sizes on the image
    cv.putText(src, "{:.1f}mm".format(dimA),
                (int(tltrX - 15), int(tltrY - 10)), cv.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)
    cv.putText(src, "{:.1f}mm".format(dimB),
                (int(trbrX + 10), int(trbrY)), cv.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)
    cv.putText(src, "{:.1f}mm (perimeter)".format(perimeter),
                (int(trbrX), int(trbrY+30)), cv.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)
    # cv.putText(src, "{:.1f}mm^2 (area)".format(area),
    #             (int(trbrX), int(trbrY-30)), cv.FONT_HERSHEY_SIMPLEX,
    #             0.65, (255, 255, 255), 2)
    cv.putText(src, "{:.2f}mm".format(perimeter),
                (int(trbrX-60), int(trbrY-100)), cv.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)
    cv.putText(src, "{:.2f}mm^2".format(area),
                (int(trbrX-60), int(trbrY-80)), cv.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)

# ((cx, cy), (width, height), theta) = cv.minAreaRect(contours)
# print(areas_list)
# print(perimeter_list)
# for cidx,cnt in enumerate(contours):
#     ((cx, cy), (width, height), theta) = cv.minAreaRect(cnt)
#     print(cidx, 'center: cx=%.3f, cy=%.3f, width=%.3f, height=%.3f, roate_angle=%.3f'%(cx, cy, width, height, theta))
df_aera = pd.DataFrame(areas_list, columns=['AREA'])
df_perimeter = pd.DataFrame(perimeter_list, columns=['PERIMETER'])
df_S_D = pd.concat([df_aera,df_perimeter],axis=1)
df_S_D = df_S_D.round(decimals=2)
df_S_D.to_csv('Aera_Perimeter_7mm.csv')
# 图像显示
# cv.imshow("contours_analysis", src)
# cv.imwrite("./results/results_85_mm.png", src)
# cv.waitKey(0)
# cv.destroyAllWindows()