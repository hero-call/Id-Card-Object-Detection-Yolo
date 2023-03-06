# import timeit
# start = timeit.default_timer()

import TextDetection as td
import BalanceAngle as ba
import cv2 as cv



def main():
    filename = '02.jpeg'
    img = cv.imread(filename)

    while cv.waitKey(1) != ord('0'):
        cv.imshow("Rotated", img)

    angle = ba.find_angle(img)
    print(angle)
    if type(angle) == str:
        print(angle)
        return 0

    img = ba.rotate(img, angle)

    while cv.waitKey(1) != ord('0'):
        cv.imshow("Rotated", img)

    td.test_single_img(img)
    return 1


if __name__ == '__main__' :
    print(main())



# stop = timeit.default_timer()
# print('Time: ', stop - start)
