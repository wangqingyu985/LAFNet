"""
Created by Mr. Qingyu Wang at 14:33 07.03.2025
E-mail address: 12013027@zju.edu.cn
"""
import cv2
import time
import serial
import numpy as np
import matplotlib
matplotlib.use('TKAgg')


fingertipImg = np.zeros((1000, 1000, 3), dtype=np.uint8)
fingertipImg[:], mean_force, mean_force_KF, desired_force_list, time_list = (255, 255, 255), [], [], [], []
coordinate_1 = np.array([(120, 220), (320, 220), (520, 220), (720, 220), (120, 420), (320, 420), (520, 420), (720, 420),
                         (120, 620), (320, 620), (520, 620), (720, 620), (120, 820), (320, 820), (520, 820), (720, 820)])
coordinate_2 = np.array([(280, 380), (480, 380), (680, 380), (880, 380), (280, 580), (480, 580), (680, 580), (880, 580),
                         (280, 780), (480, 780), (680, 780), (880, 780), (280, 980), (480, 980), (680, 980), (880, 980)])
fingertip_ch_digital = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
mean_force_list, max_force_list, fingertip_force_list = [], [], []


def main():
    force_com = serial.Serial(port='/dev/ttyUSB2', baudrate=115200, timeout=0.1)
    if 1:
        force_com.write(bytes.fromhex('AA 01 00 00 00 00 00 FF'))
        start_time = time.time()
        while True:
            ReadLine = force_com.read(38).hex()
            if len(ReadLine) != 76 or (ReadLine[0] + ReadLine[1] + ReadLine[2] + ReadLine[3]) != '2400':
                force_com.close()
                force_com.open()
                continue
            fingertip_color = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            for i in range(0, 8):
                fingertip_ch_digital[i] = int('0x' + ReadLine[4*i+4] + ReadLine[4*i+5], 16) * 256 + int('0x' + ReadLine[4*i+6] + ReadLine[4*i+7], 16)
            for i in range(8, 16):
                fingertip_ch_digital[i] = int('0x' + ReadLine[4*i+10] + ReadLine[4*i+11], 16) * 256 + int('0x' + ReadLine[4*i+12] + ReadLine[4*i+13], 16)
            for i in range(0, 16):
                fingertip_color[i] = (1 - ((fingertip_ch_digital[i] - 1200) / 3366)) * 255
            fingertip_color = tuple([int(x) for x in fingertip_color])
            fingertip_force = fingertip_ch_digital * (5/3366) - (1000/561)
            print('mean force:', np.mean(fingertip_force))
            mean_force_list.append(np.mean(fingertip_force))
            max_force_list.append(np.max(fingertip_force))

            for i in range(0, 16):
                cv2.rectangle(img=fingertipImg, pt1=coordinate_1[i], pt2=coordinate_2[i], color=(fingertip_color[i], fingertip_color[i], fingertip_color[i]), thickness=-1)
                cv2.rectangle(fingertipImg, coordinate_1[i], coordinate_2[i], (100, 100, 100), 3)
                cv2.circle(fingertipImg, (int((coordinate_1[i][0]+coordinate_2[i][0])/2),
                                          int((coordinate_1[i][1]+coordinate_2[i][1])/2)), 26, (204, 242, 255), -1)
                cv2.putText(fingertipImg, str(i+1), (int((coordinate_1[i][0]+coordinate_2[i][0])/2-15),
                            int((coordinate_1[i][1]+coordinate_2[i][1])/2)+5), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (136, 63, 0), 3)
            cv2.line(fingertipImg, (60, 1000), (60, 300), (136, 63, 0), 5)
            cv2.line(fingertipImg, (60, 300), (120, 150), (136, 63, 0), 5)
            cv2.line(fingertipImg, (120, 150), (880, 150), (136, 63, 0), 5)
            cv2.line(fingertipImg, (880, 150), (940, 300), (136, 63, 0), 5)
            cv2.line(fingertipImg, (940, 300), (940, 1000), (136, 63, 0), 5)
            cv2.putText(img=fingertipImg, text='Average Grasping Force: %.1f N' % np.mean(fingertip_force),
                        org=(25, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.9, color=(136, 63, 0),
                        thickness=6)
            cv2.namedWindow('fingertip matrix force', cv2.WINDOW_NORMAL)
            cv2.resizeWindow(winname='fingertip matrix force', height=800, width=800)
            cv2.imshow(winname='fingertip matrix force', mat=fingertipImg)
            cv2.waitKey(delay=1)
            fingertipImg[:] = (255, 255, 255)
            if time.time() - start_time > 100:
                break


if __name__ == "__main__":
    main()
