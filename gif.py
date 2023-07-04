import imageio.v2 as imageio
import cv2

import os
filepath = r"E:\公司\地震数据\AETA地震数据\数据整理\EM_GA_2021_01_01-2021_01_31\time_zhandian"
a = []
for file in os.listdir(filepath):
    print(file)
    #a.append(file)
    a.append(os.path.join(filepath, file))
print(a)
filenames =a
'''
filenames=['7.png','8.png','9.png','10.png','11.png','12.png',
           '13.png','14.png','15.png','16.png','17.png',
           '18.png','19.png','20.png','21.png','22.png',
           '23.png','24.png','25.png','26.png','27.png','28.png','29.png',
           '30.png', '31.png', '32.png', '33.png', '34.png', '35.png',
           '36.png', '37.png', '38.png', '39.png', '40.png',
           '41.png', '42.png', '43.png', '44.png', '45.png',
           '46.png', '47.png', '48.png', '49.png', '50.png', '51.png', '52.png']
'''
images=[]
for filename in filenames:
    img = imageio.imread(filename)
    img = cv2.resize(img,(192,192))
    images.append(img)
    #images.append(imageio.imread(filename))
imageio.mimsave("image1.gif",images,'GIF',duration=0.1,)
