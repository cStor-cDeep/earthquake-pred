import pandas as pd
import numpy as np
#from scipy.misc import imsave
#import cv2 as cv
from PIL import Image


data =pd.read_csv("huidutest.csv",sep=',',encoding="utf-8")
data1 = np.array(data)
data2 = np.array(data1).astype(np.int32)
print(data2)
#data3 = pd.DataFrame(data2)
#data3.to_csv("test_huidu.csv",sep=',',encoding='utf-8')

image1 = Image.fromarray(data2)
image1 = image1.convert('L')
image1.save("output.png",dpi=(300,300))
