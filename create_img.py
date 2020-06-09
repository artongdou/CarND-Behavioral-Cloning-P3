import matplotlib.pyplot as plt
from simple_model import preprocess_img, random_shift
import cv2
import numpy as np

# fig, axs = plt.subplots(2, 3, figsize=(14, 6))

a1 = plt.subplot2grid((4,3),(0,0),colspan = 3, rowspan=3)
a2 = plt.subplot2grid((4,3),(3,0))
a3 = plt.subplot2grid((4,3),(3,1))
a4 = plt.subplot2grid((4,3),(3,2))

img = plt.imread('demo_imgs/StartPos.png')
imgl = plt.imread('demo_imgs/left_StartPos.jpg')
imgc = plt.imread('demo_imgs/center_StartPos.jpg')
imgr = plt.imread('demo_imgs/right_StartPos.jpg')

# a1.imshow(imgl)
# a2.imshow(imgl)
# a3.imshow(imgl)
# a4.imshow(imgl)

a1.imshow(img)
a1.axis('off')

a2.imshow(imgl)
a2.set_title('Left Camera')
a2.axis('off')

a3.imshow(imgc)
a3.set_title('Center Camera')
a3.axis('off')

a4.imshow(imgr)
a4.set_title('Right Camera')
a4.axis('off')
# plt.show()

imgl = cv2.imread('demo_imgs/left_StartPos.jpg')
imgc = cv2.imread('demo_imgs/center_StartPos.jpg')
imgr = cv2.imread('demo_imgs/right_StartPos.jpg')

# l, r, _ = random_shift(imgc)
# fl, fr, _ = random_shift(np.fliplr(imgc))
fig, axes = plt.subplots(1,2, figsize=(16,4))
    
axes[0].imshow(imgc[:,:,::-1])
axes[0].axis('off')
axes[0].set_title('3@160x320')

axes[1].imshow(np.squeeze(preprocess_img(imgc)), cmap='gray')
axes[1].set_title('1@16x32')
axes[1].axis('off')
# ax2.imshow(np.reshape((preprocess_img(imgc)), (img_rows, img_cols)), cmap='gray')
# ax3.imshow(np.reshape((preprocess_img(imgr)), (img_rows, img_cols)), cmap='gray')

pp_img = preprocess_img(imgc)
l, r, _ = random_shift(pp_img)
flip_l, flip_r, _ = random_shift(np.fliplr(pp_img))
fig, axes = plt.subplots(3,2, figsize=(16,32))
axes[0,0].imshow(np.squeeze(pp_img), cmap='gray')
axes[0,0].axis('off')
axes[0,0].set_title('original')
axes[0,1].imshow(np.squeeze(np.fliplr(pp_img)), cmap='gray')
axes[0,1].axis('off')
axes[0,1].set_title('flipped')

axes[1,0].imshow(np.squeeze(l), cmap='gray')
axes[1,0].axis('off')
axes[1,0].set_title('original (shifted left)')
axes[2,0].imshow(np.squeeze(r), cmap='gray')
axes[2,0].axis('off')
axes[2,0].set_title('original (shifted right)')

axes[1,1].imshow(np.squeeze(flip_l), cmap='gray')
axes[1,1].axis('off')
axes[1,1].set_title('flipped (shifted left)')
axes[2,1].imshow(np.squeeze(flip_r), cmap='gray')
axes[2,1].axis('off')
axes[2,1].set_title('flipped (shifted right)')

plt.show()