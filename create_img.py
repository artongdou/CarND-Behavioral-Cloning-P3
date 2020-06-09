import matplotlib.pyplot as plt

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
plt.show()