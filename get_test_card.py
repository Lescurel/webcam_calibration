import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.io import imread, imshow
from skimage import feature

def get_roi_contours(image):
    img_g = rgb2gray(img)
    contours = measure.find_contours(img_g, 0.8)
    # ascending sort, biggest contour is the last one
    roi_contour = sorted(contours, key=lambda x:x.shape[0])[-1]
    return roi_contour 

if "__main__" == __name__:
    img = imread("./images/test_card_orig.jpg")
    c = get_roi_contours(img)
    fig, ax = plt.subplots()
    ax.imshow(img_g, cmap=plt.cm.gray)
    ax.plot(c[:, 1], c[:, 0], linewidth=2, color="red")
    plt.show()
