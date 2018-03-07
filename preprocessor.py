import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt

def main():
    image = Image.open('./data/train/Maize/4ef677ce4.png')
    # processed_image = preprocess(image)
    # processed_image.show()

    green_mask = extract_green_mask(image)

    plt.imshow(green_mask)
    plt.show()


def preprocess(image, required_size=100):
    """
    Takes a square image and resizes it to have a side length of required_size, returns a PIL Image object
    """

    return image.resize((required_size, required_size))


def extract_green_mask(image, required_size=100):
    image = image.resize((required_size, required_size))

    hsv_image = image.convert('HSV')

    hsv_array = np.asarray(hsv_image)
    hue_array = hsv_array[:, :, 0]

    green_mask = np.greater(hue_array, 40) * np.less(hue_array, 55)

    return green_mask


if __name__ == '__main__':
    main()