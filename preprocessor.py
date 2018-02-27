import PIL.Image as Image

def main():
    image = Image.open('./data/train/Maize/4ef677ce4.png')
    processed_image = preprocess(image)
    processed_image.show()

def preprocess(image, required_size=100):
    """
    Takes a square image and resizes it to have a side length of required_size, returns a PIL Image object
    """
    return image.resize((required_size, required_size))

if __name__ == '__main__':
    main()