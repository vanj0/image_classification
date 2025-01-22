import numpy as np

class ImageTransformer:
    @staticmethod
    def rescale_image(image):

        if image.width != 28 or image.height != 28:
            image = image.resize((28, 28))
        return image

    @staticmethod
    def encode_image(image):

        gray_image = image.convert("L")
        image_array = np.array(gray_image)
        normalized_array = image_array / 255.0
        return normalized_array
