import pytesseract
import os
from keras.preprocessing import image as keras_image
from preprocessing import preprocess_image, preprocess_txt
import pandas as pd

try:
    from PIL import Image
except ImportError:
    import Image
data = {"image": [], "filepath": [], "text": [], "label": []}
for dirname, _, filenames in os.walk('./dataset/'):
    for filename in filenames:
        try:
            if dirname == './dataset/positive':
                data['label'].append(int(1))
            if dirname == './dataset/neutral':
                data['label'].append(int(2))
            if dirname == './dataset/negative':
                data['label'].append(int(0))

            data['image'].append(
                preprocess_image(
                    keras_image.load_img(
                        os.path.join(dirname, filename),
                        target_size=(224, 224),
                        interpolation='bicubic')))

            data['filepath'].append(
                os.path.join(dirname, filename))

            data['text'].append(
                preprocess_txt(
                    pytesseract.image_to_string(
                        Image.open(
                            os.path.join(dirname, filename)))))

        except Exception as e:
            print(e)
            continue

        print('\r images: {} texts: {} labels : {}'.format(len(data['image']), len(data['text']), len(data['label'])), end='')

data = pd.DataFrame(data).convert_dtypes().sample(frac=1).reset_index(drop=True)
data.to_pickle("data.pkl")
