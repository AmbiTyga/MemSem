import pytesseract
import os
from preprocessing import preprocess_image, preprocess_txt
import pickle
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
                    os.path.join(dirname, filename)))

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


with open("data.pkl", "wb") as pickle_out:
    pickle.dump(data, pickle_out)
pickle_out.close()
