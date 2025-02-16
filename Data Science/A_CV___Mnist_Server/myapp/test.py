from model import Model
import emnist
import pandas as pd

images_train, labels_train = emnist.extract_training_samples('balanced')
images_test, labels_test = emnist.extract_test_samples('balanced')

mapping = pd.read_csv('./emnist-balanced-mapping.txt', sep=' ', names=['label', 'code'])
mapping['char'] = mapping['code'].apply(chr)
mapping.set_index('label', inplace=True)

m = Model()

pred = m.predict(images_test[0])
all_chars = mapping['char'].to_list()

print('Prediction:', pred)
print('All chars:', all_chars)

assert pred in all_chars

print('OK')
