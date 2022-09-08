import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #可隱藏tensorflow 警告訊息
import tensorflow as tf
from tensorflow.keras import layers
from train_test_data import train_test_data
import functools
train=train_test_data.train
test=train_test_data.test
print(train)
print("========================================")
print(test)
print("========================================")
#CSV_COLUMNS = ['survived', 'sex', 'age', 'n_siblings_spouses', 'parch', 'fare', 'class', 'deck', 'embark_town', 'alone']
LABEL_COLUMN = 'survived'
LABELS = [0, 1]
def get_dataset(file_path):
  dataset = tf.data.experimental.make_csv_dataset(
      file_path,
      batch_size=12,
      label_name=LABEL_COLUMN,
      na_value="?",
      num_epochs=1,
      ignore_errors=True)
  return dataset

raw_train_data = get_dataset("train.csv")
raw_test_data = get_dataset("eval.csv")
CATEGORIES = {  #將data有種類的先抓出來處理
    'sex': ['male', 'female'],
    'class' : ['First', 'Second', 'Third'],
    'deck' : ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
    'embark_town' : ['Cherbourg', 'Southhampton', 'Queenstown'],
    'alone' : ['y', 'n']
}

categorical_columns = []
test_array=[]
for feature, vocab in CATEGORIES.items():
    cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
        key=feature, 
        vocabulary_list=vocab
        )
    #categorical_columns.append(tf.feature_column.indicator_column(cat_col))
    categorical_columns.append(cat_col)
    test_array.append(tf.feature_column.indicator_column(cat_col))
print("===============================")
for i in categorical_columns:
    print(i)
print("===============================")
for i in test_array:
    print(i)
print("===============================")
def process_continuous_data(mean, data):#標準化
    data = tf.cast(data, tf.float32) * 1/(2*mean)
    return tf.reshape(data, [-1, 1]) 
    #tf.reshape(tensor,[-1,1])轉成一维列向量 
    #tf.reshape(tensor,[1,-1])轉成一维行向量

MEANS = { #平均值
    'age' : 29.631308,
    'n_siblings_spouses' : 0.545455,
    'parch' : 0.379585,
    'fare' : 34.385399
}

numerical_columns = []

for feature in MEANS.keys():
  num_col = tf.feature_column.numeric_column(feature, normalizer_fn=functools.partial(process_continuous_data, MEANS[feature]))#對每列平均值進行標準化
  numerical_columns.append(num_col)

preprocessing_layer = tf.keras.layers.DenseFeatures(categorical_columns+numerical_columns) #將DATA組和數據組合併

model = tf.keras.Sequential([
  preprocessing_layer,
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])