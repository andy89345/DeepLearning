import pandas as pd
import numpy as np

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)  #precision 小數點位數， suppress是否要抑制小數點顯示位數(false則全顯示)


import tensorflow as tf
from tensorflow.keras import layers
abalone_train = pd.read_csv(
    "abalone_train.csv",
    names=[ 
            "Length",
            "Diameter",
            "Height",
            "Whole weight",
            "Shucked weight",
            "Viscera weight",
            "Shell weight", 
            "Age"
            ]
    )
print(abalone_train)

abalone_features = abalone_train.copy()  #如果不用copy，那在pop之後也會將原本的train_data pop出來
abalone_labels = abalone_features.pop('Age') #將Age單獨pop出來，此時abalone_features就會少一項Age
abalone_features_ArrayData = np.array(abalone_features)
print("\n----------FEATURES-------------")
print(abalone_features_ArrayData)
print("---------------------------------")
abalone_lable_ArrayData=np.array(abalone_labels)
print("\n----------LABLES-------------")
print(abalone_lable_ArrayData)
print("---------------------------------")

abalone_model = tf.keras.Sequential([
    
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(1, activation="relu"),
    
])

abalone_model.compile(loss = tf.keras.losses.MeanSquaredError(),
                      optimizer = tf.keras.optimizers.Adam())
abalone_model.fit(abalone_features_ArrayData, abalone_lable_ArrayData, epochs=30,validation_split = 0.1) #validation_split 90%資料給訓練集