

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

model = tf.keras.models.load_model("D:/tensorflow learning/leaner_regression/hookelawmodel/hookelaw_model.h5")
print("Enter 'q' at any time to quit.\n")
while True:
	response = input("please input the F, and I will tell your the length: ")
	if response == 'q':
		break
	else:
		predicted = model.predict(pd.Series([str(response)])) 
		print(predicted)

input()