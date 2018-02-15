# -*-coding:utf8-*-
from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()
input_class = ['audi', 'ford', 'audi', 'toyota', 'ford', 'bmw']
label_encoder.fit(input_class)
print "\nClass mapping:"

for i, item in enumerate(label_encoder.classes_):
	print item, '-->', i
