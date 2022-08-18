from sklearn import tree 
features = [[150,1],[170,1],[130,0],[140,0]] #重量,皺度
labels = [0,0,1,1]#假設0是橘子,1是蘋果
clf = tree.DecisionTreeClassifier()#二元樹分類
clf = clf.fit(features,labels)
wantPredict = clf.predict([[100,0]]) 
if wantPredict == [1]:
    print('This is an apple')
elif wantPredict == [0]:
    print('This is an orange')
