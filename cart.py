from sklearn import tree

def CART(images, labels, images_test):
    model = tree.DecisionTreeClassifier(criterion='gini')
    model.fit(images, labels)
    model.score(images, labels)
    predicted = model.predict(images_test)
    
    return predicted