from sklearn import tree
from Alexnet_3dversion import Alexnet
import torch
import numpy as np

def CART(images, labels, images_test):
    model = tree.DecisionTreeClassifier(criterion='gini')
    model.fit(images, labels)
    model.score(images, labels)
    predicted = model.predict(images_test)
    
    return predicted


def main():
    net = Alexnet()
    path = './weights/alexnet_weight_svm.pt'
    net.load_state_dict(torch.load(path))
    data_control = np.load('data_normalized.npy')
    data_control = data_control[:, np.newaxis, :, :, :]
    # np.random.shuffle(data_control)
    data_control = torch.from_numpy(data_control.astype(float)).float()
    data_pd = np.load('PDdata.npy')
    data_pd = data_pd[:, np.newaxis, :, :, :]
    # np.random.shuffle(data_pd)
    data_pd = torch.from_numpy(data_pd.astype(float)).float()
    feature_ctr = net.features(data_control).view(data_control.size(0), 256 * 6 * 6 * 2).detach().numpy()
    feature_pd = net.features(data_pd).view(data_pd.size(0), 256 * 6 * 6 * 2).detach().numpy()
    age_ctr = np.load('age.npy')
    age_pd = np.load('PDage.npy')
    weight_ctr = np.load('weight.npy')
    weight_pd = np.load('PDweight.npy')
    sex_ctr = np.load('sex.npy')
    sex_pd = np.load('PDsex.npy')
    feature_ctr = np.concatenate((feature_ctr, age_ctr[:, 0].reshape(-1, 1), weight_ctr[:, 0].reshape(-1, 1), sex_ctr[:, 0].reshape(-1, 1)), axis=1)
    feature_pd = np.concatenate((feature_pd, age_pd[:, 0].reshape(-1, 1), weight_pd[:, 0].reshape(-1, 1), sex_pd[:, 0].reshape(-1, 1)), axis=1)
    y_control = -np.ones(feature_ctr.shape[0])
    y_pd = np.ones(feature_pd.shape[0])

    data_control_train = feature_ctr[0:int(feature_ctr.shape[0] * 0.8)]
    data_control_test = feature_ctr[int(feature_ctr.shape[0] * 0.8):]
    y_control_train = y_control[0:int(feature_ctr.shape[0] * 0.8)]
    y_control_test = y_control[int(feature_ctr.shape[0] * 0.8):]
    data_pd_train = feature_pd[0:int(feature_pd.shape[0] * 0.8)]
    data_pd_test = feature_pd[int(feature_pd.shape[0] * 0.8):]
    y_pd_train = y_pd[0:int(feature_pd.shape[0] * 0.8)]
    y_pd_test = y_pd[int(feature_pd.shape[0] * 0.8):]

    data_train = np.concatenate((data_control_train, data_pd_train), axis=0)
    y_train = np.concatenate((y_control_train, y_pd_train), axis=0)
    data_test = np.concatenate((data_control_test, data_pd_test), axis=0)
    y_test = np.concatenate((y_control_test, y_pd_test), axis=0)

    prediction = CART(data_train, y_train, data_test)
    correct = np.sum(prediction==y_test)
    print(correct/prediction.shape[0])


if __name__ == "__main__":
    main()





