import numpy as np
from dataset import createDataSet, createDataSet2
from sklearn.naive_bayes import GaussianNB


class NaiveBayes():
    """
    Naive Bayes Model, supporting both discrete and continuous feature inputs.
    p(c|X) = p(c|(x1, x2, ..., xm)
    p(c|X) = p(c) * p(X|c) / p(X)
    p'(c|X) = p(c) * p(X|c) = p(c) * p(x1|c) *...* p(xm|c)
    y_pred = argmax_{c}(p'(c|X))
    """
    def __init__(self, all_continuous=False):
        self.prob_label = {}
        self.prob_feat = {}
        self.feat_dim = None
        self.feat_type = []
        self.epsilon = 1e-7
        self.all_continuous = all_continuous

    def fit(self, train_x, train_y):
        """
        :param train_x: np.ndarray, [train_num, feat_dim]
        :param train_y:  np.ndarray, [train_num]
        """
        assert type(train_x).__name__ == 'ndarray', 'The type of input x must be np.ndarray !'
        assert len(train_x.shape) == 2, 'The shape of input x must be 2D !'
        assert type(train_y).__name__ == 'ndarray', 'The type of input y must be np.ndarray !'

        self.feat_dim = train_x.shape[1]
        unique_y = set(list(train_y))
        for y in unique_y:
            prob_y = self.laplacian_smoothing((train_y == y).sum(), len(train_y), len(unique_y))
            self.prob_label[y] = prob_y
            self.prob_feat[y] = {}
            train_x_ = train_x[train_y == y]
            for feat_idx in range(self.feat_dim):
                total_feat_v = train_x[:, feat_idx]  # here use feature value in whole data
                unique_feat = set(list(total_feat_v))
                feat_v_train_x_ = train_x_[:, feat_idx]
                feat_type = self.check_data_type(total_feat_v)
                if feat_type == 'int':
                    for feat_v in unique_feat:
                        feat_v_size = (feat_v == feat_v_train_x_).sum()
                        self.prob_feat[y]['{}_{}'.format(feat_idx, int(feat_v))] = self.laplacian_smoothing(feat_v_size,
                                                                                                         len(train_x_),
                                                                                                         len(unique_feat))
                if feat_type == 'float64':
                    if len(self.prob_feat[y]) is None:
                        self.prob_feat[y] = {}
                    v_mean = np.mean(feat_v_train_x_)
                    v_std = np.std(feat_v_train_x_, ddof=1)
                    self.prob_feat[y][f'{feat_idx}_mean'] = v_mean
                    self.prob_feat[y][f'{feat_idx}_std'] = v_std

    def predict(self, test_x):
        """
        :param test_x: np.ndarray, [test_num, feat_dim]
        :return: predicting result, [test_num]
        """
        assert type(test_x).__name__ == 'ndarray', 'The type of input x must be np.ndarray !'
        assert len(test_x.shape) == 2, 'The shape of input x must be 2D !'

        pred = []
        for i in range(len(test_x)):
            test_vec = test_x[i]
            pred.append(self.predict_(test_vec))
        pred = np.array(pred)

        return pred

    def score(self, test_x, target):
        """
        :param test_x: np.ndarray, [test_num, feat_dim]
        :param target: np.ndarray, [test_num]
        :return: accuracy in test data, float.
        """
        assert type(test_x).__name__ == 'ndarray', 'The type of input x must be np.ndarray !'
        assert len(test_x.shape) == 2, 'The shape of input x must be 2D !'
        pred_y = self.predict(test_x)
        return self.calc_accuracy(pred_y, target)

    def predict_(self, test_vec):
        """
        :param test_vec: [feat_dim]
        :return: label
        """
        posterior_prob = []
        for y in self.prob_label:
            prob = self.prob_label[y]
            for feat_idx in range(self.feat_dim):
                feat_value = test_vec[feat_idx]
                if self.feat_type[feat_idx] == 'int':
                    prob = self.prob_feat[y]['{}_{}'.format(feat_idx, int(feat_value))]
                    prob *= self.prob_feat[y]['{}_{}'.format(feat_idx, int(feat_value))]
                if self.feat_type[feat_idx] == 'float64':
                    mu = self.prob_feat[y][f'{feat_idx}_mean']
                    sigma = self.prob_feat[y][f'{feat_idx}_std']
                    prob *= self.normfun(feat_value, mu, sigma)
            posterior_prob.append(prob)
        posterior_prob = np.array(posterior_prob)

        return np.argmax(posterior_prob)

    def laplacian_smoothing(self, Dx, D, D_size):
        """
        :param Dx:
        :param D:
        :return: prob after laplacian smoothing
        """
        return (Dx + 1) / float(D + D_size)

    def check_data_type(self, feat_values):
        """
        :param feat_values: [data_num]
        """
        if self.all_continuous:
            self.feat_type.append('float64')
            return 'float64'
        feat_values_int = feat_values.astype('int')
        dis = abs(feat_values_int.astype('float64') - feat_values).sum()
        if dis < self.epsilon:
            self.feat_type.append('int')
            return 'int'
        self.feat_type.append('float64')
        return 'float64'

    def normfun(self, x, mu, sigma):
        pdf = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
        return pdf

    def calc_accuracy(self, pred, target):
        """
        :param pred: np.ndarray, [test_size]
            Predicted labels.
        :param target: np.ndarray, [test_size]
            Ground truth targets.
        :return: accuracy, float
        """
        assert pred.shape == target.shape, 'The shape of pred must equals to target!'

        return (pred == target).sum() / float(len(target))


def cross_validation(model_name='GaussionNB'):
    x, y, _, _ = createDataSet2()
    correct = 0
    for i in range(len(x)):
        test_index = i
        train_x = np.concatenate([x[:test_index], x[test_index + 1:]], axis=0)
        train_y = np.concatenate([y[:test_index], y[test_index + 1:]], axis=0)
        test_x = x[test_index].reshape(1, -1)
        test_y = y[test_index]
        if model_name == 'GaussionNB':
            model = GaussianNB()
        if model_name == 'NaiveBayes':
            model = NaiveBayes()
        model.fit(train_x, train_y)
        pred = model.predict(test_x)

        correct += (pred == test_y).sum()
    acc = float(correct) / float(len(x))
    print('acc:%.4f' % acc)


if __name__ == '__main__':
    x, y, _, id2feat_mapping = createDataSet2()
    print(id2feat_mapping)
    test_index = 0
    train_x = x
    train_y = y
    test_x = x[test_index].reshape(-1, x.shape[1])
    test_y = y[test_index]
    print('train size:', train_x.shape[0])
    print('test size:', test_x.shape[0])
    model = NaiveBayes()
    model.fit(train_x, train_y)
    pred = model.predict(test_x)
    print('true label:', test_y)
    print('pred label:', pred)

