import numpy as np


# 弱分類器（切り株）の実装
class DecisionStump():
    def __init__(self):
        self.polarity = 1
        self.feature_idx = None
        self.threshold = None
        self.alpha = None

    # threshold を境に、-1,1のいずれかの予測を行う
    def predict(self, X):
        n_samples = X.shape[0]
        X_column = X[:, self.feature_idx]
        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column > self.threshold] = -1
        return predictions


class Adaboost():

    # n_clfは、繰り返す回数を表している
    def __init__(self, n_clf=5):
        self.n_clf = n_clf

    # -1を1に、1を0に変換する指示関数
    def transform(self, predictions, y):
        indicator = []
        for i in range(len(predictions)):
            if predictions[i] == y[i]:
                indicator.append(0)
            else:
                indicator.append(1)
        return indicator

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # 重みの初期値を1/N(i=1,2,...,N)に設定
        w = np.full(n_samples, (1/n_samples))

        self.clfs = []

        for _ in range(self.n_clf):
            clf = DecisionStump()

            min_error = float('inf')

            # 最も誤分類の少ない、ある特徴量におけるしきい値を求める
            for feature_i in range(n_features):
                X_column = X[:, feature_i]
                thresholds = np.unique(X_column)

                for threshold in thresholds:
                    polarity = 1
                    predictions = np.ones(n_samples)
                    predictions[X_column < threshold] = -1

                    misclassified = w[y != predictions]
                    # errorは、2(b)の分子（分母は正規化して1としているので無視して良い）
                    error = sum(misclassified)

                    if error > 0.5:
                        error = 1 - error
                        polarity = -1

                    if error < min_error:
                        clf.polarity = polarity
                        clf.threshold = threshold
                        clf.feature_idx = feature_i
                        min_error = error

            EPS = 1e-10
            clf.alpha = np.log((1.0 - min_error + EPS) / (min_error + EPS))

            predictions = clf.predict(X)

            indicator = np.array(self.transform(predictions, y))
            # 誤分類した観測値の重みを増し、正しく分類した観測値の重みを減らす
            w *= np.exp(clf.alpha * indicator)

            # wの合計を1に正規化
            w /= np.sum(w)

            self.clfs.append(clf)

    # self.alphaで重み付けして、符合関数で予測
    def predict(self, X):
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.clfs]
        y_pred = np.sum(clf_preds, axis=0)
        y_pred = np.sign(y_pred)
        return y_pred
