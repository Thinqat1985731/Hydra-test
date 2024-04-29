# Standard Library
import sys

# Third Party Library
import numpy as np
import numpy.typing as npt


class NeuralNetMLP(object):
    """Feedforward neural network / Multi-layer perceptron分類器
    パラメータ
    ------------
    n_hidden : int (default: 30)
        隠れ層の数
    l2 : float (default: 0.)
        L2正規化のlambda。l2 が 0 なら正規化を行わない
    epochs : int (default: 100)
        エポック数
    eta : float (default: 0.001)
        学習率
    shuffle : bool (default: True)
        各エポックで訓練データをシャッフルする
    minibatch_size : int (default: 1)
        ミニバッチの大きさ
    seed : int (default: None)
        シード値

    属性
    -----------
    eval_ : dict
      訓練における各エポックのcost, training accuracy, validation accuracy

    """

    def __init__(
        self,
        n_hidden: int = 30,
        l2: float = 0.0,
        epochs: int = 100,
        eta: float = 0.001,
        shuffle: bool = True,
        minibatch_size: int = 1,
        seed: int | None = None,
    ):
        self.random = np.random.RandomState(seed)
        self.n_hidden = n_hidden
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.shuffle = shuffle
        self.minibatch_size = minibatch_size

    def _onehot(self, y: npt.NDArray, n_classes: int) -> npt.NDArray:
        """ラベルをone-hotに変換する

        パラメータ
        ------------
        y : array, shape = [n_examples]
            目的関数
        n_classes : int
            クラスの数

        戻り値
        -----------
        onehot : array, shape = (n_examples, n_labels)

        """
        onehot = np.zeros((n_classes, y.shape[0]))
        for idx, val in enumerate(y.astype(int)):
            onehot[val, idx] = 1.0
        return onehot.T

    def _sigmoid(self, z: npt.NDArray) -> npt.NDArray:
        """
        ロジスティック関数（シグモイド）を計算する
        """
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

    def _forward(
        self, X: npt.NDArray
    ) -> (npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray):
        """
        順伝播のステップ
        """
        z_h = np.dot(X, self.w_h) + self.b_h  # 隠れ層の総入力
        a_h = self._sigmoid(z_h)  # 隠れ層の活性化関数
        z_out = np.dot(a_h, self.w_out) + self.b_out  # 出力層の総入力
        a_out = self._sigmoid(z_out)  # 出力層の活性化関数

        return z_h, a_h, z_out, a_out

    def _compute_cost(self, y_enc: npt.NDArray, output: npt.NDArray) -> float:
        """コスト計算

        パラメータ
        ----------
        y_enc : array, shape = (n_examples, n_labels)
            one-hotエンコードされたクラスラベル
        output : array, shape = [n_examples, n_output_units]
            出力層の活性化関数

        戻り値
        ---------
        cost : float
            正規化されたコスト

        """
        L2_term = self.l2 * (np.sum(self.w_h**2.0) + np.sum(self.w_out**2.0))

        term1 = -y_enc * (np.log(output))
        term2 = (1.0 - y_enc) * np.log(1.0 - output)
        cost = np.sum(term1 - term2) + L2_term

        term1 = -y_enc * (np.log(output + 1e-5))
        term2 = (1.0 - y_enc) * np.log(1.0 - output + 1e-5)
        return cost

    def predict(self, X: npt.NDArray) -> npt.NDArray:
        """クラスラベルの予測

        パラメータ
        -----------
        X : array, shape = [n_examples, n_features]
            ある特徴値が設定された入力

        戻り値
        ----------
        y_pred : array, shape = [n_examples]
            予測されたクラスラベル

        """
        _, _, z_out, _ = self._forward(X)
        y_pred = np.argmax(z_out, axis=1)
        return y_pred

    def fit(
        self,
        X_train: npt.NDArray,
        y_train: npt.NDArray,
        X_valid: npt.NDArray,
        y_valid: npt.NDArray,
    ):
        """訓練データから重みを学習

        パラメータ
        -----------
        X_train : array, shape = [n_examples, n_features]
            元の特徴量が設定された入力層
        y_train : array, shape = [n_examples]
            目的関数
        X_valid : array, shape = [n_examples, n_features]
            訓練時の検証に用いるサンプル特徴量
        y_valid : array, shape = [n_examples]
            訓練時の検証に用いるサンプルラベル

        Returns:
        ----------
        self

        """
        # 重みの初期化
        n_output = np.unique(y_train).shape[0]  # number of class labels
        n_features = X_train.shape[1]

        # 入力層 -> 隠れ層の重み
        self.b_h = np.zeros(self.n_hidden)
        self.w_h = self.random.normal(
            loc=0.0, scale=0.1, size=(n_features, self.n_hidden)
        )

        # 隠れ層の重み -> 出力層の重み
        self.b_out = np.zeros(n_output)
        self.w_out = self.random.normal(
            loc=0.0, scale=0.1, size=(self.n_hidden, n_output)
        )

        # 書式設定
        epoch_strlen = len(str(self.epochs))  # for progress formatting
        self.eval_ = {"cost": [], "train_acc": [], "valid_acc": []}

        y_train_enc = self._onehot(y_train, n_output)

        # エポック数だけ訓練を繰り返す
        for i in range(self.epochs):
            # ミニバッチの反復処理
            indices = np.arange(X_train.shape[0])

            if self.shuffle:
                self.random.shuffle(indices)

            for start_idx in range(
                0,
                indices.shape[0] - self.minibatch_size + 1,
                self.minibatch_size,
            ):
                batch_idx = indices[start_idx : start_idx + self.minibatch_size]

                # 順伝播
                _, a_h, _, a_out = self._forward(X_train[batch_idx])

                # 逆伝播
                delta_out = a_out - y_train_enc[batch_idx]

                sigmoid_derivative_h = a_h * (1.0 - a_h)

                delta_h = np.dot(delta_out, self.w_out.T) * sigmoid_derivative_h

                grad_w_h = np.dot(X_train[batch_idx].T, delta_h)
                grad_b_h = np.sum(delta_h, axis=0)

                grad_w_out = np.dot(a_h.T, delta_out)
                grad_b_out = np.sum(delta_out, axis=0)

                # 正規化と重みの更新
                delta_w_h = grad_w_h + self.l2 * self.w_h
                delta_b_h = grad_b_h  # bias is not regularized
                self.w_h -= self.eta * delta_w_h
                self.b_h -= self.eta * delta_b_h

                delta_w_out = grad_w_out + self.l2 * self.w_out
                delta_b_out = grad_b_out
                self.w_out -= self.eta * delta_w_out
                self.b_out -= self.eta * delta_b_out

            # 各エポックでの評価
            _, _, _, a_out = self._forward(X_train)

            cost = self._compute_cost(y_enc=y_train_enc, output=a_out)

            y_train_pred = self.predict(X_train)
            y_valid_pred = self.predict(X_valid)

            train_acc = (np.sum(y_train == y_train_pred)).astype(
                float
            ) / X_train.shape[0]
            valid_acc = (np.sum(y_valid == y_valid_pred)).astype(
                float
            ) / X_valid.shape[0]

            sys.stderr.write(
                "\r%0*d/%d | Cost: %.2f "
                "| Train/Valid Acc.: %.2f%%/%.2f%% "
                % (
                    epoch_strlen,
                    i + 1,
                    self.epochs,
                    cost,
                    train_acc * 100,
                    valid_acc * 100,
                )
            )
            sys.stderr.flush()

            self.eval_["cost"].append(cost)
            self.eval_["train_acc"].append(train_acc)
            self.eval_["valid_acc"].append(valid_acc)

        return self
