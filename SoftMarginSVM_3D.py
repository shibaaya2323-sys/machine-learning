
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def solve_soft_margin_svm_3d(X, y, C=1.0, plot=True):
    """
    3次元ソフトマージンSVMを scipy.optimize.minimize で解く関数

    Parameters
    ----------
    X : ndarray, shape (N, 3)
        訓練データ

    y : ndarray, shape (N,)
        正解ラベル
        +1 または -1

    C : float
        ソフトマージンのパラメータ

    plot : bool
        Trueなら分類境界とマージン平面を3次元表示

    Returns
    -------
    result : scipy OptimizeResult
        最適化結果

    w : ndarray, shape (3,)
        最適な重みベクトル

    b : float
        バイアス

    xi : ndarray, shape (N,)
        スラック変数
    """

    # ============================================================
    # 1. 入力
    # ============================================================

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    N = len(X)


    # ============================================================
    # 2. 入力チェック
    # ============================================================

    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError(
            "X は shape (N, 3) の3次元データにしてください。"
        )

    if y.ndim != 1:
        raise ValueError(
            "y は1次元配列にしてください。"
        )

    if len(y) != N:
        raise ValueError(
            "X のデータ数と y のラベル数が一致していません。"
        )

    if not np.all(np.isin(y, [-1.0, 1.0])):
        raise ValueError(
            "ラベル y は +1 または -1 にしてください。"
        )

    if C <= 0:
        raise ValueError(
            "C は正の値にしてください。"
        )


    # ============================================================
    # 3. 目的関数
    #
    # theta =
    # [w1, w2, w3, b, xi1, xi2, ..., xiN]
    #
    # min 1/2 ||w||^2 + C Σxi
    # ============================================================

    def objective(theta):

        w = theta[:3]
        xi = theta[4:]

        return (
            0.5 * np.dot(w, w)
            + C * np.sum(xi)
        )


    # ============================================================
    # 4. マージン制約
    #
    # y_i (w^T x_i + b) >= 1 - xi_i
    #
    # ↓
    #
    # y_i (w^T x_i + b) - 1 + xi_i >= 0
    # ============================================================

    def margin_constraint(theta):

        w = theta[:3]
        b = theta[3]
        xi = theta[4:]

        return (
            y * (X @ w + b)
            - 1.0
            + xi
        )


    # ============================================================
    # 5. xi_i >= 0
    # ============================================================

    def xi_constraint(theta):

        xi = theta[4:]

        return xi


    # ============================================================
    # 6. 制約
    # ============================================================

    constraints = [
        {
            "type": "ineq",
            "fun": margin_constraint
        },
        {
            "type": "ineq",
            "fun": xi_constraint
        }
    ]


    # ============================================================
    # 7. 初期値
    #
    # w1, w2, w3, b, xi1, ..., xiN
    # ============================================================

    theta0 = np.zeros(4 + N)


    # ============================================================
    # 8. 最適化
    # ============================================================

    result = minimize(
        objective,
        theta0,
        constraints=constraints,
        method="SLSQP"
    )


    # ============================================================
    # 9. 最適解
    # ============================================================

    w = result.x[:3]
    b = result.x[3]
    xi = result.x[4:]


    # ============================================================
    # 10. 各点の値
    # ============================================================

    decision_values = X @ w + b

    margin_values = y * decision_values


    # ============================================================
    # 11. 結果表示
    # ============================================================

    print("==========================================")
    print("3D Soft-margin SVM")
    print("==========================================")

    print()
    print("C =", C)

    print()
    print("----- 最適化結果 -----")
    print("success =", result.success)
    print("message =", result.message)

    print()
    print("----- 最適解 -----")

    print("w =", w)
    print("b =", b)

    print()
    print("xi =", xi)

    print()
    print("objective value =", result.fun)

    print()
    print("----- 各訓練データ -----")

    for i in range(N):

        print(
            f"i={i+1}, "
            f"x={X[i]}, "
            f"y={y[i]:.0f}, "
            f"f(x)={decision_values[i]:.6f}, "
            f"y*f(x)={margin_values[i]:.6f}, "
            f"xi={xi[i]:.6f}"
        )


    # ============================================================
    # 12. 3次元グラフ
    # ============================================================

    if plot:

        fig = plt.figure(figsize=(9, 8))

        ax = fig.add_subplot(
            111,
            projection="3d"
        )


        # --------------------------------------------------------
        # +1 クラス
        # --------------------------------------------------------

        ax.scatter(
            X[y == 1, 0],
            X[y == 1, 1],
            X[y == 1, 2],
            s=80,
            label="y = +1"
        )


        # --------------------------------------------------------
        # -1 クラス
        # --------------------------------------------------------

        ax.scatter(
            X[y == -1, 0],
            X[y == -1, 1],
            X[y == -1, 2],
            marker="x",
            s=80,
            label="y = -1"
        )


        # --------------------------------------------------------
        # マージン上・内側の点
        # --------------------------------------------------------

        support_mask = (
            margin_values <= 1.0 + 1.0e-6
        )

        ax.scatter(
            X[support_mask, 0],
            X[support_mask, 1],
            X[support_mask, 2],
            s=250,
            facecolors="none",
            edgecolors="black",
            linewidths=1.5,
            label="on / inside margin"
        )


        # ========================================================
        # 分類境界平面
        #
        # w1*x1 + w2*x2 + w3*x3 + b = c
        #
        # x3 =
        # -(w1*x1 + w2*x2 + b - c) / w3
        # ========================================================

        x1_min = X[:, 0].min()
        x1_max = X[:, 0].max()

        x2_min = X[:, 1].min()
        x2_max = X[:, 1].max()

        pad1 = 0.3 * max(
            1.0,
            x1_max - x1_min
        )

        pad2 = 0.3 * max(
            1.0,
            x2_max - x2_min
        )

        x1_grid = np.linspace(
            x1_min - pad1,
            x1_max + pad1,
            30
        )

        x2_grid = np.linspace(
            x2_min - pad2,
            x2_max + pad2,
            30
        )

        X1, X2 = np.meshgrid(
            x1_grid,
            x2_grid
        )


        # w3 が0に近いと x3について解けない
        if abs(w[2]) > 1.0e-12:

            # 分類境界
            X3_decision = -(
                w[0] * X1
                + w[1] * X2
                + b
            ) / w[2]

            # +1側マージン
            X3_margin_plus = -(
                w[0] * X1
                + w[1] * X2
                + b
                - 1.0
            ) / w[2]

            # -1側マージン
            X3_margin_minus = -(
                w[0] * X1
                + w[1] * X2
                + b
                + 1.0
            ) / w[2]


            ax.plot_surface(
                X1,
                X2,
                X3_decision,
                alpha=0.35
            )

            ax.plot_surface(
                X1,
                X2,
                X3_margin_plus,
                alpha=0.15
            )

            ax.plot_surface(
                X1,
                X2,
                X3_margin_minus,
                alpha=0.15
            )

        else:

            print()
            print(
                "w3 がほぼ0なので、"
                "x3について解く形では"
                "平面を描画できません。"
            )


        # --------------------------------------------------------
        # 軸
        # --------------------------------------------------------

        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("x3")

        ax.legend()

        plt.show()


    return result, w, b, xi
