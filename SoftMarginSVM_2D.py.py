
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def solve_soft_margin_svm(X, y, C=1.0, plot=True):
    """
    2次元ソフトマージンSVMを scipy.optimize.minimize で解く関数

    Parameters
    ----------
    X : ndarray, shape (N, 2)
        訓練データ

    y : ndarray, shape (N,)
        正解ラベル
        +1 または -1

    C : float
        ソフトマージンのパラメータ

    plot : bool
        Trueなら分類境界とマージンを描画

    Returns
    -------
    result : scipy OptimizeResult
        最適化結果

    w : ndarray, shape (2,)
        最適な重みベクトル

    b : float
        バイアス

    xi : ndarray, shape (N,)
        スラック変数
    """

    # ============================================================
    # 1. 入力をnumpy配列に変換
    # ============================================================

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # データ数
    N = len(X)


    # ============================================================
    # 2. 入力チェック
    # ============================================================

    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("X は shape (N, 2) の2次元データにしてください。")

    if y.ndim != 1:
        raise ValueError("y は1次元配列にしてください。")

    if len(y) != N:
        raise ValueError("X のデータ数と y のラベル数が一致していません。")

    if not np.all(np.isin(y, [-1.0, 1.0])):
        raise ValueError("ラベル y は +1 または -1 にしてください。")

    if C <= 0:
        raise ValueError("C は正の値にしてください。")


    # ============================================================
    # 3. 目的関数
    #
    # theta =
    # [w1, w2, b, xi1, xi2, ..., xiN]
    #
    # min  1/2 ||w||^2 + C Σxi
    # ============================================================

    def objective(theta):

        w = theta[:2]
        xi = theta[3:]

        return (0.5 * np.dot(w, w) + C * np.sum(xi))


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

        w = theta[:2]
        b = theta[2]
        xi = theta[3:]

        return y * (X @ w + b) - 1.0 + xi


    # ============================================================
    # 5. xi_i >= 0
    # ============================================================

    def xi_constraint(theta):

        xi = theta[3:]

        return xi


    # ============================================================
    # 6. 制約をまとめる
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
    # w1, w2, b, xi1, ..., xiN
    # ============================================================

    theta0 = np.zeros(3 + N)


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
    # 9. 最適解を取り出す
    # ============================================================

    w = result.x[:2]
    b = result.x[2]
    xi = result.x[3:]


    # ============================================================
    # 10. 各点について計算
    # ============================================================

    decision_values = X @ w + b

    margin_values = y * decision_values


    # ============================================================
    # 11. 結果表示
    # ============================================================

    print("==========================================")
    print("Soft-margin SVM")
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
    # 12. グラフ
    # ============================================================

    if plot:

        # x1 の表示範囲をデータから自動設定
        x1_min = X[:, 0].min()
        x1_max = X[:, 0].max()

        padding_x = 0.5 * max(1.0, x1_max - x1_min)

        x1_plot = np.linspace(
            x1_min - padding_x,
            x1_max + padding_x,
            300
        )


        plt.figure(figsize=(8, 7))


        # --------------------------------------------------------
        # +1 クラス
        # --------------------------------------------------------

        plt.scatter(
            X[y == 1, 0],
            X[y == 1, 1],
            s=90,
            label="y = +1"
        )


        # --------------------------------------------------------
        # -1 クラス
        # --------------------------------------------------------

        plt.scatter(
            X[y == -1, 0],
            X[y == -1, 1],
            marker="x",
            s=90,
            label="y = -1"
        )


        # --------------------------------------------------------
        # 分類境界とマージン
        #
        # w1*x1 + w2*x2 + b = c
        # --------------------------------------------------------

        # w2 がほぼ0でなければ x2 = ... の形で描画
        if abs(w[1]) > 1.0e-12:

            x2_decision = -(w[0] * x1_plot + b) / w[1]

            x2_margin_plus = -(w[0] * x1_plot + b - 1.0) / w[1]

            x2_margin_minus = -(w[0] * x1_plot + b + 1.0) / w[1]


            plt.plot(
                x1_plot,
                x2_decision,
                linewidth=2,
                label="decision boundary"
            )

            plt.plot(
                x1_plot,
                x2_margin_plus,
                linestyle="--",
                label="margin +1"
            )

            plt.plot(
                x1_plot,
                x2_margin_minus,
                linestyle="--",
                label="margin -1"
            )


        # w2 ≈ 0 なら分類境界は縦線になる
        else:

            x_decision = -b / w[0]

            x_margin_plus = (1.0 - b) / w[0]

            x_margin_minus = (-1.0 - b) / w[0]

            plt.axvline(
                x_decision,
                linewidth=2,
                label="decision boundary"
            )

            plt.axvline(
                x_margin_plus,
                linestyle="--",
                label="margin +1"
            )

            plt.axvline(
                x_margin_minus,
                linestyle="--",
                label="margin -1"
            )


        # --------------------------------------------------------
        # マージン上・マージン内の点
        #
        # y_i f(x_i) <= 1
        # --------------------------------------------------------

        support_mask = margin_values <= 1.0 + 1.0e-6

        plt.scatter(
            X[support_mask, 0],
            X[support_mask, 1],
            s=300,
            facecolors="none",
            edgecolors="black",
            linewidths=1.5,
            label="on / inside margin"
        )


        # --------------------------------------------------------
        # xi と yf の表示
        # --------------------------------------------------------

        for i in range(N):

            plt.annotate(
                (
                    f"x{i+1}\n"
                    f"xi={xi[i]:.2f}\n"
                    f"yf={margin_values[i]:.2f}"
                ),
                (X[i, 0], X[i, 1]),
                textcoords="offset points",
                xytext=(8, 8)
            )


        # --------------------------------------------------------
        # グラフ範囲
        # --------------------------------------------------------

        x2_min = X[:, 1].min()
        x2_max = X[:, 1].max()

        padding_y = 0.5 * max(1.0, x2_max - x2_min)

        plt.xlim(
            x1_min - padding_x,
            x1_max + padding_x
        )

        plt.ylim(
            x2_min - padding_y,
            x2_max + padding_y
        )

        plt.xlabel("x1")
        plt.ylabel("x2")

        plt.grid()
        plt.legend()

        plt.show()


    return result, w, b, xi
