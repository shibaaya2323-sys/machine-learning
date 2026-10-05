
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. Lorenz方程式
# ============================================================

def lorenz_rhs(x, sigma=10.0, rho=28.0, beta=8.0/3.0):

    X, Y, Z = x

    return np.array([
        sigma * (Y - X),
        X * (rho - Z) - Y,
        X * Y - beta * Z
    ], dtype=float)

# ============================================================
# 2. RK4
# ============================================================

def rk4_step(f, x, dt, *fargs, **fkwargs):

    k1 = f(x, *fargs, **fkwargs)
    k2 = f(x + 0.5 * dt * k1, *fargs, **fkwargs)
    k3 = f(x + 0.5 * dt * k2, *fargs, **fkwargs)
    k4 = f(x + dt * k3, *fargs, **fkwargs)

    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# ============================================================
# 3. Lorenz系の時間積分
# ============================================================

def integrate_lorenz(
    x0,
    dt,
    nsteps,
    sigma=10.0,
    rho=28.0,
    beta=8.0/3.0
):

    x0 = np.asarray(x0, dtype=float)

    t = np.empty(nsteps + 1, dtype=float)
    x = np.empty((nsteps + 1, 3), dtype=float)

    t[0] = 0.0
    x[0] = x0

    for n in range(nsteps):

        x[n + 1] = rk4_step(
            lorenz_rhs,
            x[n],
            dt,
            sigma,
            rho,
            beta
        )

        t[n + 1] = t[n] + dt

    return t, x

# ============================================================
# 4. 計算から過渡状態除去までをまとめた関数
# ============================================================

def run_lorenz(
    dt,
    total_time,
    transient_time,
    x0=np.array([1.0, 1.0, 1.0]),
    sigma=10.0,
    rho=28.0,
    beta=8.0/3.0,
    plot=True
):

    # --------------------------------------------------------
    # ステップ数
    # --------------------------------------------------------

    nsteps = int(round(total_time / dt))

    transient_steps = int(round(transient_time / dt))

    # --------------------------------------------------------
    # 入力チェック
    # --------------------------------------------------------

    if dt <= 0:
        raise ValueError("dt は正の値にしてください。")

    if total_time <= 0:
        raise ValueError("total_time は正の値にしてください。")

    if transient_time < 0:
        raise ValueError("transient_time は0以上にしてください。")

    if transient_time >= total_time:
        raise ValueError("transient_time は total_time より小さくしてください。")

    # --------------------------------------------------------
    # 計算条件
    # --------------------------------------------------------

    print("--- 計算条件 ---")
    print(f"時間刻み dt              = {dt}")
    print(f"計算時間 total_time      = {total_time}")
    print(f"過渡時間 transient_time  = {transient_time}")
    print(f"ステップ数 nsteps        = {nsteps:,}")
    print(f"過渡ステップ数           = {transient_steps:,}")

    # --------------------------------------------------------
    # Lorenz系を時間積分
    # --------------------------------------------------------

    t, x = integrate_lorenz(
        x0,
        dt=dt,
        nsteps=nsteps,
        sigma=sigma,
        rho=rho,
        beta=beta
    )

    # --------------------------------------------------------
    # 過渡状態を除去
    # --------------------------------------------------------

    t_attractor = t[transient_steps:]
    x_attractor = x[transient_steps:]

    # --------------------------------------------------------
    # Lorenzアトラクターを表示
    # --------------------------------------------------------

    if plot:

        fig = plt.figure(
            figsize=(8, 6)
        )

        ax = fig.add_subplot(
            111,
            projection="3d"
        )

        ax.plot(
            x_attractor[:, 0],
            x_attractor[:, 1],
            x_attractor[:, 2],
            linewidth=0.4
        )

        ax.set_title(
            "Lorenz Attractor (RK4)"
        )

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        ax.view_init(
            elev=25,
            azim=145
        )

        plt.tight_layout()
        plt.show()


    # --------------------------------------------------------
    # 結果を返す
    # --------------------------------------------------------

    return (
        t,
        x,
        t_attractor,
        x_attractor
    )
