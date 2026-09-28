"""Regenera las cinco figuras del informe.

Corre con el venv del proyecto:

    ./venv/bin/python export_figures.py

No dibuja nada dentro del cuaderno: el cuaderno calcula y muestra, este
script escribe en assets/images/. Todo lo que hay aqui sale del mismo
marco temporal y del mismo corte que el cuaderno, 1880-01 a 2025-11 con
prueba desde 2001-01.
"""

import collections
import warnings
from pathlib import Path

import gymnasium as gym
import kagglehub
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import find_peaks
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("assets/images")
OUT.mkdir(parents=True, exist_ok=True)

CUTOFF = "2001-01-01"
FIT_END = "1990-12-01"
WINDOW = 24


def load_frame():
    sun_dir = Path(
        kagglehub.dataset_download("gallo33henrique/sunspot-numbers-dataset-17002025")
    )
    temp_dir = Path(
        kagglehub.dataset_download(
            "adarshsalukhe/global-land-temperature-anomalies-1880-2024"
        )
    )
    sun = pd.read_csv(
        sun_dir / "SN_m_tot_V2.0.csv",
        sep=";",
        header=None,
        names=["year", "month", "decimal_date", "sunspots", "sunspots_std", "n_obs", "flag"],
    )
    temp = pd.read_csv(next(temp_dir.glob("Global_Land_Temperature_Anomalies_*.csv")))
    sun["date"] = pd.to_datetime(dict(year=sun.year, month=sun.month, day=1))
    sun = sun.set_index("date").sort_index()
    year = temp["time"].astype(int)
    month = ((temp["time"] - year) * 12).astype(int) + 1
    temp["date"] = pd.to_datetime(dict(year=year, month=month.clip(1, 12), day=1))
    temp = temp.set_index("date").sort_index()
    frame = temp.join(sun[["sunspots"]], how="inner").loc["1880":"2025-11"]
    return frame[["land", "sunspots"]]


def fig_nivel(frame):
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    frame["land"].plot(ax=axes[0], color="tab:blue", linewidth=0.8)
    axes[0].set_title("Anomalia terrestre land (Berkeley Earth)")
    axes[0].set_ylabel("°C sobre la referencia")
    axes[0].axhline(0, color="black", linewidth=0.5)
    frame["sunspots"].plot(ax=axes[1], color="tab:orange", linewidth=0.6)
    axes[1].set_title("Manchas solares mensuales (SILSO v2.0)")
    axes[1].set_ylabel("manchas")
    axes[1].set_xlabel("año")
    fig.tight_layout()
    fig.savefig(OUT / "fig_nivel.png", dpi=150)
    plt.close(fig)


MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
         "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]


def fig_descomposicion(frame):
    land = frame["land"]
    result = seasonal_decompose(land, model="additive", period=12, extrapolate_trend="freq")
    trend = pd.Series(result.trend, index=land.index)
    # Mismo residuo que el cuaderno: la serie menos su tendencia, agrupado por mes.
    # Un solo enero para todo el periodo, no uno por ano.
    detrended = land - trend
    mean_by_month = detrended.groupby(land.index.month).mean()
    median_by_month = detrended.groupby(land.index.month).median()
    low_by_month = detrended.groupby(land.index.month).quantile(0.25)
    high_by_month = detrended.groupby(land.index.month).quantile(0.75)

    first, last = land.index[0].year, land.index[-1].year
    fig, axes = plt.subplots(3, 1, figsize=(11, 8))

    land.plot(ax=axes[0], color="tab:blue", linewidth=0.8)
    axes[0].set_title(f"Anomalía terrestre land, {first}-{last}")
    axes[0].set_ylabel("observada (°C)")
    axes[0].set_xlabel("año")

    trend.plot(ax=axes[1], color="tab:red", linewidth=1.4)
    axes[1].set_title("Tendencia: promedio móvil centrado de 12 meses")
    axes[1].set_ylabel("tendencia (°C)")
    axes[1].set_xlabel("año")

    months = np.arange(1, 13)
    axes[2].fill_between(months, low_by_month.values, high_by_month.values,
                         color="tab:green", alpha=0.2,
                         label=f"rango intercuartílico entre los {last - first + 1} años")
    axes[2].plot(months, mean_by_month.values, color="tab:green",
                 marker="o", linewidth=1.6, label="media del mes")
    axes[2].plot(months, median_by_month.values, color="tab:purple",
                 marker="s", linestyle="--", linewidth=1.2, label="mediana del mes")
    axes[2].axhline(0, color="black", linewidth=0.6)
    axes[2].set_xticks(months)
    axes[2].set_xticklabels(MESES)
    axes[2].set_title("Perfil estacional: los doce meses del residuo, no 146 años superpuestos")
    axes[2].set_ylabel("componente (°C)")
    axes[2].set_xlabel("mes")
    axes[2].legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "fig_descomposicion.png", dpi=150)
    plt.close(fig)

    print(f"perfil estacional {first}-{last} (°C), media | mediana:")
    for month in months:
        print(f"  {MESES[month - 1]}  {mean_by_month[month]:+.4f}  {median_by_month[month]:+.4f}")
    print(f"  max {MESES[mean_by_month.idxmax() - 1]} {mean_by_month.max():+.4f}"
          f"   min {MESES[mean_by_month.idxmin() - 1]} {mean_by_month.min():+.4f}"
          f"   rango {mean_by_month.max() - mean_by_month.min():.4f}")


def fig_maximos(frame):
    smoothed = frame["sunspots"].rolling(13, center=True).mean()
    peaks, _ = find_peaks(smoothed, distance=110)
    maxima = smoothed.index[peaks]
    intervals = maxima.to_series().diff().dt.days.dropna() / 30.44
    fig, ax = plt.subplots(figsize=(11, 4))
    frame["sunspots"].plot(ax=ax, alpha=0.4, label="mensual")
    smoothed.plot(ax=ax, label="suavizada 13 meses")
    ax.scatter(maxima, smoothed.loc[maxima], color="tab:red", zorder=5, label="máximos")
    ax.set_title(f"{len(maxima)} máximos, intervalo medio {intervals.mean():.0f} meses")
    ax.set_ylabel("manchas")
    ax.set_xlabel("año")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_maximos_solares.png", dpi=150)
    plt.close(fig)


class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU()
        )
        self.head = nn.Linear(128, 5)

    def forward(self, batch):
        return self.head(self.body(batch))


def fig_pronosticos(frame):
    train = frame.loc[:"2000-12"]
    test = frame.loc[CUTOFF:]
    best_order = (2, 1, 1)
    arima = SARIMAX(train["land"], order=best_order).fit(disp=False)
    arimax = SARIMAX(
        train["land"], exog=train[["sunspots"]], order=best_order
    ).fit(disp=False)
    arima_fc = arima.get_forecast(steps=len(test)).predicted_mean
    arimax_fc = arimax.get_forecast(steps=len(test), exog=test[["sunspots"]]).predicted_mean
    arima_fc.index = test.index
    arimax_fc.index = test.index
    persistence = frame["land"].shift(1).loc[test.index]
    slope = (train["land"].iloc[-1] - train["land"].iloc[0]) / (len(train) - 1)
    drift = pd.Series(
        train["land"].iloc[-1] + slope * np.arange(1, len(test) + 1), index=test.index
    )

    fig, ax = plt.subplots(figsize=(11, 4))
    test["land"].plot(ax=ax, label="observada", color="black", linewidth=1.2)
    persistence.plot(ax=ax, label="persistencia", linewidth=1.2)
    arima_fc.plot(ax=ax, label="ARIMA(2,1,1)")
    arimax_fc.plot(ax=ax, label="ARIMAX con sol")
    drift.plot(ax=ax, label="drift", linewidth=1.0, linestyle="--")
    ax.set_title("Sobre la prueba 2001 en adelante: la persistencia gana")
    ax.set_ylabel("°C sobre la referencia")
    ax.set_xlabel("año")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_pronosticos.png", dpi=150)
    plt.close(fig)

    table = pd.DataFrame(
        {
            "rmse": [
                mean_squared_error(test["land"], persistence) ** 0.5,
                mean_squared_error(test["land"], drift) ** 0.5,
                mean_squared_error(test["land"], arima_fc) ** 0.5,
                mean_squared_error(test["land"], arimax_fc) ** 0.5,
            ],
            "mae": [
                mean_absolute_error(test["land"], persistence),
                mean_absolute_error(test["land"], drift),
                mean_absolute_error(test["land"], arima_fc),
                mean_absolute_error(test["land"], arimax_fc),
            ],
        },
        index=["persistencia", "drift", "arima", "arimax_sol"],
    )
    print(table.round(4))


def fig_curva_rl():
    torques = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    train_seed = 11
    torch.manual_seed(train_seed)
    np.random.seed(train_seed)
    policy = QNet()
    target = QNet()
    target.load_state_dict(policy.state_dict())
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    memory = collections.deque(maxlen=20000)
    env = gym.make("Pendulum-v1")
    epsilon, gamma = 1.0, 0.99
    returns = []
    for episode in range(300):
        observation, _ = env.reset(seed=train_seed + episode)
        total = 0.0
        for _ in range(200):
            if np.random.rand() < epsilon:
                choice = np.random.randint(len(torques))
            else:
                with torch.no_grad():
                    choice = int(policy(torch.Tensor(observation)).argmax())
            step = env.step(np.array([torques[choice]], dtype=np.float32))
            next_observation, reward, terminated, truncated, _ = step
            memory.append(
                (observation, choice, reward, next_observation, terminated or truncated)
            )
            observation = next_observation
            total += reward
            if len(memory) >= 512:
                batch = [memory[i] for i in np.random.choice(len(memory), 64, replace=False)]
                states = torch.Tensor(np.array([r[0] for r in batch]))
                actions = torch.LongTensor([r[1] for r in batch])
                rewards = torch.Tensor([r[2] for r in batch])
                next_states = torch.Tensor(np.array([r[3] for r in batch]))
                dones = torch.Tensor([r[4] for r in batch])
                with torch.no_grad():
                    upcoming = rewards + gamma * target(next_states).max(dim=1).values * (1 - dones)
                guesses = policy(states)[range(64), actions]
                optimizer.zero_grad()
                loss_fn(guesses, upcoming).backward()
                optimizer.step()
        if episode % 25 == 0:
            target.load_state_dict(policy.state_dict())
        epsilon = max(0.05, epsilon * 0.985)
        returns.append(total)
    env.close()

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(pd.Series(returns).rolling(20).mean())
    ax.set_title("Retorno por episodio (media móvil de 20)")
    ax.set_ylabel("retorno")
    ax.set_xlabel("episodio")
    fig.tight_layout()
    fig.savefig(OUT / "fig_curva_rl.png", dpi=150)
    plt.close(fig)
    print("DQN últimos 20 episodios:", round(pd.Series(returns).iloc[-20:].mean(), 1))


if __name__ == "__main__":
    frame = load_frame()
    fig_nivel(frame)
    fig_descomposicion(frame)
    fig_maximos(frame)
    fig_pronosticos(frame)
    fig_curva_rl()
    print("figuras escritas en", OUT)
