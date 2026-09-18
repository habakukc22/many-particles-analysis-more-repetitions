"""
f45 x DC — teste de efeitos de superficie no estilo da Fig. 6 de Galindo (2021).

Uso: rode este script (ou cole o conteudo numa celula) DEPOIS da celula 15 do
notebook, ou seja, depois de `get_error_bar(exc_data_npy, p_list)`.
Ele espera encontrar em memoria:
    p_list  -> lista de particulas, cada uma com:
                 p["p_label"]    rotulo
                 p["error_bar"]  array (n_freq, 3) = [freq, delta_phi, erro]
                 p["p_data"]     lista de steps, cada step com step["data"]
                                 = lista de medidas, cada medida array (n,2)
                                 com coluna 0 = tempo, coluna 1 = contagens

Saida: dois ajustes e um grafico bilog de k45 = 2*pi*f45 contra o termo DC.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import odr

# ----------------------------------------------------------------------
# Configuracao
# ----------------------------------------------------------------------
PHI_TARGET = -np.pi / 4      # fase alvo para definir f45
BACKGROUND = 0.0             # contagens de fundo por bin, medidas sem particula
N_MC = 2000                  # sorteios Monte Carlo para a incerteza de f45
DC_MODE = "rate"             # "rate": contagens/bin dividido pela largura do bin
                             # "counts": media de contagens por bin
FREQ_MAX_DC = None           # se quiser restringir o DC a freqs <= este valor
SKIP_PARTICLES = []          # rotulos a excluir
RNG = np.random.default_rng(0)


# ----------------------------------------------------------------------
# 1. Termo DC de cada particula
# ----------------------------------------------------------------------
def particle_dc(p, mode=DC_MODE, background=BACKGROUND, freq_max=FREQ_MAX_DC):
    """Media do sinal (bin zero da FFT) sobre repeticoes e frequencias.

    Retorna (dc, erro_dc). O erro e o desvio padrao da media entre steps, que
    tambem serve de diagnostico: o DC nao deveria depender da frequencia.
    """
    per_step = []
    for step in p["p_data"]:
        if freq_max is not None and step["freq"] > freq_max:
            continue
        for meas in step["data"]:
            t = np.asarray(meas[:, 0], dtype=float)
            counts = np.asarray(meas[:, 1], dtype=float) - background
            if mode == "rate":
                dt = np.median(np.diff(t))
                per_step.append(counts.mean() / dt)
            else:
                per_step.append(counts.mean())

    per_step = np.asarray(per_step)
    dc = per_step.mean()
    err = per_step.std(ddof=1) / np.sqrt(len(per_step)) if len(per_step) > 1 else 0.0
    return dc, err


# ----------------------------------------------------------------------
# 2. f45 de cada particula, sem modelo
# ----------------------------------------------------------------------
def _cross_f45(freq, phase, phi_target=PHI_TARGET):
    """Interpola em log10(f) o primeiro cruzamento de phi_target.

    So considera o trecho antes do minimo da fase, onde a curva e monotonica.
    Retorna np.nan se nao houver cruzamento.
    """
    i_min = int(np.argmin(phase))
    f = freq[: i_min + 1]
    ph = phase[: i_min + 1]

    below = np.where(ph <= phi_target)[0]
    if below.size == 0 or below[0] == 0:
        return np.nan
    j = below[0]

    x0, x1 = np.log10(f[j - 1]), np.log10(f[j])
    y0, y1 = ph[j - 1], ph[j]
    if y1 == y0:
        return np.nan
    x = x0 + (phi_target - y0) * (x1 - x0) / (y1 - y0)
    return 10.0 ** x


def particle_f45(p, n_mc=N_MC, phi_target=PHI_TARGET):
    """f45 e sua incerteza, propagada por Monte Carlo sobre as barras de erro."""
    arr = np.asarray(p["error_bar"], dtype=float)
    order = np.argsort(arr[:, 0])
    freq, phase, sigma = arr[order, 0], arr[order, 1], arr[order, 2]

    f45 = _cross_f45(freq, phase, phi_target)
    if not np.isfinite(f45):
        return np.nan, np.nan

    draws = np.empty(n_mc)
    for i in range(n_mc):
        draws[i] = _cross_f45(freq, phase + RNG.normal(0.0, sigma), phi_target)
    draws = draws[np.isfinite(draws)]
    err = draws.std(ddof=1) if draws.size > 10 else np.nan
    return f45, err


# ----------------------------------------------------------------------
# 3. Coleta
# ----------------------------------------------------------------------
def collect(p_list, skip=SKIP_PARTICLES):
    labels, dc, dc_e, f45, f45_e = [], [], [], [], []
    for p in p_list:
        if p["p_label"] in skip:
            continue
        f, fe = particle_f45(p)
        if not np.isfinite(f):
            print(f"Particula {p['p_label']}: sem cruzamento de -pi/4, descartada.")
            continue
        d, de = particle_dc(p)
        if not np.isfinite(d) or d <= 0:
            print(f"Particula {p['p_label']}: DC invalido ({d}), descartada.")
            continue
        labels.append(p["p_label"])
        f45.append(f)
        f45_e.append(fe if np.isfinite(fe) else 0.0)
        dc.append(d)
        dc_e.append(de)
    return (np.array(labels), np.array(dc), np.array(dc_e),
            np.array(f45), np.array(f45_e))


# ----------------------------------------------------------------------
# 4. Ajustes
# ----------------------------------------------------------------------
def fit_powerlaw(dc, dc_e, k, k_e):
    """Reta em log-log: log10(k) = a + b*log10(I). Devolve b (expoente)."""
    x, y = np.log10(dc), np.log10(k)
    sx = np.where(dc > 0, dc_e / (dc * np.log(10)), 0.0)
    sy = np.where(k > 0, k_e / (k * np.log(10)), 0.0)
    sx = np.where(sx > 0, sx, np.full_like(sx, 1e-6))
    sy = np.where(sy > 0, sy, np.full_like(sy, 1e-6))

    model = odr.Model(lambda B, x: B[0] + B[1] * x)
    data = odr.RealData(x, y, sx=sx, sy=sy)
    out = odr.ODR(data, model, beta0=[np.median(y), -1 / 3]).run()
    return out


def fit_bulk_plus_surface(dc, dc_e, k, k_e, expo=-1 / 3):
    """k = k_bulk + A * I**expo, com o expoente fixo em -1/3."""
    sx = np.where(dc_e > 0, dc_e, 1e-6)
    sy = np.where(k_e > 0, k_e, 1e-6)
    model = odr.Model(lambda B, x: B[0] + B[1] * x ** expo)
    data = odr.RealData(dc, k, sx=sx, sy=sy)
    A0 = (k.max() - k.min()) / (dc.min() ** expo - dc.max() ** expo + 1e-30)
    out = odr.ODR(data, model, beta0=[max(k.min() * 0.5, 0.0), abs(A0)]).run()
    return out


# ----------------------------------------------------------------------
# 5. Execucao
# ----------------------------------------------------------------------
def run(p_list):
    labels, dc, dc_e, f45, f45_e = collect(p_list)
    k = 2 * np.pi * f45
    k_e = 2 * np.pi * f45_e
    print(f"\n{len(labels)} particulas aproveitadas.")
    print(f"Faixa de DC: {dc.min():.3g} a {dc.max():.3g} "
          f"({np.log10(dc.max() / dc.min()):.2f} decadas)")
    print(f"Faixa de f45: {f45.min():.4g} Hz a {f45.max():.4g} Hz")

    if len(labels) < 4:
        print("Poucas particulas para ajustar.")
        return None

    fit1 = fit_powerlaw(dc, dc_e, k, k_e)
    b, sb = fit1.beta[1], fit1.sd_beta[1]
    print("\n--- Ajuste 1: lei de potencia pura ---")
    print(f"expoente b = {b:.3f} +/- {sb:.3f}   (esperado -0.333)")
    print(f"desvio em relacao a -1/3: {(b + 1/3) / sb:.1f} sigma"
          if sb > 0 else "")
    print(f"chi2 reduzido = {fit1.res_var:.2f}")

    fit2 = fit_bulk_plus_surface(dc, dc_e, k, k_e)
    kb, skb = fit2.beta[0], fit2.sd_beta[0]
    print("\n--- Ajuste 2: k = k_bulk + A*I^(-1/3) ---")
    print(f"k_bulk = {kb:.4g} +/- {skb:.4g} s^-1  "
          f"(tau_bulk = {1/kb*1e3:.3g} ms)" if kb > 0 else
          f"k_bulk = {kb:.4g} +/- {skb:.4g} s^-1")
    print(f"A = {fit2.beta[1]:.4g} +/- {fit2.sd_beta[1]:.4g}")
    print(f"chi2 reduzido = {fit2.res_var:.2f}")

    # grafico
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(dc, k, xerr=dc_e, yerr=k_e, fmt="o", capsize=4,
                color="tab:blue", label="particulas")
    for lab, x, y in zip(labels, dc, k):
        ax.annotate(str(lab), (x, y), fontsize=7,
                    textcoords="offset points", xytext=(4, 4))

    xs = np.logspace(np.log10(dc.min()) - 0.1, np.log10(dc.max()) + 0.1, 200)
    ax.plot(xs, 10 ** (fit1.beta[0] + fit1.beta[1] * np.log10(xs)),
            "r-", label=f"lei de potencia, b = {b:.2f} +/- {sb:.2f}")
    ax.plot(xs, fit2.beta[0] + fit2.beta[1] * xs ** (-1 / 3),
            "g--", label="k_bulk + A I^(-1/3)")
    if kb > 0:
        ax.axhline(kb, color="gray", ls=":", lw=1, label="k_bulk")

    ref = np.median(k) * (xs / np.median(dc)) ** (-1 / 3)
    ax.plot(xs, ref, color="k", lw=0.8, alpha=0.5, label="referencia -1/3")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Termo DC da luminescencia (u.a.)")
    ax.set_ylabel(r"$k_{45} = 2\pi f_{45}$  (s$^{-1}$)")
    ax.set_title("Teste de efeitos de superficie (estilo Galindo 2021)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    plt.show()

    return dict(labels=labels, dc=dc, dc_e=dc_e, f45=f45, f45_e=f45_e,
                fit_powerlaw=fit1, fit_bulk=fit2)


if __name__ == "__main__":
    try:
        res = run(p_list)  # noqa: F821  (vem do notebook)
    except NameError:
        print("Rode este script no mesmo kernel do notebook, depois da celula 15.")
