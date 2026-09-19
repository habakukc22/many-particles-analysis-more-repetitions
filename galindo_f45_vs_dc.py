"""
f45 x DC — teste de efeitos de superficie no estilo da Fig. 6 de Galindo (2021).
Versao 2.

Rode depois da celula 15 do notebook (a que chama get_error_bar):

    from galindo_f45_vs_dc import run
    res = run(p_list)

Mudancas em relacao a versao 1:
  * DC calculado por step e so depois combinado entre steps, para que os steps
    com nrep alto nao dominem a media.
  * DC em contagens por bin (nao dividido pela largura do bin): com ~100 bins
    por periodo e tempo de integracao fixo, essa grandeza ja e proporcional ao
    brilho, independente da frequencia de modulacao.
  * Steps de frequencia muito baixa excluidos do DC: em 1-3 Hz dois ciclos
    duram mais que o tempo de integracao e o histograma sai incompleto.
  * Diagnostico do DC contra a frequencia: se o DC nao for plano, o eixo
    horizontal nao mede brilho.
  * Incertezas dos ajustes reescalonadas pelo chi2 quando a dispersao entre
    particulas domina (que e o caso).
  * Spearman como teste principal, por nao depender das barras de erro.
  * Tabela .csv por particula, para cruzar com os tamanhos do MEV depois.

IMPORTANTE: antes de rodar, corrija o pipeline em treament_tcspc_data.py:
    points_off = 0                                   # mantem 2 periodos inteiros
    sample_interval = max(time)/(len(curve)-1)/1e6   # em get_fft
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import odr
from scipy.stats import spearmanr

# ----------------------------------------------------------------------
# Configuracao
# ----------------------------------------------------------------------
PHI_TARGET = -np.pi / 4      # fase alvo que define f45
BACKGROUND = 0.0             # contagens de fundo por bin (medidas sem particula)
N_MC = 3000                  # sorteios Monte Carlo para a incerteza de f45
FREQ_MIN_DC = 5.0            # ignora steps abaixo disto no calculo do DC
FREQ_MAX_DC = np.inf
SKIP_PARTICLES = []
CSV_OUT = "galindo_table.csv"
RNG = np.random.default_rng(0)


# ----------------------------------------------------------------------
# 1. DC
# ----------------------------------------------------------------------
def particle_dc_by_step(p, background=BACKGROUND):
    """Retorna (freqs, dc_por_step): media de contagens por bin em cada step."""
    freqs, dcs = [], []
    for step in p["p_data"]:
        reps = [np.asarray(m[:, 1], dtype=float).mean() - background
                for m in step["data"]]
        if not reps:
            continue
        freqs.append(step["freq"])
        dcs.append(np.mean(reps))
    order = np.argsort(freqs)
    return np.asarray(freqs)[order], np.asarray(dcs)[order]


def particle_dc(p):
    """DC combinado da particula: mediana entre steps, robusta a outliers.

    O erro e o desvio padrao da media entre steps. Se a dispersao relativa for
    grande, o DC depende da frequencia e o eixo horizontal nao e confiavel.
    """
    freqs, dcs = particle_dc_by_step(p)
    m = (freqs >= FREQ_MIN_DC) & (freqs <= FREQ_MAX_DC)
    vals = dcs[m]
    if vals.size == 0:
        return np.nan, np.nan, np.nan
    dc = np.median(vals)
    err = vals.std(ddof=1) / np.sqrt(vals.size) if vals.size > 1 else 0.0
    spread = vals.std(ddof=1) / abs(dc) if vals.size > 1 and dc != 0 else np.nan
    return dc, err, spread


# ----------------------------------------------------------------------
# 2. f45, sem modelo
# ----------------------------------------------------------------------
def _cross_f45(freq, phase, phi_target=PHI_TARGET):
    i_min = int(np.argmin(phase))
    f, ph = freq[: i_min + 1], phase[: i_min + 1]
    below = np.where(ph <= phi_target)[0]
    if below.size == 0 or below[0] == 0:
        return np.nan
    j = below[0]
    x0, x1 = np.log10(f[j - 1]), np.log10(f[j])
    y0, y1 = ph[j - 1], ph[j]
    if y1 == y0:
        return np.nan
    return 10.0 ** (x0 + (phi_target - y0) * (x1 - x0) / (y1 - y0))


def particle_f45(p, n_mc=N_MC):
    arr = np.asarray(p["error_bar"], dtype=float)
    o = np.argsort(arr[:, 0])
    freq, phase, sigma = arr[o, 0], arr[o, 1], arr[o, 2]

    f45 = _cross_f45(freq, phase)
    if not np.isfinite(f45):
        return np.nan, np.nan
    draws = np.array([_cross_f45(freq, phase + RNG.normal(0.0, sigma))
                      for _ in range(n_mc)])
    draws = draws[np.isfinite(draws)]
    err = draws.std(ddof=1) if draws.size > 10 else np.nan
    return f45, err


def particle_shape(p):
    """Duas grandezas extra sem modelo: minimo da fase e onde ele ocorre."""
    arr = np.asarray(p["error_bar"], dtype=float)
    i = int(np.argmin(arr[:, 1]))
    return arr[i, 1], arr[i, 0]


# ----------------------------------------------------------------------
# 3. Coleta
# ----------------------------------------------------------------------
def collect(p_list, skip=SKIP_PARTICLES):
    rows = []
    for p in p_list:
        if p["p_label"] in skip:
            continue
        f, fe = particle_f45(p)
        if not np.isfinite(f):
            print(f"Particula {p['p_label']}: sem cruzamento de -pi/4, descartada.")
            continue
        d, de, spread = particle_dc(p)
        if not np.isfinite(d) or d <= 0:
            print(f"Particula {p['p_label']}: DC invalido ({d}), descartada.")
            continue
        phmin, fmin = particle_shape(p)
        rows.append((p["p_label"], d, de, spread, f, fe, phmin, fmin))
    cols = list(zip(*rows))
    keys = ["label", "dc", "dc_e", "dc_spread", "f45", "f45_e", "phi_min", "f_min"]
    return {k: np.asarray(v, dtype=float) for k, v in zip(keys, cols)}


# ----------------------------------------------------------------------
# 4. Ajustes
# ----------------------------------------------------------------------
def fit_powerlaw(dc, dc_e, k, k_e):
    x, y = np.log10(dc), np.log10(k)
    sx = np.maximum(dc_e / (dc * np.log(10)), 1e-9)
    sy = np.maximum(k_e / (k * np.log(10)), 1e-9)
    model = odr.Model(lambda B, x: B[0] + B[1] * x)
    return odr.ODR(odr.RealData(x, y, sx=sx, sy=sy), model,
                   beta0=[np.median(y), -1 / 3]).run()


def fit_bulk_plus_surface(dc, dc_e, k, k_e, expo=-1 / 3):
    """Minimos quadrados comuns (fit_type=2): nao desloca o eixo x, o que
    evita valores negativos de I dentro da potencia fracionaria."""
    model = odr.Model(lambda B, x: B[0] + B[1] * np.abs(x) ** expo)
    data = odr.RealData(dc, k, sy=np.maximum(k_e, 1e-9))
    # chute inicial pela reta em I^expo
    A0, k0 = np.polyfit(dc ** expo, k, 1)
    job = odr.ODR(data, model, beta0=[k0, A0])
    job.set_job(fit_type=2)
    return job.run()


# ----------------------------------------------------------------------
# 5. Execucao
# ----------------------------------------------------------------------
def run(p_list, save_csv=CSV_OUT):
    r = collect(p_list)
    dc, dc_e, f45, f45_e = r["dc"], r["dc_e"], r["f45"], r["f45_e"]
    k, k_e = 2 * np.pi * f45, 2 * np.pi * f45_e
    n = len(r["label"])

    print(f"\n{n} particulas aproveitadas.")
    print(f"DC:  {dc.min():.4g} a {dc.max():.4g} "
          f"({np.log10(dc.max()/dc.min()):.2f} decadas)")
    print(f"f45: {f45.min():.4g} a {f45.max():.4g} Hz "
          f"(dispersao {100*f45.std(ddof=1)/f45.mean():.0f}%)")
    bad = r["label"][r["dc_spread"] > 0.2]
    if bad.size:
        print(f"ATENCAO: DC varia mais de 20% entre steps nas particulas "
              f"{bad.astype(int)}. Veja o grafico de diagnostico.")

    # -- teste principal, sem modelo --
    rho, pval = spearmanr(dc, f45)
    print("\n--- Spearman (teste principal) ---")
    print(f"rho = {rho:.3f}, p = {pval:.3f}")
    sx, sy = np.log10(dc).std(ddof=1), np.log10(f45).std(ddof=1)
    rho_lim = np.tanh(1.96 / np.sqrt(n - 3))
    print(f"sensibilidade: so detectaria |inclinacao| > {rho_lim*sy/sx:.2f} "
          f"(a superficie preveria 0.33)")

    if n >= 4:
        fit1 = fit_powerlaw(dc, dc_e, k, k_e)
        s1 = max(1.0, np.sqrt(fit1.res_var))
        print(f"\n--- Ajuste 1: lei de potencia (erros x{s1:.0f} pelo chi2) ---")
        print(f"b = {fit1.beta[1]:.3f} +/- {fit1.sd_beta[1]*s1:.3f}   "
              "(esperado -0.333)")

        fit2 = fit_bulk_plus_surface(dc, dc_e, k, k_e)
        s2 = max(1.0, np.sqrt(fit2.res_var))
        print("\n--- Ajuste 2: k = k_bulk + A*I^(-1/3) ---")
        print(f"k_bulk = {fit2.beta[0]:.4g} +/- {fit2.sd_beta[0]*s2:.4g} s^-1")
        print(f"A      = {fit2.beta[1]:.4g}"
              + ("   (NEGATIVO: o modelo nao se aplica)"
                 if fit2.beta[1] < 0 else ""))
    else:
        fit1 = fit2 = None

    _plot(r, k, k_e, fit1, fit2, rho, pval)
    _plot_dc_diagnostic(p_list, r["label"])

    if save_csv:
        hdr = "label,dc,dc_e,dc_spread,f45,f45_e,phi_min,f_min"
        np.savetxt(save_csv, np.column_stack([r[c] for c in hdr.split(",")]),
                   delimiter=",", header=hdr, comments="", fmt="%.6g")
        print(f"\nTabela salva em {save_csv} "
              "(acrescente a coluna de raio quando tiver o MEV).")

    r.update(k=k, k_e=k_e, fit_powerlaw=fit1, fit_bulk=fit2,
             spearman=(rho, pval))
    return r


def _plot(r, k, k_e, fit1, fit2, rho, pval):
    dc, dc_e = r["dc"], r["dc_e"]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(dc, k, xerr=dc_e, yerr=k_e, fmt="o", capsize=4, color="tab:blue")
    for lab, x, y in zip(r["label"], dc, k):
        ax.annotate(str(int(lab)), (x, y), fontsize=7,
                    textcoords="offset points", xytext=(4, 4))

    xs = np.logspace(np.log10(dc.min()) - 0.1, np.log10(dc.max()) + 0.1, 200)
    if fit1 is not None:
        ax.plot(xs, 10 ** (fit1.beta[0] + fit1.beta[1] * np.log10(xs)), "r-",
                label=f"lei de potencia, b = {fit1.beta[1]:.2f}")
    if fit2 is not None and fit2.beta[1] > 0:
        ax.plot(xs, fit2.beta[0] + fit2.beta[1] * xs ** (-1 / 3), "g--",
                label="k_bulk + A I^(-1/3)")
    ax.plot(xs, np.median(k) * (xs / np.median(dc)) ** (-1 / 3), "k",
            lw=0.8, alpha=0.5, label="inclinacao -1/3 (referencia)")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Termo DC da luminescencia (contagens/bin)")
    ax.set_ylabel(r"$k_{45} = 2\pi f_{45}$  (s$^{-1}$)")
    ax.set_title(f"Teste de superficie — Spearman rho={rho:.2f}, p={pval:.2f}")
    ax.legend(fontsize=8)
    fig.tight_layout(); plt.show()


def _plot_dc_diagnostic(p_list, labels):
    """DC normalizado contra a frequencia. Deveria ser plano."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for p in p_list:
        if p["p_label"] not in labels:
            continue
        f, d = particle_dc_by_step(p)
        m = (f >= FREQ_MIN_DC) & (f <= FREQ_MAX_DC)
        if m.sum() == 0 or np.median(d[m]) == 0:
            continue
        ax.plot(f[m], d[m] / np.median(d[m]), "-o", ms=3, alpha=0.6, lw=0.8)
    ax.axhline(1.0, color="k", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("Frequencia do step (Hz)")
    ax.set_ylabel("DC / mediana da particula")
    ax.set_title("Diagnostico: o DC deveria ser plano em frequencia")
    fig.tight_layout(); plt.show()
