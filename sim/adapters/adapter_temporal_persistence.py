import os

import numpy as np


def run_sim(config, outdir=None):
    """
    Run a simple temporal persistence simulation.

    config: dict with keys: epsilon, eta, rho, lambda_init, timesteps, seed

    Returns dict with keys:
      fid_curve: list of F(t)
      cci_trace: list of CCI(t) (cumulative change index, synthetic)
      hazard_trace: list of hazard rates
      lambda_est_trace: estimated lambda over time
      t_arrow_var: variance of t_arrow (first-passage times) across reps
      tau_c: characteristic time from fit
      fit_R2: R^2 of fit
    """
    # Make deterministic
    seed = int(config.get("seed", 0)) + int(config.get("determinism_base_seed", 0))
    rng = np.random.RandomState(seed)

    epsilon = float(config.get("epsilon", config.get("eps", 0.01)))
    eta = float(config.get("noise_eta", config.get("eta", 0.0)))
    rho = float(config.get("rho", 0.5))
    lambda_init = float(config.get("lambda_init", 0.9))
    timesteps = int(config.get("epochs", config.get("timesteps", 300)))
    n_reps = int(config.get("n_reps", 200))
    logging_cfg = config.get("logging", {})
    save_curves = logging_cfg.get("save_curves", False)
    save_traces = logging_cfg.get("save_traces", False)

    # We'll model F(t) as mean survival across reps of an AR(1)-like survival variable
    # x_{t+1} = lambda * x_t - epsilon * x_t + noise, where lambda drifts with rho
    # For simplicity, create n_reps trajectories and compute mean persistence F(t)=mean(x>threshold)
    threshold = config.get("threshold", 0.5)

    all_F = np.zeros((n_reps, timesteps))
    all_lambda_est = np.zeros((n_reps, timesteps))
    all_t_first_cross = np.full(n_reps, timesteps)

    for rep in range(n_reps):
        x = lambda_init
        lam = lambda_init
        for t in range(timesteps):
            noise = rng.normal(scale=eta)
            # lambda relaxes toward 1 with strength rho
            lam = lam * (1 - rho * 0.01) + (1 - rho * 0.01) * 1.0
            x = lam * x - epsilon * x + noise
            x = max(0.0, min(1.0, x))
            all_F[rep, t] = 1.0 if x > threshold else 0.0
            all_lambda_est[rep, t] = lam
            if all_t_first_cross[rep] == timesteps and x <= threshold:
                all_t_first_cross[rep] = t

    fid_curve = list(all_F.mean(axis=0))
    cci_trace = list(np.cumsum(np.abs(np.diff(all_F.mean(axis=0), prepend=0))))
    # hazard: probability density of first-crossing at t / survival at t
    first_cross_counts = np.bincount(all_t_first_cross, minlength=timesteps)
    surv = np.cumsum(first_cross_counts[::-1])[::-1]
    surv = np.maximum(surv, 1)
    hazard = first_cross_counts / surv
    hazard_trace = list(hazard)
    lambda_est_trace = list(all_lambda_est.mean(axis=0))

    t_arrow_var = float(np.var(all_t_first_cross))

    # Fit an exponential to fid_curve to estimate tau_c
    times = np.arange(timesteps)
    y = np.array(fid_curve)
    try:
        # avoid zeros for log fit
        y_clip = np.clip(y, 1e-6, 1.0)
        # fit log y = -t/tau + b
        A = np.vstack([-times, np.ones_like(times)]).T
        sol, residuals, rank, s = np.linalg.lstsq(A, np.log(y_clip), rcond=None)
        slope, intercept = sol
        tau_c = 1.0 / slope if slope != 0 else float("inf")
        # compute R2
        y_pred = np.exp(intercept + slope * times)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        fit_R2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    except Exception:
        tau_c = float("nan")
        fit_R2 = 0.0

    out = dict(
        fid_curve=fid_curve,
        cci_trace=cci_trace,
        hazard_trace=(
            hazard_trace.tolist()
            if isinstance(hazard_trace, np.ndarray)
            else hazard_trace
        ),
        lambda_est_trace=lambda_est_trace,
        t_arrow_var=t_arrow_var,
        tau_c=float(tau_c),
        fit_R2=float(fit_R2),
    )

    # If an outdir is provided, write a JSON file with config + result for downstream analysis
    if outdir:
        try:
            import json

            os.makedirs(outdir, exist_ok=True)
            fname = os.path.join(outdir, f'result_seed_{config.get("seed",0)}.json')
            with open(fname, "w") as fh:
                json.dump({"config": config, "result": out}, fh, indent=2)
            # optionally save curves/traces as PNGs/npz
            if save_curves:
                try:
                    import matplotlib.pyplot as plt

                    times = np.arange(len(fid_curve))
                    plt.figure(figsize=(6, 3))
                    plt.plot(times, fid_curve, label="F(t)")
                    plt.xlabel("t")
                    plt.ylabel("F(t)")
                    plt.title(f'Fidelity curve seed={config.get("seed",0)}')
                    plt.legend()
                    pngname = os.path.join(
                        outdir, f'curves_fidelity_seed{config.get("seed",0)}.png'
                    )
                    plt.savefig(pngname, dpi=150)
                    plt.close()
                except Exception:
                    pass
            if save_traces:
                try:
                    np.savez_compressed(
                        os.path.join(outdir, f'traces_seed_{config.get("seed",0)}.npz'),
                        fid_curve=np.array(fid_curve),
                        hazard_trace=np.array(hazard_trace),
                        lambda_est_trace=np.array(lambda_est_trace),
                    )
                except Exception:
                    pass
        except Exception:
            pass

    return out
