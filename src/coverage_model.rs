use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result};

use crate::types::SampleSummary;

const TARGET_COVERAGE: f64 = 0.95;
pub const DEFAULT_MODEL_CURVE_MAX_EFFORT_BP: f64 = 1.0e13;
const FIT_WEIGHT_EXPONENTS: [f64; 9] = [0.0, 1.0, -1.0, 1.3, -1.1, 1.5, -1.5, 3.0, -3.0];

#[derive(Clone, Debug)]
pub struct ModelPoint {
    pub reads: u64,
    pub bases: u64,
    pub portion: f64,
    pub redundancy: f64,
    pub sd: f64,
    pub q1: f64,
    pub median: f64,
    pub q3: f64,
    pub coverage: f64,
    pub q1_coverage: f64,
    pub median_coverage: f64,
    pub q3_coverage: f64,
    pub adjusted_effort: f64,
    pub fitted_coverage: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct ModelCurvePoint {
    pub adjusted_effort: f64,
    pub coverage: f64,
}

#[derive(Clone, Debug)]
pub struct CoverageModel {
    pub total_reads: usize,
    pub total_bases: usize,
    pub average_read_length: f64,
    pub coverage_factor: f64,
    pub kappa: f64,
    pub observed_coverage: f64,
    pub total_effort: f64,
    pub effort_at_target: Option<f64>,
    pub diversity: Option<f64>,
    pub model_r: Option<f64>,
    pub alpha: Option<f64>,
    pub beta: Option<f64>,
    pub target_coverage: f64,
    pub points: Vec<ModelPoint>,
    pub curve: Vec<ModelCurvePoint>,
    pub warning: Option<String>,
}

#[derive(Clone, Debug)]
struct FitDatum {
    x: f64,
    y: f64,
    sd: f64,
}

#[derive(Clone, Copy, Debug)]
struct FitResult {
    alpha: f64,
    beta: f64,
    objective: f64,
    unweighted_sse: f64,
    weight_exp: f64,
}

pub fn fit_from_summaries(
    summaries: &[SampleSummary],
    total_reads: usize,
    total_bases: usize,
) -> CoverageModel {
    let average_read_length = if total_reads == 0 {
        0.0
    } else {
        total_bases as f64 / total_reads as f64
    };

    let coverage_factor = long_read_coverage_factor();
    let kappa = summaries.last().map(|s| clamp01(s.mean)).unwrap_or(0.0);
    let observed_coverage = redundancy_to_coverage(kappa, coverage_factor);
    let total_effort = total_bases as f64;
    let mut points = build_points(
        summaries,
        total_reads,
        average_read_length,
        coverage_factor,
        observed_coverage,
    );

    let fit_data = points
        .iter()
        .filter_map(|point| {
            (point.coverage > 0.0 && point.coverage < 0.9 && point.adjusted_effort > 0.0).then_some(
                FitDatum {
                    x: point.adjusted_effort,
                    y: point.coverage,
                    sd: point.sd.max(1e-6),
                },
            )
        })
        .collect::<Vec<_>>();

    let mut warning = None;
    let fit = if fit_data.len() >= 3 {
        fit_gamma_model(&fit_data)
    } else {
        warning = Some("fewer than three empirical points in the model fitting range".into());
        None
    };

    let mut alpha = None;
    let mut beta = None;
    let mut model_r = None;
    let mut effort_at_target = None;
    let mut diversity = None;
    let mut curve = Vec::new();

    if let Some(fit) = fit {
        alpha = Some(fit.alpha);
        beta = Some(fit.beta);

        for point in &mut points {
            point.fitted_coverage = predict_gamma(fit.alpha, fit.beta, point.adjusted_effort);
        }

        let observed = fit_data.iter().map(|d| d.y).collect::<Vec<_>>();
        let fitted = fit_data
            .iter()
            .filter_map(|d| predict_gamma(fit.alpha, fit.beta, d.x))
            .collect::<Vec<_>>();
        if observed.len() == fitted.len() {
            model_r = pearson(&observed, &fitted);
        }

        effort_at_target = gamma_inverse_effort(fit.alpha, fit.beta, TARGET_COVERAGE);
        diversity = (fit.alpha > 1.0).then_some((fit.alpha - 1.0) / fit.beta);
        curve = build_model_curve(fit.alpha, fit.beta, &points, effort_at_target);

        if warning.is_none() && fit.unweighted_sse.is_finite() {
            warning = Some(format!(
                "best weighted fit used sd exponent {:.2}; objective {:.6}",
                fit.weight_exp, fit.objective
            ));
        }
    }

    CoverageModel {
        total_reads,
        total_bases,
        average_read_length,
        coverage_factor,
        kappa,
        observed_coverage,
        total_effort,
        effort_at_target,
        diversity,
        model_r,
        alpha,
        beta,
        target_coverage: TARGET_COVERAGE,
        points,
        curve,
        warning,
    }
}

pub fn write_model(path: &Path, model: &CoverageModel) -> Result<()> {
    let file =
        File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    let mut w = BufWriter::new(file);

    writeln!(w, "# @impl: FastCover coverage gamma model")?;
    writeln!(w, "# @version: {}", env!("CARGO_PKG_VERSION"))?;
    writeln!(w, "# @reads: {}", model.total_reads)?;
    writeln!(w, "# @bases: {}", model.total_bases)?;
    writeln!(
        w,
        "# @average_read_length: {}",
        fmt_float(model.average_read_length)
    )?;
    writeln!(
        w,
        "# @coverage_factor: {}",
        fmt_float(model.coverage_factor)
    )?;
    writeln!(w, "# @kappa: {}", fmt_float(model.kappa))?;
    writeln!(w, "# @C: {}", fmt_float(model.observed_coverage))?;
    writeln!(w, "# @LR: {}", fmt_float(model.total_effort))?;
    writeln!(w, "# @LRstar: {}", fmt_optional(model.effort_at_target))?;
    writeln!(w, "# @diversity: {}", fmt_optional(model.diversity))?;
    writeln!(w, "# @modelR: {}", fmt_optional(model.model_r))?;
    writeln!(w, "# @alpha: {}", fmt_optional(model.alpha))?;
    writeln!(w, "# @beta: {}", fmt_optional(model.beta))?;
    writeln!(
        w,
        "# @target_coverage: {}",
        fmt_float(model.target_coverage)
    )?;
    if let Some(warning) = &model.warning {
        writeln!(w, "# @note: {warning}")?;
    }
    writeln!(
        w,
        "kind\treads\tbases\tportion\tadjusted_effort_bp\tredundant_fraction\tsd\tq1\tmedian\tq3\tcoverage\tq1_coverage\tmedian_coverage\tq3_coverage\tfitted_coverage"
    )?;
    for point in &model.points {
        writeln!(
            w,
            "observed\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            point.reads,
            point.bases,
            fmt_float(point.portion),
            fmt_float(point.adjusted_effort),
            fmt_float(point.redundancy),
            fmt_float(point.sd),
            fmt_float(point.q1),
            fmt_float(point.median),
            fmt_float(point.q3),
            fmt_float(point.coverage),
            fmt_float(point.q1_coverage),
            fmt_float(point.median_coverage),
            fmt_float(point.q3_coverage),
            fmt_optional(point.fitted_coverage),
        )?;
    }
    for point in &model.curve {
        writeln!(
            w,
            "model\t.\t.\t.\t{}\t.\t.\t.\t.\t.\t{}\t.\t.\t.\t{}",
            fmt_float(point.adjusted_effort),
            fmt_float(point.coverage),
            fmt_float(point.coverage),
        )?;
    }

    Ok(())
}

fn build_points(
    summaries: &[SampleSummary],
    _total_reads: usize,
    _average_read_length: f64,
    coverage_factor: f64,
    observed_coverage: f64,
) -> Vec<ModelPoint> {
    let positive_bases = summaries
        .iter()
        .filter_map(|s| (s.bases > 0).then_some(s.bases as f64))
        .collect::<Vec<_>>();
    let max_log_xobs = positive_bases
        .iter()
        .copied()
        .map(f64::ln)
        .fold(f64::NEG_INFINITY, f64::max);
    let c_scale = observed_coverage.max(1e-12).powf(0.27);

    let mut pre_adjusted = Vec::with_capacity(summaries.len());
    let mut max_pre_adjusted = 0.0_f64;
    for summary in summaries {
        let value = if summary.bases == 0 || !max_log_xobs.is_finite() {
            0.0
        } else {
            (max_log_xobs + c_scale * ((summary.bases as f64).ln() - max_log_xobs)).exp()
        };
        max_pre_adjusted = max_pre_adjusted.max(value);
        pre_adjusted.push(value);
    }

    let scaling = if max_pre_adjusted > 0.0 {
        summaries
            .last()
            .map(|summary| summary.bases as f64)
            .unwrap_or(0.0)
            / max_pre_adjusted
    } else {
        0.0
    };

    summaries
        .iter()
        .zip(pre_adjusted)
        .map(|(summary, adjusted)| {
            let coverage = redundancy_to_coverage(summary.mean, coverage_factor);
            ModelPoint {
                reads: summary.reads,
                bases: summary.bases,
                portion: summary.portion,
                redundancy: clamp01(summary.mean),
                sd: summary.sd,
                q1: clamp01(summary.q1),
                median: clamp01(summary.median),
                q3: clamp01(summary.q3),
                coverage,
                q1_coverage: redundancy_to_coverage(summary.q1, coverage_factor),
                median_coverage: redundancy_to_coverage(summary.median, coverage_factor),
                q3_coverage: redundancy_to_coverage(summary.q3, coverage_factor),
                adjusted_effort: adjusted * scaling,
                fitted_coverage: None,
            }
        })
        .collect()
}

fn long_read_coverage_factor() -> f64 {
    // FastCover uses the long-read overlap ratio directly rather than a short-read overlap correction.
    // The observed redundant read fraction is therefore interpreted on the coverage scale.
    1.0
}

fn redundancy_to_coverage(redundancy: f64, coverage_factor: f64) -> f64 {
    clamp01(redundancy).powf(coverage_factor)
}

fn fit_gamma_model(data: &[FitDatum]) -> Option<FitResult> {
    let mut best = None;
    for weight_exp in FIT_WEIGHT_EXPONENTS {
        let Some(initial) = grid_search(data, weight_exp) else {
            continue;
        };
        let refined = refine_fit(data, initial.alpha, initial.beta, weight_exp);
        let candidate = FitResult {
            unweighted_sse: objective(data, refined.alpha, refined.beta, 0.0),
            ..refined
        };
        if best
            .as_ref()
            .map(|b: &FitResult| candidate.unweighted_sse < b.unweighted_sse)
            .unwrap_or(true)
        {
            best = Some(candidate);
        }
    }
    best
}

fn grid_search(data: &[FitDatum], weight_exp: f64) -> Option<FitResult> {
    let mut best = None;
    for alpha in log_grid(0.25, 250.0, 42) {
        for beta in log_grid(0.005, 25.0, 44) {
            let score = objective(data, alpha, beta, weight_exp);
            if !score.is_finite() {
                continue;
            }
            let candidate = FitResult {
                alpha,
                beta,
                objective: score,
                unweighted_sse: score,
                weight_exp,
            };
            if best
                .as_ref()
                .map(|b: &FitResult| candidate.objective < b.objective)
                .unwrap_or(true)
            {
                best = Some(candidate);
            }
        }
    }
    best
}

fn refine_fit(data: &[FitDatum], alpha: f64, beta: f64, weight_exp: f64) -> FitResult {
    let mut log_alpha = alpha.ln();
    let mut log_beta = beta.ln();
    let mut best_score = objective(data, alpha, beta, weight_exp);
    let mut step = 0.75_f64;

    while step > 1e-5 {
        let mut improved = false;
        let mut next_log_alpha = log_alpha;
        let mut next_log_beta = log_beta;
        let mut next_score = best_score;

        for da in [-step, 0.0, step] {
            for db in [-step, 0.0, step] {
                if da == 0.0 && db == 0.0 {
                    continue;
                }
                let a = (log_alpha + da).exp().clamp(0.05, 1000.0);
                let b = (log_beta + db).exp().clamp(1e-5, 1000.0);
                let score = objective(data, a, b, weight_exp);
                if score < next_score {
                    next_score = score;
                    next_log_alpha = a.ln();
                    next_log_beta = b.ln();
                    improved = true;
                }
            }
        }

        if improved {
            log_alpha = next_log_alpha;
            log_beta = next_log_beta;
            best_score = next_score;
        } else {
            step *= 0.5;
        }
    }

    FitResult {
        alpha: log_alpha.exp(),
        beta: log_beta.exp(),
        objective: best_score,
        unweighted_sse: best_score,
        weight_exp,
    }
}

fn objective(data: &[FitDatum], alpha: f64, beta: f64, weight_exp: f64) -> f64 {
    if !(alpha.is_finite() && beta.is_finite()) || alpha <= 0.0 || beta <= 0.0 {
        return f64::INFINITY;
    }
    let mut sse = 0.0_f64;
    for datum in data {
        let Some(yhat) = gamma_cdf_shape_rate(alpha, beta, datum.x.ln_1p()) else {
            return f64::INFINITY;
        };
        if !yhat.is_finite() {
            return f64::INFINITY;
        }
        let weight = datum.sd.max(1e-6).powf(weight_exp);
        let weight = if weight.is_finite() && weight > 0.0 {
            weight
        } else {
            1.0
        };
        let diff = yhat - datum.y;
        sse += weight * diff * diff;
    }
    sse
}

fn predict_gamma(alpha: f64, beta: f64, effort: f64) -> Option<f64> {
    if effort <= 0.0 || !effort.is_finite() {
        return Some(0.0);
    }
    gamma_cdf_shape_rate(alpha, beta, effort.ln_1p()).map(clamp01)
}

fn gamma_inverse_effort(alpha: f64, beta: f64, coverage: f64) -> Option<f64> {
    let log_effort = gamma_inverse_shape_rate(alpha, beta, clamp01(coverage))?;
    if !log_effort.is_finite() || log_effort > 700.0 {
        return None;
    }
    Some(log_effort.exp_m1())
}

fn build_model_curve(
    alpha: f64,
    beta: f64,
    points: &[ModelPoint],
    effort_at_target: Option<f64>,
) -> Vec<ModelCurvePoint> {
    let max_observed = points
        .iter()
        .map(|p| p.adjusted_effort)
        .fold(0.0_f64, f64::max);
    let max_effort = max_observed
        .max(effort_at_target.unwrap_or(0.0))
        .max(DEFAULT_MODEL_CURVE_MAX_EFFORT_BP);
    let min_effort = points
        .iter()
        .filter_map(|p| (p.adjusted_effort > 0.0).then_some(p.adjusted_effort))
        .fold(1_000.0_f64, f64::min)
        .max(1.0);

    log_grid(min_effort, max_effort, 768)
        .into_iter()
        .filter_map(|adjusted_effort| {
            predict_gamma(alpha, beta, adjusted_effort).map(|coverage| ModelCurvePoint {
                adjusted_effort,
                coverage,
            })
        })
        .collect()
}

fn log_grid(min: f64, max: f64, count: usize) -> Vec<f64> {
    if count <= 1 || min <= 0.0 || max <= min {
        return vec![min.max(1e-12)];
    }
    let min_ln = min.ln();
    let max_ln = max.ln();
    (0..count)
        .map(|i| {
            let t = i as f64 / (count - 1) as f64;
            (min_ln + t * (max_ln - min_ln)).exp()
        })
        .collect()
}

fn pearson(x: &[f64], y: &[f64]) -> Option<f64> {
    if x.len() != y.len() || x.len() < 2 {
        return None;
    }
    let mean_x = x.iter().sum::<f64>() / x.len() as f64;
    let mean_y = y.iter().sum::<f64>() / y.len() as f64;
    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for (&xi, &yi) in x.iter().zip(y) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    if var_x <= 0.0 || var_y <= 0.0 {
        None
    } else {
        Some(cov / (var_x * var_y).sqrt())
    }
}

fn gamma_cdf_shape_rate(alpha: f64, beta: f64, value: f64) -> Option<f64> {
    if !(alpha.is_finite() && beta.is_finite() && value.is_finite()) || alpha <= 0.0 || beta <= 0.0
    {
        return None;
    }
    if value <= 0.0 {
        return Some(0.0);
    }
    regularized_gamma_p(alpha, beta * value).map(clamp01)
}

fn gamma_inverse_shape_rate(alpha: f64, beta: f64, p: f64) -> Option<f64> {
    if !(alpha.is_finite() && beta.is_finite()) || alpha <= 0.0 || beta <= 0.0 {
        return None;
    }
    let p = clamp01(p);
    if p <= 0.0 {
        return Some(0.0);
    }
    if p >= 1.0 {
        return None;
    }

    let mut low = 0.0_f64;
    let mut high = alpha.max(1.0);
    for _ in 0..512 {
        let cdf = regularized_gamma_p(alpha, high)?;
        if cdf >= p {
            break;
        }
        high *= 2.0;
        if !high.is_finite() || high > 1e12 {
            return None;
        }
    }

    for _ in 0..120 {
        let mid = 0.5 * (low + high);
        let cdf = regularized_gamma_p(alpha, mid)?;
        if cdf < p {
            low = mid;
        } else {
            high = mid;
        }
    }

    Some(0.5 * (low + high) / beta)
}

fn regularized_gamma_p(alpha: f64, x: f64) -> Option<f64> {
    if !(alpha.is_finite() && x.is_finite()) || alpha <= 0.0 || x < 0.0 {
        return None;
    }
    if x == 0.0 {
        return Some(0.0);
    }
    if x < alpha + 1.0 {
        gamma_series_p(alpha, x)
    } else {
        gamma_continued_fraction_q(alpha, x).map(|q| 1.0 - q)
    }
}

fn gamma_series_p(alpha: f64, x: f64) -> Option<f64> {
    const EPS: f64 = 1e-14;
    const MAX_ITER: usize = 10_000;

    let gln = ln_gamma_lanczos(alpha);
    let mut ap = alpha;
    let mut del = 1.0 / alpha;
    let mut sum = del;
    for _ in 0..MAX_ITER {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if del.abs() < sum.abs() * EPS {
            let value = sum * (-x + alpha * x.ln() - gln).exp();
            return Some(value);
        }
    }
    None
}

fn gamma_continued_fraction_q(alpha: f64, x: f64) -> Option<f64> {
    const EPS: f64 = 1e-14;
    const FPMIN: f64 = 1e-300;
    const MAX_ITER: usize = 10_000;

    let gln = ln_gamma_lanczos(alpha);
    let mut b = x + 1.0 - alpha;
    let mut c = 1.0 / FPMIN;
    let mut d = 1.0 / b.max(FPMIN);
    let mut h = d;

    for i in 1..=MAX_ITER {
        let i_f = i as f64;
        let an = -i_f * (i_f - alpha);
        b += 2.0;
        d = an * d + b;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = b + an / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            let value = (-x + alpha * x.ln() - gln).exp() * h;
            return Some(value);
        }
    }
    None
}

fn ln_gamma_lanczos(z: f64) -> f64 {
    const COEFFS: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];

    if z < 0.5 {
        return std::f64::consts::PI.ln()
            - (std::f64::consts::PI * z).sin().ln()
            - ln_gamma_lanczos(1.0 - z);
    }

    let z = z - 1.0;
    let mut x = COEFFS[0];
    for (i, coeff) in COEFFS.iter().enumerate().skip(1) {
        x += coeff / (z + i as f64);
    }
    let t = z + 7.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + x.ln()
}

fn clamp01(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn fmt_optional(value: Option<f64>) -> String {
    value.map(fmt_float).unwrap_or_else(|| "NA".into())
}

fn fmt_float(value: f64) -> String {
    if value.is_finite() {
        format!("{value:.8}")
    } else {
        "NA".into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adjusted_effort_reaches_total_bases_at_last_point() {
        let summaries = vec![
            SampleSummary {
                reads: 0,
                bases: 0,
                portion: 0.0,
                mean: 0.0,
                sd: 0.0,
                q1: 0.0,
                median: 0.0,
                q3: 0.0,
            },
            SampleSummary {
                reads: 100,
                bases: 12_000,
                portion: 1.0,
                mean: 0.4,
                sd: 0.01,
                q1: 0.3,
                median: 0.4,
                q3: 0.5,
            },
        ];
        let model = fit_from_summaries(&summaries, 100, 12_000);
        let last = model.points.last().unwrap();
        assert!((last.adjusted_effort - 12_000.0).abs() < 1e-6);
    }
}
