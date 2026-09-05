use anyhow::{Result, bail};

use crate::coverage_model::{CoverageModel, DEFAULT_MODEL_CURVE_MAX_EFFORT_BP};

const MIN_SCORE: f64 = 1e-300;

#[derive(Clone, Debug)]
pub struct FamilyCurvePoint {
    pub family: String,
    pub effort: f64,
    pub coverage: f64,
}

#[derive(Clone, Debug)]
pub struct FamilyFit {
    pub family: String,
    pub params: String,
    pub parameter_count: usize,
    pub n: usize,
    pub sse: f64,
    pub rmse: f64,
    pub model_r: Option<f64>,
    pub aic: f64,
    pub bic: f64,
    pub effort_at_target: Option<f64>,
    pub mode_diversity: Option<f64>,
    pub area_diversity: Option<f64>,
    pub restricted_area_diversity: Option<f64>,
    pub coverage_at_observed_effort: Option<f64>,
    pub remaining_area_at_observed_effort: Option<f64>,
    distribution: Distribution,
}

#[derive(Clone, Copy, Debug)]
struct Datum {
    x: f64,
    y: f64,
}

#[derive(Clone, Copy, Debug)]
enum Distribution {
    Gamma {
        shape: f64,
        rate: f64,
    },
    GeneralizedGamma {
        shape: f64,
        scale: f64,
        power: f64,
    },
    BurrXii {
        scale: f64,
        c: f64,
        h: f64,
    },
    GammaMixture2 {
        weight: f64,
        shape1: f64,
        rate1: f64,
        shape2: f64,
        rate2: f64,
    },
}

pub fn fit_family_models(
    model: &CoverageModel,
    target_coverage: f64,
    restricted_quantile: f64,
) -> Result<Vec<FamilyFit>> {
    if !(target_coverage > 0.0 && target_coverage < 1.0) {
        bail!("target coverage must be in (0, 1)");
    }
    if !(restricted_quantile > 0.0 && restricted_quantile < 1.0) {
        bail!("restricted quantile must be in (0, 1)");
    }

    let data = fit_data(model);
    if data.len() < 5 {
        bail!(
            "need at least five empirical points in the fitting range; found {}",
            data.len()
        );
    }

    let mut fits = Vec::new();
    if let Some(distribution) = fit_gamma(&data, model) {
        fits.push(summarize_fit(
            distribution,
            &data,
            model.total_effort,
            target_coverage,
            restricted_quantile,
        ));
    }
    if let Some(distribution) =
        fit_generalized_gamma(&data, fits.first().map(|fit| fit.distribution))
    {
        fits.push(summarize_fit(
            distribution,
            &data,
            model.total_effort,
            target_coverage,
            restricted_quantile,
        ));
    }
    if let Some(distribution) = fit_burr(&data) {
        fits.push(summarize_fit(
            distribution,
            &data,
            model.total_effort,
            target_coverage,
            restricted_quantile,
        ));
    }
    if let Some(distribution) = fit_gamma_mixture2(&data, fits.first().map(|fit| fit.distribution))
    {
        fits.push(summarize_fit(
            distribution,
            &data,
            model.total_effort,
            target_coverage,
            restricted_quantile,
        ));
    }

    fits.sort_by(|a, b| a.bic.total_cmp(&b.bic));
    Ok(fits)
}

pub fn build_family_curves(fits: &[FamilyFit], model: &CoverageModel) -> Vec<FamilyCurvePoint> {
    let max_model_effort = fits
        .iter()
        .filter_map(|fit| fit.effort_at_target)
        .fold(0.0_f64, f64::max);
    let max_effort = model
        .total_effort
        .max(max_model_effort)
        .max(DEFAULT_MODEL_CURVE_MAX_EFFORT_BP);
    let min_effort = model
        .points
        .iter()
        .filter_map(|point| (point.adjusted_effort > 0.0).then_some(point.adjusted_effort))
        .fold(1.0e6_f64, f64::min)
        .max(1.0);

    let mut curves = Vec::new();
    for effort in log_grid(min_effort, max_effort, 512) {
        let x = effort.ln_1p();
        for fit in fits {
            if let Some(coverage) = fit.distribution.cdf_x(x) {
                curves.push(FamilyCurvePoint {
                    family: fit.family.clone(),
                    effort,
                    coverage,
                });
            }
        }
    }
    curves
}

fn fit_data(model: &CoverageModel) -> Vec<Datum> {
    model
        .points
        .iter()
        .filter_map(|point| {
            (point.coverage > 0.0 && point.coverage < 0.9 && point.adjusted_effort > 0.0).then_some(
                Datum {
                    x: point.adjusted_effort.ln_1p(),
                    y: point.coverage,
                },
            )
        })
        .collect()
}

fn fit_gamma(data: &[Datum], model: &CoverageModel) -> Option<Distribution> {
    let mut starts = Vec::new();
    if let (Some(shape), Some(rate)) = (model.alpha, model.beta) {
        starts.push(vec![shape.ln(), rate.ln()]);
    }

    let mean_guess = median_transition_x(data).unwrap_or_else(|| data[data.len() / 2].x);
    let means = unique_finite([
        12.0,
        15.0,
        18.0,
        mean_guess * 0.8,
        mean_guess,
        mean_guess * 1.2,
        24.0,
        30.0,
    ]);
    let shapes = [1.5_f64, 3.0, 8.0, 20.0, 50.0, 120.0, 320.0, 800.0, 1600.0];
    for shape in shapes {
        for mean in &means {
            starts.push(vec![shape.ln(), (shape / mean).ln()]);
        }
    }

    best_distribution_from_starts(
        starts,
        8,
        |z| {
            let shape = z[0].exp();
            let rate = z[1].exp();
            if !(0.05..=5_000.0).contains(&shape) || !(1e-5..=1_000.0).contains(&rate) {
                return None;
            }
            Some(Distribution::Gamma { shape, rate })
        },
        data,
    )
}

fn fit_generalized_gamma(data: &[Datum], gamma_fit: Option<Distribution>) -> Option<Distribution> {
    let mut starts = Vec::new();
    let (gamma_shape, gamma_rate, gamma_mean) = match gamma_fit {
        Some(Distribution::Gamma { shape, rate }) => (shape, rate, shape / rate),
        _ => {
            let mean = median_transition_x(data).unwrap_or_else(|| data[data.len() / 2].x);
            (80.0, 80.0 / mean, mean)
        }
    };
    starts.push(vec![
        gamma_shape.ln(),
        (1.0 / gamma_rate).ln(),
        1.0_f64.ln(),
    ]);

    let shapes = unique_finite([
        1.5,
        4.0,
        10.0,
        25.0,
        80.0,
        gamma_shape * 0.25,
        gamma_shape,
        gamma_shape * 4.0,
    ]);
    let powers = [
        0.35_f64, 0.5, 0.75, 1.0, 1.35, 1.8, 2.5, 3.5, 6.0, 10.0, 14.0,
    ];
    let means = unique_finite([
        gamma_mean * 0.8,
        gamma_mean,
        gamma_mean * 1.2,
        18.0,
        22.0,
        28.0,
    ]);

    for shape in shapes {
        for power in powers {
            let denom = (ln_gamma_lanczos(shape + 1.0 / power) - ln_gamma_lanczos(shape)).exp();
            if !denom.is_finite() || denom <= 0.0 {
                continue;
            }
            for mean in &means {
                let scale = mean / denom;
                starts.push(vec![shape.ln(), scale.ln(), power.ln()]);
            }
        }
    }

    best_distribution_from_starts(
        starts,
        12,
        |z| {
            let shape = z[0].exp();
            let scale = z[1].exp();
            let power = z[2].exp();
            if !(0.05..=5_000.0).contains(&shape)
                || !(1e-4..=500.0).contains(&scale)
                || !(0.2..=20.0).contains(&power)
            {
                return None;
            }
            Some(Distribution::GeneralizedGamma {
                shape,
                scale,
                power,
            })
        },
        data,
    )
}

fn fit_burr(data: &[Datum]) -> Option<Distribution> {
    let mut starts = Vec::new();
    let scale_center = median_transition_x(data).unwrap_or_else(|| data[data.len() / 2].x);
    let scales = unique_finite([
        6.0,
        10.0,
        14.0,
        scale_center * 0.75,
        scale_center,
        scale_center * 1.25,
        28.0,
        40.0,
    ]);
    let cs = [
        0.45_f64, 0.7, 1.0, 1.5, 2.2, 3.5, 6.0, 10.0, 16.0, 32.0, 55.0,
    ];
    let hs = [
        0.35_f64, 0.7, 1.0, 1.6, 2.5, 4.0, 8.0, 16.0, 32.0, 128.0, 512.0,
    ];
    for scale in scales {
        for c in cs {
            for h in hs {
                starts.push(vec![scale.ln(), c.ln(), h.ln()]);
            }
        }
    }

    best_distribution_from_starts(
        starts,
        12,
        |z| {
            let scale = z[0].exp();
            let c = z[1].exp();
            let h = z[2].exp();
            if !(0.1..=200.0).contains(&scale)
                || !(0.2..=100.0).contains(&c)
                || !(0.05..=2_000.0).contains(&h)
            {
                return None;
            }
            Some(Distribution::BurrXii { scale, c, h })
        },
        data,
    )
}

fn fit_gamma_mixture2(data: &[Datum], gamma_fit: Option<Distribution>) -> Option<Distribution> {
    let mut starts = Vec::new();
    let gamma_mean = match gamma_fit {
        Some(Distribution::Gamma { shape, rate }) => shape / rate,
        _ => median_transition_x(data).unwrap_or_else(|| data[data.len() / 2].x),
    };
    let weights = [0.2_f64, 0.35, 0.5, 0.65, 0.8];
    let low_means = unique_finite([gamma_mean * 0.65, gamma_mean * 0.8, 14.0, 17.0, 20.0]);
    let high_means = unique_finite([gamma_mean * 1.05, gamma_mean * 1.25, 23.0, 27.0, 34.0]);
    let shapes = [12.0_f64, 40.0, 120.0, 400.0, 1000.0];

    for weight in weights {
        for mean1 in &low_means {
            for mean2 in &high_means {
                if mean2 <= mean1 {
                    continue;
                }
                for shape in shapes {
                    starts.push(vec![
                        logit((weight - 0.01) / 0.98),
                        shape.ln(),
                        (shape / mean1).ln(),
                        shape.ln(),
                        (shape / mean2).ln(),
                    ]);
                }
            }
        }
    }

    best_distribution_from_starts(
        starts,
        8,
        |z| {
            let weight = 0.01 + 0.98 * logistic(z[0]);
            let shape1 = z[1].exp();
            let rate1 = z[2].exp();
            let shape2 = z[3].exp();
            let rate2 = z[4].exp();
            if !(0.05..=5_000.0).contains(&shape1)
                || !(0.05..=5_000.0).contains(&shape2)
                || !(1e-5..=1_000.0).contains(&rate1)
                || !(1e-5..=1_000.0).contains(&rate2)
            {
                return None;
            }
            let mut distribution = Distribution::GammaMixture2 {
                weight,
                shape1,
                rate1,
                shape2,
                rate2,
            };
            distribution.canonicalize();
            Some(distribution)
        },
        data,
    )
}

fn best_distribution_from_starts<F>(
    starts: Vec<Vec<f64>>,
    keep: usize,
    build: F,
    data: &[Datum],
) -> Option<Distribution>
where
    F: Fn(&[f64]) -> Option<Distribution>,
{
    let mut scored = starts
        .into_iter()
        .filter_map(|z| {
            let distribution = build(&z)?;
            let score = sse(data, distribution)?;
            score.is_finite().then_some((score, z))
        })
        .collect::<Vec<_>>();
    scored.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut best = None;
    for (_, start) in scored.into_iter().take(keep) {
        let Some((z, _)) = nelder_mead_refine(start, &build, data) else {
            continue;
        };
        let Some((z, score)) = coordinate_refine(z, &build, data) else {
            continue;
        };
        let Some(distribution) = build(&z) else {
            continue;
        };
        if best
            .as_ref()
            .map(|(best_score, _): &(f64, Distribution)| score < *best_score)
            .unwrap_or(true)
        {
            best = Some((score, distribution));
        }
    }
    best.map(|(_, distribution)| distribution)
}

fn nelder_mead_refine<F>(start: Vec<f64>, build: &F, data: &[Datum]) -> Option<(Vec<f64>, f64)>
where
    F: Fn(&[f64]) -> Option<Distribution>,
{
    let dim = start.len();
    let mut simplex = Vec::with_capacity(dim + 1);
    simplex.push((score_z(&start, build, data)?, start.clone()));
    for idx in 0..dim {
        let mut point = start.clone();
        point[idx] += 0.25;
        simplex.push((score_z(&point, build, data).unwrap_or(f64::INFINITY), point));
    }

    for _ in 0..900 {
        simplex.sort_by(|a, b| a.0.total_cmp(&b.0));
        let best_score = simplex[0].0;
        let worst_score = simplex[dim].0;
        if (worst_score - best_score).abs() < 1e-10 {
            break;
        }

        let mut centroid = vec![0.0_f64; dim];
        for (_, point) in simplex.iter().take(dim) {
            for (value, coord) in centroid.iter_mut().zip(point) {
                *value += *coord;
            }
        }
        for value in &mut centroid {
            *value /= dim as f64;
        }

        let worst = simplex[dim].1.clone();
        let reflected = affine_point(&centroid, &worst, 1.0);
        let reflected_score = score_z(&reflected, build, data).unwrap_or(f64::INFINITY);

        if reflected_score < simplex[0].0 {
            let expanded = affine_point(&centroid, &worst, 2.0);
            let expanded_score = score_z(&expanded, build, data).unwrap_or(f64::INFINITY);
            simplex[dim] = if expanded_score < reflected_score {
                (expanded_score, expanded)
            } else {
                (reflected_score, reflected)
            };
            continue;
        }

        if reflected_score < simplex[dim - 1].0 {
            simplex[dim] = (reflected_score, reflected);
            continue;
        }

        let contracted = affine_point(&centroid, &worst, -0.5);
        let contracted_score = score_z(&contracted, build, data).unwrap_or(f64::INFINITY);
        if contracted_score < worst_score {
            simplex[dim] = (contracted_score, contracted);
            continue;
        }

        let best_point = simplex[0].1.clone();
        for item in simplex.iter_mut().take(dim + 1).skip(1) {
            for (coord, best_coord) in item.1.iter_mut().zip(&best_point) {
                *coord = *best_coord + 0.5 * (*coord - *best_coord);
            }
            item.0 = score_z(&item.1, build, data).unwrap_or(f64::INFINITY);
        }
    }

    simplex.sort_by(|a, b| a.0.total_cmp(&b.0));
    simplex
        .into_iter()
        .next()
        .map(|(score, point)| (point, score))
}

fn affine_point(centroid: &[f64], worst: &[f64], factor: f64) -> Vec<f64> {
    centroid
        .iter()
        .zip(worst)
        .map(|(&c, &w)| c + factor * (c - w))
        .collect()
}

fn coordinate_refine<F>(mut z: Vec<f64>, build: &F, data: &[Datum]) -> Option<(Vec<f64>, f64)>
where
    F: Fn(&[f64]) -> Option<Distribution>,
{
    let mut best_score = score_z(&z, build, data)?;
    let mut step = 0.6_f64;
    let mut sweeps = 0_usize;

    while step > 1e-4 && sweeps < 80 {
        sweeps += 1;
        let mut improved = false;
        for dim in 0..z.len() {
            let original = z[dim];
            let mut best_dim_value = original;
            let mut best_dim_score = best_score;
            for delta in [-step, step] {
                z[dim] = original + delta;
                if let Some(score) = score_z(&z, build, data) {
                    if score < best_dim_score {
                        best_dim_score = score;
                        best_dim_value = z[dim];
                    }
                }
            }
            z[dim] = best_dim_value;
            if best_dim_score < best_score {
                best_score = best_dim_score;
                improved = true;
            }
        }
        if !improved {
            step *= 0.5;
        }
    }

    Some((z, best_score))
}

fn score_z<F>(z: &[f64], build: &F, data: &[Datum]) -> Option<f64>
where
    F: Fn(&[f64]) -> Option<Distribution>,
{
    sse(data, build(z)?)
}

fn summarize_fit(
    distribution: Distribution,
    data: &[Datum],
    observed_effort: f64,
    target_coverage: f64,
    restricted_quantile: f64,
) -> FamilyFit {
    let predictions = data
        .iter()
        .map(|datum| distribution.cdf_x(datum.x).unwrap_or(f64::NAN))
        .collect::<Vec<_>>();
    let observed = data.iter().map(|datum| datum.y).collect::<Vec<_>>();
    let sse = observed
        .iter()
        .zip(&predictions)
        .map(|(&y, &yhat)| {
            let diff = yhat - y;
            diff * diff
        })
        .sum::<f64>();
    let n = data.len();
    let parameter_count = distribution.parameter_count();
    let rmse = (sse / n as f64).sqrt();
    let model_r = pearson(&observed, &predictions);
    let aic = information_criterion(sse, n, parameter_count, false);
    let bic = information_criterion(sse, n, parameter_count, true);
    let effort_at_target = distribution
        .quantile_x(target_coverage)
        .and_then(log_effort_to_effort);
    let mode_diversity = distribution.mode();
    let area_diversity = distribution.mean();
    let restricted_area_diversity = distribution
        .quantile_x(restricted_quantile)
        .map(|q| integrate_survival(distribution, 0.0, q));
    let coverage_at_observed_effort = (observed_effort > 0.0 && observed_effort.is_finite())
        .then(|| distribution.cdf_x(observed_effort.ln_1p()))
        .flatten();
    let remaining_area_at_observed_effort = area_diversity.and_then(|mean| {
        (observed_effort > 0.0 && observed_effort.is_finite()).then(|| {
            let observed_x = observed_effort.ln_1p();
            (mean - integrate_survival(distribution, 0.0, observed_x)).max(0.0)
        })
    });

    FamilyFit {
        family: distribution.family_name().to_string(),
        params: distribution.params_string(),
        parameter_count,
        n,
        sse,
        rmse,
        model_r,
        aic,
        bic,
        effort_at_target,
        mode_diversity,
        area_diversity,
        restricted_area_diversity,
        coverage_at_observed_effort,
        remaining_area_at_observed_effort,
        distribution,
    }
}

fn sse(data: &[Datum], distribution: Distribution) -> Option<f64> {
    let mut total = 0.0_f64;
    for datum in data {
        let yhat = distribution.cdf_x(datum.x)?;
        if !yhat.is_finite() {
            return None;
        }
        let diff = yhat - datum.y;
        total += diff * diff;
    }
    Some(total)
}

impl Distribution {
    fn family_name(self) -> &'static str {
        match self {
            Distribution::Gamma { .. } => "gamma",
            Distribution::GeneralizedGamma { .. } => "generalized_gamma",
            Distribution::BurrXii { .. } => "burr_xii",
            Distribution::GammaMixture2 { .. } => "gamma_mixture2",
        }
    }

    fn parameter_count(self) -> usize {
        match self {
            Distribution::Gamma { .. } => 2,
            Distribution::GeneralizedGamma { .. } | Distribution::BurrXii { .. } => 3,
            Distribution::GammaMixture2 { .. } => 5,
        }
    }

    fn params_string(self) -> String {
        match self {
            Distribution::Gamma { shape, rate } => {
                format!("shape={shape:.8};rate={rate:.8}")
            }
            Distribution::GeneralizedGamma {
                shape,
                scale,
                power,
            } => {
                format!("shape={shape:.8};scale={scale:.8};power={power:.8}")
            }
            Distribution::BurrXii { scale, c, h } => {
                format!("scale={scale:.8};c={c:.8};h={h:.8}")
            }
            Distribution::GammaMixture2 {
                weight,
                shape1,
                rate1,
                shape2,
                rate2,
            } => format!(
                "weight1={weight:.8};shape1={shape1:.8};rate1={rate1:.8};weight2={:.8};shape2={shape2:.8};rate2={rate2:.8}",
                1.0 - weight
            ),
        }
    }

    fn cdf_x(self, x: f64) -> Option<f64> {
        if !x.is_finite() || x <= 0.0 {
            return Some(0.0);
        }
        match self {
            Distribution::Gamma { shape, rate } => gamma_cdf_shape_rate(shape, rate, x),
            Distribution::GeneralizedGamma {
                shape,
                scale,
                power,
            } => {
                if !(shape > 0.0 && scale > 0.0 && power > 0.0) {
                    return None;
                }
                regularized_gamma_p(shape, (x / scale).powf(power)).map(clamp01)
            }
            Distribution::BurrXii { scale, c, h } => {
                if !(scale > 0.0 && c > 0.0 && h > 0.0) {
                    return None;
                }
                Some(clamp01(1.0 - (1.0 + (x / scale).powf(c)).powf(-h)))
            }
            Distribution::GammaMixture2 {
                weight,
                shape1,
                rate1,
                shape2,
                rate2,
            } => {
                let c1 = gamma_cdf_shape_rate(shape1, rate1, x)?;
                let c2 = gamma_cdf_shape_rate(shape2, rate2, x)?;
                Some(clamp01(weight * c1 + (1.0 - weight) * c2))
            }
        }
    }

    fn quantile_x(self, p: f64) -> Option<f64> {
        let p = clamp01(p);
        if p <= 0.0 {
            return Some(0.0);
        }
        if p >= 1.0 {
            return None;
        }
        match self {
            Distribution::Gamma { shape, rate } => gamma_inverse_shape_rate(shape, rate, p),
            Distribution::GeneralizedGamma {
                shape,
                scale,
                power,
            } => {
                let gamma_q = gamma_inverse_shape_rate(shape, 1.0, p)?;
                Some(scale * gamma_q.powf(1.0 / power))
            }
            Distribution::BurrXii { scale, c, h } => {
                Some(scale * ((1.0 - p).powf(-1.0 / h) - 1.0).powf(1.0 / c))
            }
            Distribution::GammaMixture2 { .. } => self.inverse_by_bisection(p),
        }
    }

    fn inverse_by_bisection(self, p: f64) -> Option<f64> {
        let mut low = 0.0_f64;
        let mut high = 1.0_f64;
        for _ in 0..512 {
            if self.cdf_x(high)? >= p {
                break;
            }
            high *= 2.0;
            if !high.is_finite() || high > 1e6 {
                return None;
            }
        }
        for _ in 0..120 {
            let mid = 0.5 * (low + high);
            if self.cdf_x(mid)? < p {
                low = mid;
            } else {
                high = mid;
            }
        }
        Some(0.5 * (low + high))
    }

    fn mode(self) -> Option<f64> {
        match self {
            Distribution::Gamma { shape, rate } => (shape > 1.0).then_some((shape - 1.0) / rate),
            Distribution::GeneralizedGamma {
                shape,
                scale,
                power,
            } => {
                let numerator = shape * power - 1.0;
                (numerator > 0.0).then_some(scale * (numerator / power).powf(1.0 / power))
            }
            Distribution::BurrXii { scale, c, h } => {
                (c > 1.0).then_some(scale * ((c - 1.0) / (c * h + 1.0)).powf(1.0 / c))
            }
            Distribution::GammaMixture2 { .. } => None,
        }
    }

    fn mean(self) -> Option<f64> {
        match self {
            Distribution::Gamma { shape, rate } => Some(shape / rate),
            Distribution::GeneralizedGamma {
                shape,
                scale,
                power,
            } => Some(
                scale * (ln_gamma_lanczos(shape + 1.0 / power) - ln_gamma_lanczos(shape)).exp(),
            ),
            Distribution::BurrXii { scale, c, h } => {
                if c * h <= 1.0 {
                    return None;
                }
                Some(
                    scale
                        * (ln_gamma_lanczos(1.0 + 1.0 / c) + ln_gamma_lanczos(h - 1.0 / c)
                            - ln_gamma_lanczos(h))
                        .exp(),
                )
            }
            Distribution::GammaMixture2 {
                weight,
                shape1,
                rate1,
                shape2,
                rate2,
            } => Some(weight * shape1 / rate1 + (1.0 - weight) * shape2 / rate2),
        }
    }

    fn canonicalize(&mut self) {
        if let Distribution::GammaMixture2 {
            weight,
            shape1,
            rate1,
            shape2,
            rate2,
        } = self
        {
            let mean1 = *shape1 / *rate1;
            let mean2 = *shape2 / *rate2;
            if mean1 > mean2 {
                std::mem::swap(shape1, shape2);
                std::mem::swap(rate1, rate2);
                *weight = 1.0 - *weight;
            }
        }
    }
}

fn integrate_survival(distribution: Distribution, low: f64, high: f64) -> f64 {
    if !(low.is_finite() && high.is_finite()) || high <= low {
        return 0.0;
    }
    let intervals = 4096_usize;
    let h = (high - low) / intervals as f64;
    let mut sum = 0.0_f64;
    for i in 0..=intervals {
        let x = low + h * i as f64;
        let y = 1.0 - distribution.cdf_x(x).unwrap_or(1.0);
        let weight = if i == 0 || i == intervals {
            1.0
        } else if i % 2 == 0 {
            2.0
        } else {
            4.0
        };
        sum += weight * y.max(0.0);
    }
    sum * h / 3.0
}

fn information_criterion(sse: f64, n: usize, k: usize, bic: bool) -> f64 {
    let n = n as f64;
    let k = k as f64;
    let fit_term = n * (sse.max(MIN_SCORE) / n).ln();
    if bic {
        fit_term + k * n.ln()
    } else {
        fit_term + 2.0 * k
    }
}

fn median_transition_x(data: &[Datum]) -> Option<f64> {
    for pair in data.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        if (a.y <= 0.5 && b.y >= 0.5) || (a.y >= 0.5 && b.y <= 0.5) {
            let denom = b.y - a.y;
            if denom.abs() < 1e-12 {
                return Some(0.5 * (a.x + b.x));
            }
            let t = ((0.5 - a.y) / denom).clamp(0.0, 1.0);
            return Some(a.x + t * (b.x - a.x));
        }
    }
    data.last().map(|datum| datum.x)
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

fn log_effort_to_effort(log_effort: f64) -> Option<f64> {
    if !log_effort.is_finite() || log_effort > 700.0 {
        return None;
    }
    Some(log_effort.exp_m1())
}

fn gamma_cdf_shape_rate(shape: f64, rate: f64, value: f64) -> Option<f64> {
    if !(shape.is_finite() && rate.is_finite() && value.is_finite()) || shape <= 0.0 || rate <= 0.0
    {
        return None;
    }
    if value <= 0.0 {
        return Some(0.0);
    }
    regularized_gamma_p(shape, rate * value).map(clamp01)
}

fn gamma_inverse_shape_rate(shape: f64, rate: f64, p: f64) -> Option<f64> {
    if !(shape.is_finite() && rate.is_finite()) || shape <= 0.0 || rate <= 0.0 {
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
    let mut high = shape.max(1.0);
    for _ in 0..512 {
        let cdf = regularized_gamma_p(shape, high)?;
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
        let cdf = regularized_gamma_p(shape, mid)?;
        if cdf < p {
            low = mid;
        } else {
            high = mid;
        }
    }

    Some(0.5 * (low + high) / rate)
}

fn regularized_gamma_p(shape: f64, x: f64) -> Option<f64> {
    if !(shape.is_finite() && x.is_finite()) || shape <= 0.0 || x < 0.0 {
        return None;
    }
    if x == 0.0 {
        return Some(0.0);
    }
    if x < shape + 1.0 {
        gamma_series_p(shape, x)
    } else {
        gamma_continued_fraction_q(shape, x).map(|q| 1.0 - q)
    }
}

fn gamma_series_p(shape: f64, x: f64) -> Option<f64> {
    const EPS: f64 = 1e-14;
    const MAX_ITER: usize = 10_000;

    let gln = ln_gamma_lanczos(shape);
    let mut ap = shape;
    let mut del = 1.0 / shape;
    let mut sum = del;
    for _ in 0..MAX_ITER {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if del.abs() < sum.abs() * EPS {
            let value = sum * (-x + shape * x.ln() - gln).exp();
            return Some(value);
        }
    }
    None
}

fn gamma_continued_fraction_q(shape: f64, x: f64) -> Option<f64> {
    const EPS: f64 = 1e-14;
    const FPMIN: f64 = 1e-300;
    const MAX_ITER: usize = 10_000;

    let gln = ln_gamma_lanczos(shape);
    let mut b = x + 1.0 - shape;
    let mut c = 1.0 / FPMIN;
    let mut d = 1.0 / b.max(FPMIN);
    let mut h = d;

    for i in 1..=MAX_ITER {
        let i_f = i as f64;
        let an = -i_f * (i_f - shape);
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
            let value = (-x + shape * x.ln() - gln).exp() * h;
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

fn unique_finite(values: impl IntoIterator<Item = f64>) -> Vec<f64> {
    let mut values = values
        .into_iter()
        .filter(|value| value.is_finite() && *value > 0.0)
        .collect::<Vec<_>>();
    values.sort_by(|a, b| a.total_cmp(b));
    values.dedup_by(|a, b| (*a - *b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1.0));
    values
}

fn logistic(x: f64) -> f64 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

fn logit(p: f64) -> f64 {
    (p / (1.0 - p)).ln()
}

fn clamp01(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn fmt_option(value: Option<f64>) -> String {
    value
        .filter(|value| value.is_finite())
        .map(|value| format!("{value:.8}"))
        .unwrap_or_else(|| "NA".to_string())
}

impl FamilyFit {
    pub fn coverage_at_effort(&self, effort: f64) -> Option<f64> {
        if effort <= 0.0 || !effort.is_finite() {
            return Some(0.0);
        }
        self.distribution.cdf_x(effort.ln_1p())
    }

    pub fn legacy_gamma_params(&self) -> (Option<f64>, Option<f64>) {
        match self.distribution {
            Distribution::Gamma { shape, rate } => (Some(shape), Some(rate)),
            _ => (None, None),
        }
    }

    pub fn tsv_header() -> &'static str {
        "family\tparameters\tk\tn\tsse\trmse\tmodel_r\taic\tbic\teffort95_bp\tmode_diversity\tarea_diversity\tarea_diversity_q99\tcoverage_at_observed_effort\tremaining_area_at_observed_effort"
    }

    pub fn to_tsv_row(&self) -> String {
        format!(
            "{}\t{}\t{}\t{}\t{:.10}\t{:.10}\t{}\t{:.8}\t{:.8}\t{}\t{}\t{}\t{}\t{}\t{}",
            self.family,
            self.params,
            self.parameter_count,
            self.n,
            self.sse,
            self.rmse,
            fmt_option(self.model_r),
            self.aic,
            self.bic,
            fmt_option(self.effort_at_target),
            fmt_option(self.mode_diversity),
            fmt_option(self.area_diversity),
            fmt_option(self.restricted_area_diversity),
            fmt_option(self.coverage_at_observed_effort),
            fmt_option(self.remaining_area_at_observed_effort),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generalized_gamma_with_power_one_matches_gamma() {
        let gamma = Distribution::Gamma {
            shape: 4.2,
            rate: 0.7,
        };
        let gg = Distribution::GeneralizedGamma {
            shape: 4.2,
            scale: 1.0 / 0.7,
            power: 1.0,
        };
        for x in [0.1, 1.0, 4.0, 10.0] {
            let a = gamma.cdf_x(x).unwrap();
            let b = gg.cdf_x(x).unwrap();
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn burr_quantile_roundtrips() {
        let burr = Distribution::BurrXii {
            scale: 20.0,
            c: 3.0,
            h: 2.0,
        };
        for p in [0.1, 0.5, 0.95] {
            let x = burr.quantile_x(p).unwrap();
            let p2 = burr.cdf_x(x).unwrap();
            assert!((p - p2).abs() < 1e-12);
        }
    }
}
