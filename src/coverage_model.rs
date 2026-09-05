use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result};

use crate::model_families;
use crate::types::SampleSummary;

const TARGET_COVERAGE: f64 = 0.95;
pub const DEFAULT_MODEL_CURVE_MAX_EFFORT_BP: f64 = 1.0e13;
const DEFAULT_PRODUCTION_MODEL_FAMILY: &str = "gamma_mixture2";
const RESTRICTED_DIVERSITY_QUANTILE: f64 = 0.99;

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
    pub c_adjust: Option<f64>,
    pub effort_adjust_scale: f64,
    pub kappa: f64,
    pub observed_coverage: f64,
    pub total_effort: f64,
    pub effort_at_target: Option<f64>,
    pub diversity: Option<f64>,
    pub diversity_q99: Option<f64>,
    pub remaining_diversity: Option<f64>,
    pub model_r: Option<f64>,
    pub model_family: String,
    pub model_params: Option<String>,
    pub alpha: Option<f64>,
    pub beta: Option<f64>,
    pub target_coverage: f64,
    pub points: Vec<ModelPoint>,
    pub curve: Vec<ModelCurvePoint>,
    pub warning: Option<String>,
}

pub fn fit_from_summaries(
    summaries: &[SampleSummary],
    total_reads: usize,
    total_bases: usize,
    c_adjust: Option<f64>,
) -> CoverageModel {
    let average_read_length = if total_reads == 0 {
        0.0
    } else {
        total_bases as f64 / total_reads as f64
    };

    let coverage_factor = long_read_coverage_factor();
    let kappa = summaries.last().map(|s| clamp01(s.mean)).unwrap_or(0.0);
    let observed_coverage = redundancy_to_coverage(kappa, coverage_factor);
    let effort_adjust_scale = effort_adjust_scale(observed_coverage, c_adjust);
    let total_effort = total_bases as f64;
    let points = build_points(
        summaries,
        total_reads,
        average_read_length,
        coverage_factor,
        observed_coverage,
        c_adjust,
    );

    let mut model = CoverageModel {
        total_reads,
        total_bases,
        average_read_length,
        coverage_factor,
        c_adjust,
        effort_adjust_scale,
        kappa,
        observed_coverage,
        total_effort,
        effort_at_target: None,
        diversity: None,
        diversity_q99: None,
        remaining_diversity: None,
        model_r: None,
        model_family: DEFAULT_PRODUCTION_MODEL_FAMILY.to_string(),
        model_params: None,
        alpha: None,
        beta: None,
        target_coverage: TARGET_COVERAGE,
        points,
        curve: Vec::new(),
        warning: None,
    };

    match model_families::fit_family_models(&model, TARGET_COVERAGE, RESTRICTED_DIVERSITY_QUANTILE)
    {
        Ok(fits) => {
            let selected = fits
                .iter()
                .find(|fit| fit.family == DEFAULT_PRODUCTION_MODEL_FAMILY)
                .or_else(|| fits.first())
                .cloned();
            if let Some(fit) = selected {
                for point in &mut model.points {
                    point.fitted_coverage = fit.coverage_at_effort(point.adjusted_effort);
                }
                model.curve =
                    model_families::build_family_curves(std::slice::from_ref(&fit), &model)
                        .into_iter()
                        .map(|point| ModelCurvePoint {
                            adjusted_effort: point.effort,
                            coverage: point.coverage,
                        })
                        .collect();
                let (alpha, beta) = fit.legacy_gamma_params();
                model.model_family = fit.family.clone();
                model.model_params = Some(fit.params.clone());
                model.effort_at_target = fit.effort_at_target;
                model.diversity = fit.area_diversity;
                model.diversity_q99 = fit.restricted_area_diversity;
                model.remaining_diversity = fit.remaining_area_at_observed_effort;
                model.model_r = fit.model_r;
                model.alpha = alpha;
                model.beta = beta;
                model.warning = Some(format!(
                    "selected {} coverage model; SSE {:.6}; BIC {:.3}",
                    fit.family, fit.sse, fit.bic
                ));
            }
        }
        Err(err) => {
            model.warning = Some(err.to_string());
        }
    }

    model
}

pub fn write_model(path: &Path, model: &CoverageModel) -> Result<()> {
    let file =
        File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    let mut w = BufWriter::new(file);

    writeln!(
        w,
        "# @impl: FastCover coverage {} model",
        model.model_family
    )?;
    writeln!(w, "# @version: {}", env!("CARGO_PKG_VERSION"))?;
    writeln!(w, "# @model_family: {}", model.model_family)?;
    writeln!(
        w,
        "# @model_params: {}",
        model.model_params.as_deref().unwrap_or("NA")
    )?;
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
    writeln!(
        w,
        "# @C_adjust: {}",
        model
            .c_adjust
            .map(fmt_float)
            .unwrap_or_else(|| "none".into())
    )?;
    writeln!(
        w,
        "# @effort_adjust_scale: {}",
        fmt_float(model.effort_adjust_scale)
    )?;
    writeln!(w, "# @kappa: {}", fmt_float(model.kappa))?;
    writeln!(w, "# @C: {}", fmt_float(model.observed_coverage))?;
    writeln!(w, "# @LR: {}", fmt_float(model.total_effort))?;
    writeln!(w, "# @LRstar: {}", fmt_optional(model.effort_at_target))?;
    writeln!(w, "# @diversity: {}", fmt_optional(model.diversity))?;
    writeln!(
        w,
        "# @diversity_definition: area_under_log_effort_survival_curve"
    )?;
    writeln!(w, "# @diversity_q99: {}", fmt_optional(model.diversity_q99))?;
    writeln!(
        w,
        "# @remaining_diversity_at_observed_effort: {}",
        fmt_optional(model.remaining_diversity)
    )?;
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
    c_adjust: Option<f64>,
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
    let c_scale = effort_adjust_scale(observed_coverage, c_adjust);

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

fn effort_adjust_scale(observed_coverage: f64, c_adjust: Option<f64>) -> f64 {
    c_adjust
        .map(|exponent| observed_coverage.max(1e-12).powf(exponent))
        .unwrap_or(1.0)
}

fn redundancy_to_coverage(redundancy: f64, coverage_factor: f64) -> f64 {
    clamp01(redundancy).powf(coverage_factor)
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
        let model = fit_from_summaries(&summaries, 100, 12_000, None);
        let last = model.points.last().unwrap();
        assert!((last.adjusted_effort - 12_000.0).abs() < 1e-6);
        assert_eq!(model.c_adjust, None);
        assert!((model.effort_adjust_scale - 1.0).abs() < 1e-12);
    }

    #[test]
    fn default_adjusted_effort_uses_raw_bases() {
        let summaries = vec![
            SampleSummary {
                reads: 50,
                bases: 6_000,
                portion: 0.5,
                mean: 0.2,
                sd: 0.01,
                q1: 0.15,
                median: 0.2,
                q3: 0.25,
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
        let model = fit_from_summaries(&summaries, 100, 12_000, None);
        assert!((model.points[0].adjusted_effort - 6_000.0).abs() < 1e-6);
        assert!((model.points[1].adjusted_effort - 12_000.0).abs() < 1e-6);
    }

    #[test]
    fn c_adjust_moves_lower_effort_points_toward_full_effort() {
        let summaries = vec![
            SampleSummary {
                reads: 50,
                bases: 6_000,
                portion: 0.5,
                mean: 0.2,
                sd: 0.01,
                q1: 0.15,
                median: 0.2,
                q3: 0.25,
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
        let model = fit_from_summaries(&summaries, 100, 12_000, Some(0.27));
        assert_eq!(model.c_adjust, Some(0.27));
        assert!(model.effort_adjust_scale > 0.0);
        assert!(model.effort_adjust_scale < 1.0);
        assert!(model.points[0].adjusted_effort > 6_000.0);
        assert!(model.points[0].adjusted_effort < 12_000.0);
        assert!((model.points[1].adjusted_effort - 12_000.0).abs() < 1e-6);
    }
}
