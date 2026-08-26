use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use crate::coverage_model::{CoverageModel, ModelCurvePoint, ModelPoint};

#[derive(Clone, Debug)]
pub struct ModelListEntry {
    pub path: PathBuf,
    pub label: Option<String>,
}

pub fn read_model(path: &Path) -> Result<CoverageModel> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let reader = BufReader::new(file);
    parse_model(reader).with_context(|| format!("failed to parse {}", path.display()))
}

pub fn read_model_list(path: &Path) -> Result<Vec<ModelListEntry>> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let reader = BufReader::new(file);
    let base_dir = path.parent().unwrap_or_else(|| Path::new("."));
    let mut entries = Vec::new();
    for (idx, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("failed to read line {}", idx + 1))?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut fields = line.splitn(2, char::is_whitespace);
        let Some(path_field) = fields.next().filter(|value| !value.is_empty()) else {
            continue;
        };
        let label = fields
            .next()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned);
        let mut model_path = PathBuf::from(path_field);
        if model_path.is_relative() {
            model_path = base_dir.join(model_path);
        }
        entries.push(ModelListEntry {
            path: model_path,
            label,
        });
    }
    anyhow::ensure!(
        !entries.is_empty(),
        "{} did not contain any model paths",
        path.display()
    );
    Ok(entries)
}

pub fn derive_label(path: &Path) -> String {
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("sample");
    name.strip_suffix(".model.tsv")
        .or_else(|| name.strip_suffix(".tsv"))
        .unwrap_or(name)
        .to_string()
}

fn parse_model<R: BufRead>(reader: R) -> Result<CoverageModel> {
    let mut total_reads = 0_usize;
    let mut total_bases = 0_usize;
    let mut average_read_length = 0.0;
    let mut coverage_factor = 1.0;
    let mut c_adjust = None;
    let mut effort_adjust_scale = 1.0;
    let mut kappa = 0.0;
    let mut observed_coverage = 0.0;
    let mut total_effort = 0.0;
    let mut effort_at_target = None;
    let mut diversity = None;
    let mut model_r = None;
    let mut alpha = None;
    let mut beta = None;
    let mut target_coverage = 0.95;
    let mut warning = None;
    let mut points = Vec::new();
    let mut curve = Vec::new();

    for (idx, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("failed to read model line {}", idx + 1))?;
        if line.starts_with("# @") {
            parse_metadata(
                &line,
                &mut total_reads,
                &mut total_bases,
                &mut average_read_length,
                &mut coverage_factor,
                &mut c_adjust,
                &mut effort_adjust_scale,
                &mut kappa,
                &mut observed_coverage,
                &mut total_effort,
                &mut effort_at_target,
                &mut diversity,
                &mut model_r,
                &mut alpha,
                &mut beta,
                &mut target_coverage,
                &mut warning,
            )
            .with_context(|| format!("invalid metadata on line {}", idx + 1))?;
            continue;
        }
        if line.starts_with("kind\t") || line.trim().is_empty() {
            continue;
        }
        let fields = line.split('\t').collect::<Vec<_>>();
        anyhow::ensure!(
            fields.len() == 15,
            "line {} has {} fields; expected 15",
            idx + 1,
            fields.len()
        );
        match fields[0] {
            "observed" => points.push(parse_observed_point(&fields, idx + 1)?),
            "model" => curve.push(ModelCurvePoint {
                adjusted_effort: parse_f64(fields[4], "adjusted_effort_bp", idx + 1)?,
                coverage: parse_f64(fields[10], "coverage", idx + 1)?,
            }),
            other => bail!("line {} has unknown row kind {other}", idx + 1),
        }
    }

    anyhow::ensure!(!points.is_empty(), "model file contains no observed points");

    Ok(CoverageModel {
        total_reads,
        total_bases,
        average_read_length,
        coverage_factor,
        c_adjust,
        effort_adjust_scale,
        kappa,
        observed_coverage,
        total_effort,
        effort_at_target,
        diversity,
        model_r,
        alpha,
        beta,
        target_coverage,
        points,
        curve,
        warning,
    })
}

#[allow(clippy::too_many_arguments)]
fn parse_metadata(
    line: &str,
    total_reads: &mut usize,
    total_bases: &mut usize,
    average_read_length: &mut f64,
    coverage_factor: &mut f64,
    c_adjust: &mut Option<f64>,
    effort_adjust_scale: &mut f64,
    kappa: &mut f64,
    observed_coverage: &mut f64,
    total_effort: &mut f64,
    effort_at_target: &mut Option<f64>,
    diversity: &mut Option<f64>,
    model_r: &mut Option<f64>,
    alpha: &mut Option<f64>,
    beta: &mut Option<f64>,
    target_coverage: &mut f64,
    warning: &mut Option<String>,
) -> Result<()> {
    let Some((key, value)) = line[3..].split_once(':') else {
        return Ok(());
    };
    let value = value.trim();
    match key.trim() {
        "reads" => *total_reads = value.parse()?,
        "bases" => *total_bases = value.parse()?,
        "average_read_length" => *average_read_length = value.parse()?,
        "coverage_factor" => *coverage_factor = value.parse()?,
        "C_adjust" => *c_adjust = parse_optional(value)?,
        "effort_adjust_scale" => *effort_adjust_scale = value.parse()?,
        "kappa" => *kappa = value.parse()?,
        "C" => *observed_coverage = value.parse()?,
        "LR" => *total_effort = value.parse()?,
        "LRstar" => *effort_at_target = parse_optional(value)?,
        "diversity" => *diversity = parse_optional(value)?,
        "modelR" => *model_r = parse_optional(value)?,
        "alpha" => *alpha = parse_optional(value)?,
        "beta" => *beta = parse_optional(value)?,
        "target_coverage" => *target_coverage = value.parse()?,
        "note" => *warning = Some(value.to_string()),
        _ => {}
    }
    Ok(())
}

fn parse_observed_point(fields: &[&str], line_no: usize) -> Result<ModelPoint> {
    Ok(ModelPoint {
        reads: parse_u64(fields[1], "reads", line_no)?,
        bases: parse_u64(fields[2], "bases", line_no)?,
        portion: parse_f64(fields[3], "portion", line_no)?,
        adjusted_effort: parse_f64(fields[4], "adjusted_effort_bp", line_no)?,
        redundancy: parse_f64(fields[5], "redundant_fraction", line_no)?,
        sd: parse_f64(fields[6], "sd", line_no)?,
        q1: parse_f64(fields[7], "q1", line_no)?,
        median: parse_f64(fields[8], "median", line_no)?,
        q3: parse_f64(fields[9], "q3", line_no)?,
        coverage: parse_f64(fields[10], "coverage", line_no)?,
        q1_coverage: parse_f64(fields[11], "q1_coverage", line_no)?,
        median_coverage: parse_f64(fields[12], "median_coverage", line_no)?,
        q3_coverage: parse_f64(fields[13], "q3_coverage", line_no)?,
        fitted_coverage: parse_optional(fields[14])?,
    })
}

fn parse_u64(value: &str, field: &str, line_no: usize) -> Result<u64> {
    value
        .parse()
        .with_context(|| format!("line {line_no}: invalid {field} value {value:?}"))
}

fn parse_f64(value: &str, field: &str, line_no: usize) -> Result<f64> {
    value
        .parse()
        .with_context(|| format!("line {line_no}: invalid {field} value {value:?}"))
}

fn parse_optional(value: &str) -> Result<Option<f64>> {
    if value == "NA" || value == "." || value.eq_ignore_ascii_case("none") || value.is_empty() {
        Ok(None)
    } else {
        Ok(Some(value.parse()?))
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    #[test]
    fn derives_label_from_model_name() {
        assert_eq!(
            derive_label(Path::new("/tmp/marine.model.tsv")),
            "marine".to_string()
        );
        assert_eq!(
            derive_label(Path::new("/tmp/stool.tsv")),
            "stool".to_string()
        );
    }

    #[test]
    fn reads_model_metadata_and_rows() {
        let text = "\
# @impl: FastCover coverage gamma model
# @reads: 10
# @bases: 1000
# @average_read_length: 100
# @coverage_factor: 1
# @C_adjust: none
# @effort_adjust_scale: 1
# @kappa: 0.5
# @C: 0.5
# @LR: 1000
# @LRstar: 2000
# @diversity: 12.5
# @modelR: 0.99
# @alpha: 3
# @beta: 0.16
# @target_coverage: 0.95
# @note: parser smoke test
kind\treads\tbases\tportion\tadjusted_effort_bp\tredundant_fraction\tsd\tq1\tmedian\tq3\tcoverage\tq1_coverage\tmedian_coverage\tq3_coverage\tfitted_coverage
observed\t10\t1000\t1\t1000\t0.5\t0.01\t0.4\t0.5\t0.6\t0.5\t0.4\t0.5\t0.6\t0.51
model\t.\t.\t.\t1500\t.\t.\t.\t.\t.\t0.7\t.\t.\t.\t0.7
";
        let model = parse_model(Cursor::new(text)).unwrap();
        assert_eq!(model.total_reads, 10);
        assert_eq!(model.c_adjust, None);
        assert_eq!(model.diversity, Some(12.5));
        assert_eq!(model.points.len(), 1);
        assert_eq!(model.curve.len(), 1);
        assert_eq!(model.points[0].fitted_coverage, Some(0.51));
        assert_eq!(model.warning.as_deref(), Some("parser smoke test"));
    }

    #[test]
    fn reads_legacy_model_without_c_adjust_metadata() {
        let text = "\
# @reads: 1
# @bases: 100
kind\treads\tbases\tportion\tadjusted_effort_bp\tredundant_fraction\tsd\tq1\tmedian\tq3\tcoverage\tq1_coverage\tmedian_coverage\tq3_coverage\tfitted_coverage
observed\t1\t100\t1\t100\t0.2\t0\t0.2\t0.2\t0.2\t0.2\t0.2\t0.2\t0.2\tNA
";
        let model = parse_model(Cursor::new(text)).unwrap();
        assert_eq!(model.c_adjust, None);
        assert_eq!(model.effort_adjust_scale, 1.0);
    }
}
