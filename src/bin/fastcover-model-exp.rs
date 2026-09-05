use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use fastcover::model_families::{self, FamilyFit};
use fastcover::model_io;

#[derive(Debug, Parser)]
#[command(
    name = "fastcover-model-exp",
    version,
    about = "Experimental FastCover curve-family fits from an existing per-sample model TSV"
)]
struct Args {
    #[arg(
        short = 'm',
        long = "model",
        value_name = "MODEL_TSV",
        help = "Existing FastCover per-sample model TSV"
    )]
    model: PathBuf,

    #[arg(
        short = 'p',
        long = "prefix",
        value_name = "PREFIX",
        help = "Output prefix; defaults to MODEL_TSV without .model.tsv plus .model-exp"
    )]
    prefix: Option<PathBuf>,

    #[arg(
        long = "target-coverage",
        default_value_t = 0.95,
        help = "Coverage target for projected sequencing effort"
    )]
    target_coverage: f64,

    #[arg(
        long = "restricted-quantile",
        default_value_t = 0.99,
        help = "Quantile used for restricted area diversity"
    )]
    restricted_quantile: f64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let model = model_io::read_model(&args.model)?;
    let fits =
        model_families::fit_family_models(&model, args.target_coverage, args.restricted_quantile)?;
    let curves = model_families::build_family_curves(&fits, &model);
    let prefix = args
        .prefix
        .clone()
        .unwrap_or_else(|| default_prefix(&args.model));
    let family_path = PathBuf::from(format!("{}.families.tsv", prefix.display()));
    let curve_path = PathBuf::from(format!("{}.family-curves.tsv", prefix.display()));

    write_families(&family_path, &fits)?;
    write_curves(&curve_path, &curves)?;

    println!("{}", FamilyFit::tsv_header());
    for fit in &fits {
        println!("{}", fit.to_tsv_row());
    }
    eprintln!(
        "wrote experimental model-family fits to {} and curves to {}",
        family_path.display(),
        curve_path.display()
    );
    Ok(())
}

fn default_prefix(model_path: &Path) -> PathBuf {
    let path = model_path.to_string_lossy();
    if let Some(stripped) = path.strip_suffix(".model.tsv") {
        PathBuf::from(format!("{stripped}.model-exp"))
    } else if let Some(stripped) = path.strip_suffix(".tsv") {
        PathBuf::from(format!("{stripped}.model-exp"))
    } else {
        PathBuf::from(format!("{path}.model-exp"))
    }
}

fn write_families(path: &Path, fits: &[FamilyFit]) -> Result<()> {
    let file =
        File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    writeln!(writer, "{}", FamilyFit::tsv_header())?;
    for fit in fits {
        writeln!(writer, "{}", fit.to_tsv_row())?;
    }
    Ok(())
}

fn write_curves(path: &Path, curves: &[model_families::FamilyCurvePoint]) -> Result<()> {
    let file =
        File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    writeln!(writer, "family\teffort_bp\tcoverage")?;
    for point in curves {
        writeln!(
            writer,
            "{}\t{:.8}\t{:.8}",
            point.family, point.effort, point.coverage
        )?;
    }
    Ok(())
}
