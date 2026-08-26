use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use fastcover::model_io::{self, ModelListEntry};
use fastcover::plot::{self, PlotSample};

#[derive(Debug, Parser)]
#[command(
    name = "fastcover-plot",
    version,
    about = "Merge FastCover per-sample model outputs into one SVG/PDF coverage and diversity plot"
)]
struct Args {
    #[arg(
        short = 'm',
        long = "model",
        value_name = "MODEL_TSV",
        conflicts_with = "list",
        help = "Per-sample FastCover model TSV; may be provided multiple times"
    )]
    models: Vec<PathBuf>,

    #[arg(
        long = "list",
        value_name = "LIST",
        conflicts_with = "models",
        help = "Text/TSV list of per-sample model files: path/to/sample.model.tsv [sample_label]"
    )]
    list: Option<PathBuf>,

    #[arg(
        long = "label",
        value_name = "LABEL",
        requires = "models",
        help = "Optional label for each --model entry; repeat in the same order as --model"
    )]
    labels: Vec<String>,

    #[arg(
        short = 'p',
        long = "prefix",
        value_name = "PREFIX",
        default_value = "fastcover-merged",
        help = "Output prefix for merged SVG/PDF"
    )]
    prefix: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let samples = load_samples(&args)?;
    let (svg_path, pdf_path) = output_paths(&args.prefix);
    plot::write_coverage_plots(&samples, &svg_path, &pdf_path)?;
    eprintln!(
        "merged {} FastCover model files into {} and {}",
        samples.len(),
        svg_path.display(),
        pdf_path.display()
    );
    Ok(())
}

fn load_samples(args: &Args) -> Result<Vec<PlotSample>> {
    let entries = if let Some(list) = &args.list {
        anyhow::ensure!(
            args.labels.is_empty(),
            "--label can only be used together with --model"
        );
        model_io::read_model_list(list)?
    } else {
        anyhow::ensure!(
            !args.models.is_empty(),
            "provide at least one --model or a --list file"
        );
        anyhow::ensure!(
            args.labels.is_empty() || args.labels.len() == args.models.len(),
            "--label must be repeated exactly once per --model entry"
        );
        args.models
            .iter()
            .enumerate()
            .map(|(idx, path)| ModelListEntry {
                path: path.clone(),
                label: args.labels.get(idx).cloned(),
            })
            .collect()
    };

    let samples = entries
        .iter()
        .map(|entry| {
            let model = model_io::read_model(&entry.path)?;
            let label = entry
                .label
                .clone()
                .unwrap_or_else(|| model_io::derive_label(&entry.path));
            Ok(PlotSample { label, model })
        })
        .collect::<Result<Vec<_>>>()
        .context("failed to load FastCover model files")?;

    anyhow::ensure!(!samples.is_empty(), "no samples to plot");
    Ok(samples)
}

fn output_paths(prefix: &Path) -> (PathBuf, PathBuf) {
    let s = prefix.to_string_lossy();
    (
        PathBuf::from(format!("{s}.svg")),
        PathBuf::from(format!("{s}.pdf")),
    )
}
