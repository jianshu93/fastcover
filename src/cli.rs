use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Arg, ArgGroup, ArgMatches, Command, value_parser};

use crate::coverage::RunConfig;

pub fn parse_args() -> Result<RunConfig> {
    config_from_matches(command().get_matches())
}

pub fn command() -> Command {
    Command::new("fastcover")
        .version(env!("CARGO_PKG_VERSION"))
        .about(
            "FastCover: long-read metagenomic coverage and diversity estimation with a minimizer-Jaccard seed-chain quasi-aligner",
        )
        .arg(
            Arg::new("input")
                .short('i')
                .long("input")
                .help("Input FASTA/FASTQ, optionally gzip-compressed")
                .value_name("INPUT")
                .value_parser(value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("list")
                .long("list")
                .help("Text/TSV list of input files. Each non-comment line is: path [sample_label]")
                .value_name("LIST")
                .value_parser(value_parser!(PathBuf)),
        )
        .group(
            ArgGroup::new("input-source")
                .args(["input", "list"])
                .required(true)
                .multiple(false),
        )
        .arg(
            Arg::new("prefix")
                .short('p')
                .long("prefix")
                .help("Output prefix")
                .value_name("PREFIX")
                .default_value("fastcover")
                .value_parser(value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("threads")
                .short('t')
                .long("threads")
                .help("Number of worker threads. Defaults to all logical cores")
                .value_name("THREADS")
                .value_parser(value_parser!(usize)),
        )
        .arg(
            Arg::new("kmer")
                .short('k')
                .long("kmer")
                .help("K-mer size used by SIMD canonical minimizers and Mash-style identity conversion")
                .value_name("KMER")
                .default_value("16")
                .value_parser(value_parser!(usize)),
        )
        .arg(
            Arg::new("minimizer-window")
                .long("minimizer-window")
                .alias("windowSize")
                .help("Override the p-value-derived minimizer window size")
                .value_name("WINDOW")
                .value_parser(value_parser!(usize)),
        )
        .arg(
            Arg::new("sketch-size")
                .short('s')
                .long("sketch-size")
                .help("Bottom-k sketch size for each sliding minimizer-window identity estimate")
                .value_name("SKETCH_SIZE")
                .default_value("128")
                .value_parser(value_parser!(usize)),
        )
        .arg(
            Arg::new("identity")
                .long("identity")
                .alias("minIdentity")
                .help("Minimum identity for the sketch prefilter and final DP verifier; accepts fractions or percentages")
                .value_name("IDENTITY")
                .default_value("95")
                .value_parser(value_parser!(f64)),
        )
        .arg(
            Arg::new("p-value")
                .long("p-value")
                .alias("pValue")
                .help("P-value cutoff used to derive the minimizer window")
                .value_name("P_VALUE")
                .default_value("0.001")
                .value_parser(value_parser!(f64)),
        )
        .arg(
            Arg::new("reference-size")
                .long("reference-size")
                .alias("referenceSize")
                .help("Reference length scale used by the p-value-derived minimizer window")
                .value_name("BASES")
                .default_value("100000")
                .value_parser(value_parser!(u64)),
        )
        .arg(
            Arg::new("min-alignment-ratio")
                .long("min-alignment-ratio")
                .help("Minimum overlap fraction of the shorter read")
                .value_name("RATIO")
                .default_value("0.50")
                .value_parser(value_parser!(f64)),
        )
        .arg(
            Arg::new("min-query-coverage")
                .long("min-query-coverage")
                .help("Minimum aligned query fraction; use 0 for shorter-read-only overlap behavior")
                .value_name("FRACTION")
                .default_value("0.75")
                .value_parser(value_parser!(f64)),
        )
        .arg(
            Arg::new("min-shared-minimizers")
                .long("min-shared-minimizers")
                .help("Override the p-value-derived minimum unique shared minimizer hits in a diagonal band")
                .value_name("HITS")
                .value_parser(value_parser!(usize)),
        )
        .arg(
            Arg::new("prefilter-targets")
                .long("prefilter-targets")
                .alias("top-k")
                .help("Keep at most this many seed-supported prefilter targets per query")
                .value_name("K")
                .default_value("16")
                .value_parser(value_parser!(usize)),
        )
        .arg(
            Arg::new("alignment-targets")
                .long("alignment-targets")
                .alias("final-top-k")
                .help("Search at most this many top sketch-passing targets with rammap-core semi-global DP per query; only the best alignment can count as the mate; 0 disables final DP, maximum 16")
                .value_name("K")
                .default_value("3")
                .value_parser(value_parser!(usize)),
        )
        .arg(
            Arg::new("final-bandwidth")
                .long("final-bandwidth")
                .help("Bandwidth for final rammap-core DP alignment; -1 uses the full matrix")
                .value_name("BASES")
                .default_value("256")
                .value_parser(value_parser!(i32)),
        )
        .arg(
            Arg::new("max-hash-occ")
                .long("max-hash-occ")
                .help("Ignore minimizer hashes occurring in more than this many read/orientation entries")
                .value_name("OCC")
                .default_value("128")
                .value_parser(value_parser!(usize)),
        )
        .arg(
            Arg::new("diag-bin")
                .long("diag-bin")
                .help("Diagonal bin size for candidate grouping")
                .value_name("BASES")
                .default_value("1000")
                .value_parser(value_parser!(usize)),
        )
        .arg(
            Arg::new("slide-radius")
                .long("slide-radius")
                .help("Search this many bases on each side of a candidate diagonal center")
                .value_name("BASES")
                .default_value("3000")
                .value_parser(value_parser!(usize)),
        )
        .arg(
            Arg::new("slide-step")
                .long("slide-step")
                .help("Step size between evaluated sliding offsets")
                .value_name("BASES")
                .default_value("500")
                .value_parser(value_parser!(usize)),
        )
        .arg(
            Arg::new("tab-hash-seed")
                .long("tab-hash-seed")
                .help("Seed for deterministic twisted tabulation rehashing of SIMD minimizer values")
                .value_name("SEED")
                .default_value("42")
                .value_parser(value_parser!(u64)),
        )
        .arg(
            Arg::new("replicates")
                .short('n')
                .long("replicates")
                .help("Random replicates per coverage-curve point")
                .value_name("N")
                .default_value("32")
                .value_parser(value_parser!(usize)),
        )
        .arg(
            Arg::new("divide")
                .long("divide")
                .help("Log-spaced sampling divider")
                .value_name("DIVIDE")
                .default_value("0.70")
                .value_parser(value_parser!(f64)),
        )
        .arg(
            Arg::new("seed")
                .long("seed")
                .help("Random seed for reproducible curve resampling")
                .value_name("SEED")
                .default_value("1")
                .value_parser(value_parser!(u64)),
        )
        .arg(
            Arg::new("c-adjust")
                .long("c-adjust")
                .help("Enable C^EXP adjusted effort; omit this option for raw base effort")
                .value_name("EXP")
                .num_args(0..=1)
                .default_missing_value("0.27")
                .value_parser(value_parser!(f64)),
        )
}

fn config_from_matches(matches: ArgMatches) -> Result<RunConfig> {
    let identity = normalize_identity(
        *matches
            .get_one::<f64>("identity")
            .context("missing --identity")?,
        "--identity",
    )?;
    let alignment_targets = *matches
        .get_one::<usize>("alignment-targets")
        .context("missing --alignment-targets")?;
    anyhow::ensure!(
        alignment_targets <= 16,
        "--alignment-targets must be between 0 and 16"
    );
    let final_bandwidth = *matches
        .get_one::<i32>("final-bandwidth")
        .context("missing --final-bandwidth")?;
    anyhow::ensure!(
        final_bandwidth >= -1,
        "--final-bandwidth must be -1 or a non-negative integer"
    );
    let c_adjust = matches.get_one::<f64>("c-adjust").copied();
    if let Some(value) = c_adjust {
        anyhow::ensure!(
            value.is_finite() && value > 0.0 && value < 1.0,
            "--c-adjust must be finite and in the open interval (0,1)"
        );
    }

    Ok(RunConfig {
        input: matches.get_one::<PathBuf>("input").cloned(),
        list: matches.get_one::<PathBuf>("list").cloned(),
        prefix: matches
            .get_one::<PathBuf>("prefix")
            .cloned()
            .context("missing --prefix")?,
        threads: matches
            .get_one::<usize>("threads")
            .copied()
            .unwrap_or_else(num_cpus::get)
            .max(1),
        kmer: *matches.get_one::<usize>("kmer").context("missing --kmer")?,
        minimizer_window: matches.get_one::<usize>("minimizer-window").copied(),
        sketch_size: *matches
            .get_one::<usize>("sketch-size")
            .context("missing --sketch-size")?,
        identity,
        p_value: *matches
            .get_one::<f64>("p-value")
            .context("missing --p-value")?,
        reference_size: *matches
            .get_one::<u64>("reference-size")
            .context("missing --reference-size")?,
        min_alignment_ratio: *matches
            .get_one::<f64>("min-alignment-ratio")
            .context("missing --min-alignment-ratio")?,
        min_query_coverage: *matches
            .get_one::<f64>("min-query-coverage")
            .context("missing --min-query-coverage")?,
        min_shared_minimizers: matches.get_one::<usize>("min-shared-minimizers").copied(),
        prefilter_targets: *matches
            .get_one::<usize>("prefilter-targets")
            .context("missing --prefilter-targets")?,
        alignment_targets,
        final_bandwidth,
        max_hash_occ: *matches
            .get_one::<usize>("max-hash-occ")
            .context("missing --max-hash-occ")?,
        diag_bin: *matches
            .get_one::<usize>("diag-bin")
            .context("missing --diag-bin")?,
        slide_radius: *matches
            .get_one::<usize>("slide-radius")
            .context("missing --slide-radius")?,
        slide_step: *matches
            .get_one::<usize>("slide-step")
            .context("missing --slide-step")?,
        tab_hash_seed: *matches
            .get_one::<u64>("tab-hash-seed")
            .context("missing --tab-hash-seed")?,
        replicates: *matches
            .get_one::<usize>("replicates")
            .context("missing --replicates")?,
        divide: *matches
            .get_one::<f64>("divide")
            .context("missing --divide")?,
        seed: *matches.get_one::<u64>("seed").context("missing --seed")?,
        c_adjust,
    })
}

fn normalize_identity(value: f64, name: &str) -> Result<f64> {
    anyhow::ensure!(value.is_finite(), "{name} must be finite");
    if (0.0..=1.0).contains(&value) {
        return Ok(value);
    }
    if (1.0..=100.0).contains(&value) {
        return Ok(value / 100.0);
    }
    anyhow::bail!("{name} must be in [0,1] or [1,100]");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn c_adjust_is_disabled_by_default() {
        let matches = command()
            .try_get_matches_from(["fastcover", "--input", "reads.fq"])
            .unwrap();
        let config = config_from_matches(matches).unwrap();
        assert_eq!(config.c_adjust, None);
    }

    #[test]
    fn c_adjust_without_value_uses_nonpareil_exponent() {
        let matches = command()
            .try_get_matches_from(["fastcover", "--input", "reads.fq", "--c-adjust"])
            .unwrap();
        let config = config_from_matches(matches).unwrap();
        assert_eq!(config.c_adjust, Some(0.27));
    }

    #[test]
    fn c_adjust_accepts_custom_value() {
        let matches = command()
            .try_get_matches_from(["fastcover", "--input", "reads.fq", "--c-adjust=0.4"])
            .unwrap();
        let config = config_from_matches(matches).unwrap();
        assert_eq!(config.c_adjust, Some(0.4));
    }

    #[test]
    fn c_adjust_rejects_closed_interval_edges() {
        for value in ["0", "1"] {
            let matches = command()
                .try_get_matches_from(["fastcover", "--input", "reads.fq", "--c-adjust", value])
                .unwrap();
            assert!(config_from_matches(matches).is_err());
        }
    }
}
