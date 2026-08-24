use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use simd_minimizers::packed_seq::{PackedSeqVec, SeqVec};
use tab_hash::Tab64Twisted;

use crate::plot::{self, PlotSample};
use crate::sketch_params::{self, FilterParams};
use crate::types::{AlignmentStats, MateResult, OutputPaths, ReadRecord, SampleSummary};
use crate::{coverage_model, final_align, io, utils};

#[derive(Debug, Clone)]
pub struct RunConfig {
    /// Input FASTA/FASTQ, optionally gzip-compressed.
    pub input: Option<PathBuf>,

    /// Text/TSV list of input files. Each non-comment line is: path [sample_label].
    pub list: Option<PathBuf>,

    /// Output prefix.
    pub prefix: PathBuf,

    /// Number of worker threads.
    pub threads: usize,

    /// K-mer size used by SIMD canonical minimizers and Mash-style identity conversion.
    pub kmer: usize,

    /// Optional override for the minimizer window size.
    pub minimizer_window: Option<usize>,

    /// Bottom-k sketch size for each sliding minimizer-window identity estimate.
    pub sketch_size: usize,

    /// Minimum identity. With final DP enabled, this is both sketch prefilter and final identity.
    pub identity: f64,

    /// P-value cutoff used to derive the minimizer window.
    pub p_value: f64,

    /// Reference length scale used to derive the minimizer window.
    pub reference_size: u64,

    /// Minimum overlap fraction of the shorter read.
    pub min_alignment_ratio: f64,

    /// Optional extra minimum aligned query fraction. Use 0 for shorter-read overlap behavior.
    pub min_query_coverage: f64,

    /// Optional override for minimum unique shared minimizer hits in a diagonal band.
    pub min_shared_minimizers: Option<usize>,

    /// Keep at most this many seed-supported prefilter targets per query.
    pub prefilter_targets: usize,

    /// Run final base-level DP alignment on at most this many top alignment targets.
    pub alignment_targets: usize,

    /// Bandwidth for final rammap-core semi-global DP alignment; -1 is full matrix.
    pub final_bandwidth: i32,

    /// Ignore minimizer hashes occurring in more than this many read/orientation entries.
    pub max_hash_occ: usize,

    /// Diagonal bin size for candidate grouping.
    pub diag_bin: usize,

    /// Search this many bases on each side of a candidate diagonal center.
    pub slide_radius: usize,

    /// Step size between evaluated sliding offsets.
    pub slide_step: usize,

    /// Seed for deterministic twisted tabulation rehashing of SIMD minimizer values.
    pub tab_hash_seed: u64,

    /// Random replicates per coverage-curve point.
    pub replicates: usize,

    /// Log-spaced sampling divider.
    pub divide: f64,

    /// Random seed for reproducible curve resampling.
    pub seed: u64,

    /// Optional Nonpareil-style C^EXP effort adjustment. None means raw base effort.
    pub c_adjust: Option<f64>,
}

#[derive(Clone, Debug)]
struct Seed {
    hash: u64,
    pos: usize,
}

#[derive(Clone, Copy, Debug)]
struct IndexHit {
    target: usize,
    pos: usize,
    rev: bool,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct BandKey {
    target: usize,
    rev: bool,
    diag_bin: isize,
}

#[derive(Clone, Debug, Default)]
struct BandAccum {
    hits: usize,
    sum_diag: i128,
    min_diag: isize,
    max_diag: isize,
    query_positions: Vec<usize>,
}

#[derive(Clone, Debug)]
struct SlideCandidate {
    target: usize,
    rev: bool,
    center_diag: isize,
    min_diag: isize,
    max_diag: isize,
    hits: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct SketchIdentity {
    identity: f64,
    shared: usize,
    denominator: usize,
}

#[derive(Clone, Copy, Debug)]
struct WindowScore {
    q_start: usize,
    t_start: usize,
    offset: isize,
    overlap_len: usize,
    alignment_ratio: f64,
    query_coverage: f64,
    identity: f64,
    shared: usize,
    denominator: usize,
}

#[derive(Clone, Debug, Default)]
struct RunStats {
    candidate_edges: usize,
    evaluated_windows: usize,
    final_alignments: usize,
    passing_best_alignments: usize,
}

#[derive(Clone, Debug)]
struct ScoredCandidate {
    target: usize,
    rev: bool,
    hits: usize,
    window: WindowScore,
}

#[derive(Clone, Debug)]
struct BestAlignmentChoice {
    alignment: AlignmentStats,
    tie_key: u64,
}

#[derive(Clone, Debug)]
struct SampleInput {
    path: PathBuf,
    label: String,
}

#[derive(Clone, Copy, Debug)]
struct ResolvedFilterParams {
    recommended: FilterParams,
    filter_fragment_len: usize,
    requested_minimizer_window: Option<usize>,
    actual_minimizer_window: usize,
    requested_min_shared_minimizers: Option<usize>,
    actual_min_shared_minimizers: usize,
}

pub fn run(args: RunConfig) -> Result<()> {
    ensure_parent(&args.prefix)?;
    ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build_global()
        .context("initialize rayon thread pool")?;
    eprintln!("using {} rayon threads", rayon::current_num_threads());

    let samples = load_sample_inputs(&args)?;
    let list_mode = args.list.is_some();
    let mut plot_samples = Vec::new();
    for sample in &samples {
        let sample_prefix = if list_mode {
            prefix_with_sample_label(&args.prefix, &sample.label)
        } else {
            args.prefix.clone()
        };
        let plot_sample = run_sample(&args, sample, &sample_prefix)?;
        plot_samples.push(plot_sample);
    }

    if list_mode {
        let paths = OutputPaths::from_prefix(&args.prefix);
        plot::write_coverage_plots(&plot_samples, &paths.svg, &paths.pdf)?;
        eprintln!("wrote combined {}", paths.svg.display());
        eprintln!("wrote combined {}", paths.pdf.display());
    }
    Ok(())
}

fn run_sample(args: &RunConfig, sample: &SampleInput, prefix: &Path) -> Result<PlotSample> {
    ensure_parent(prefix)?;
    let paths = OutputPaths::from_prefix(prefix);
    let start = Instant::now();
    eprintln!("reading {} ({})", sample.path.display(), sample.label);
    let reads = io::read_fastx(&sample.path)?;
    let total_bases = reads.iter().map(ReadRecord::len).sum::<usize>();
    eprintln!(
        "{}: loaded {} reads, {:.3} Mbp in {:.2}s",
        sample.label,
        reads.len(),
        total_bases as f64 / 1e6,
        start.elapsed().as_secs_f64()
    );

    let tab_hasher = deterministic_tab64_twisted(args.tab_hash_seed);
    let filter_fragment_len = infer_filter_fragment_len(reads.len(), total_bases, args.kmer);
    let filter_params = resolve_filter_params(args, filter_fragment_len);
    let minimizer_window = filter_params.actual_minimizer_window;
    eprintln!(
        "{}: extracting SIMD minimizers with k={} w={} identity={:.3}% p={} filter_fragment_len={} reference_size={} min_shared={} and twisted tabulation seed {}",
        sample.label,
        args.kmer,
        minimizer_window,
        args.identity * 100.0,
        args.p_value,
        filter_params.filter_fragment_len,
        args.reference_size,
        filter_params.actual_min_shared_minimizers,
        args.tab_hash_seed
    );

    let seed_start = Instant::now();
    let reverse_complements = reads
        .par_iter()
        .map(|read| reverse_complement(&read.seq))
        .collect::<Vec<_>>();
    let forward_seeds = reads
        .par_iter()
        .map(|read| simd_twisted_minimizers(&read.seq, args.kmer, minimizer_window, &tab_hasher))
        .collect::<Vec<_>>();
    let reverse_seeds = reverse_complements
        .par_iter()
        .map(|seq| simd_twisted_minimizers(seq, args.kmer, minimizer_window, &tab_hasher))
        .collect::<Vec<_>>();
    let lookup = build_lookup(&forward_seeds, &reverse_seeds);
    eprintln!(
        "{}: built minimizer lookup in {:.2}s",
        sample.label,
        seed_start.elapsed().as_secs_f64()
    );
    let (forward_nt4, reverse_nt4) = if args.alignment_targets > 0 {
        let encode_start = Instant::now();
        let forward_nt4 = reads
            .par_iter()
            .map(|read| final_align::encode_ascii_nt4(&read.seq))
            .collect::<Vec<_>>();
        let reverse_nt4 = reverse_complements
            .par_iter()
            .map(|seq| final_align::encode_ascii_nt4(seq))
            .collect::<Vec<_>>();
        eprintln!(
            "{}: pre-encoded reads for final DP in {:.2}s",
            sample.label,
            encode_start.elapsed().as_secs_f64()
        );
        (forward_nt4, reverse_nt4)
    } else {
        (Vec::new(), Vec::new())
    };

    let score_start = Instant::now();
    eprintln!(
        "{}: running de novo sliding minimizer MinHash scoring",
        sample.label
    );
    let per_query = forward_seeds
        .par_iter()
        .enumerate()
        .map(|(query, seeds)| {
            score_query(
                query,
                seeds,
                &lookup,
                &reads,
                &reverse_complements,
                &forward_seeds,
                &reverse_seeds,
                &forward_nt4,
                &reverse_nt4,
                &args,
                &filter_params,
            )
        })
        .collect::<Vec<_>>();
    let mates = per_query
        .iter()
        .map(|(mate, _)| mate.clone())
        .collect::<Vec<_>>();
    let stats = per_query
        .iter()
        .fold(RunStats::default(), |mut acc, (_, stats)| {
            acc.candidate_edges += stats.candidate_edges;
            acc.evaluated_windows += stats.evaluated_windows;
            acc.final_alignments += stats.final_alignments;
            acc.passing_best_alignments += stats.passing_best_alignments;
            acc
        });
    eprintln!(
        "{}: sliding MinHash scoring completed in {:.2}s: {} candidates, {} windows, {} final alignments, {} passing best alignments",
        sample.label,
        score_start.elapsed().as_secs_f64(),
        stats.candidate_edges,
        stats.evaluated_windows,
        stats.final_alignments,
        stats.passing_best_alignments
    );

    let keep = utils::greedy_representatives(&mates);
    let kept = keep.iter().filter(|&&x| x).count();
    eprintln!(
        "{}: non-redundant representatives: {} / {} ({:.2}%)",
        sample.label,
        kept,
        reads.len(),
        kept as f64 * 100.0 / reads.len().max(1) as f64
    );

    let sample_start = Instant::now();
    let read_lengths = reads.iter().map(ReadRecord::len).collect::<Vec<_>>();
    let (summaries, all_values) = utils::sample_curve(
        &mates,
        &read_lengths,
        args.replicates,
        args.divide,
        args.seed,
    );
    eprintln!(
        "{}: coverage-curve resampling completed in {:.2}s",
        sample.label,
        sample_start.elapsed().as_secs_f64()
    );

    let fitted_model =
        coverage_model::fit_from_summaries(&summaries, reads.len(), total_bases, args.c_adjust);
    write_summary(
        &paths.summary,
        &summaries,
        &args,
        &sample.label,
        &filter_params,
        &stats,
        reads.len(),
        total_bases,
    )?;
    write_all(&paths.all, &all_values)?;
    coverage_model::write_model(&paths.model, &fitted_model)?;
    write_mates(&paths.mates, &reads, &mates, &keep)?;
    write_ids(
        &paths.nonredundant_ids,
        reads
            .iter()
            .zip(&keep)
            .filter_map(|(read, &is_kept)| is_kept.then(|| read.id.as_str())),
    )?;
    let plot_sample = PlotSample {
        label: sample.label.clone(),
        model: fitted_model,
    };
    plot::write_coverage_plots(std::slice::from_ref(&plot_sample), &paths.svg, &paths.pdf)?;

    eprintln!("wrote {}", paths.summary.display());
    eprintln!("wrote {}", paths.model.display());
    eprintln!("wrote {}", paths.mates.display());
    eprintln!("wrote {}", paths.nonredundant_ids.display());
    eprintln!("wrote {}", paths.svg.display());
    eprintln!("wrote {}", paths.pdf.display());
    Ok(plot_sample)
}

fn resolve_filter_params(args: &RunConfig, filter_fragment_len: usize) -> ResolvedFilterParams {
    let recommended = sketch_params::recommended_filter_params(
        args.p_value,
        args.kmer,
        args.identity,
        filter_fragment_len,
        args.reference_size,
    );
    let requested_minimizer_window = args.minimizer_window;
    let window = requested_minimizer_window.unwrap_or(recommended.minimizer_window);
    let actual_minimizer_window = simd_compatible_window_size(args.kmer, window.max(1));
    let requested_min_shared_minimizers = args.min_shared_minimizers;
    let actual_min_shared_minimizers =
        requested_min_shared_minimizers.unwrap_or(recommended.min_shared_minimizers.max(1));

    ResolvedFilterParams {
        recommended,
        filter_fragment_len,
        requested_minimizer_window,
        actual_minimizer_window,
        requested_min_shared_minimizers,
        actual_min_shared_minimizers,
    }
}

fn infer_filter_fragment_len(read_count: usize, total_bases: usize, kmer: usize) -> usize {
    if read_count == 0 {
        return kmer.saturating_add(1).max(1);
    }
    ((total_bases as f64 / read_count as f64).round() as usize)
        .max(kmer.saturating_add(1))
        .max(1)
}

fn load_sample_inputs(args: &RunConfig) -> Result<Vec<SampleInput>> {
    if let Some(list) = &args.list {
        let text = fs::read_to_string(list)
            .with_context(|| format!("failed to read {}", list.display()))?;
        let mut samples = Vec::new();
        for (line_number, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let fields = line.split_whitespace().collect::<Vec<_>>();
            let path = PathBuf::from(fields[0]);
            let label = fields
                .get(1)
                .map(|value| sanitize_label(value))
                .unwrap_or_else(|| infer_sample_label(&path));
            if label.is_empty() {
                anyhow::bail!(
                    "empty sample label on {}:{}",
                    list.display(),
                    line_number + 1
                );
            }
            samples.push(SampleInput { path, label });
        }
        anyhow::ensure!(
            !samples.is_empty(),
            "{} contains no samples",
            list.display()
        );
        return Ok(samples);
    }

    let path = args
        .input
        .clone()
        .context("--input is required unless --list is provided")?;
    let label = infer_sample_label(&path);
    Ok(vec![SampleInput { path, label }])
}

fn infer_sample_label(path: &Path) -> String {
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("sample");
    let name = name
        .strip_suffix(".fastq.gz")
        .or_else(|| name.strip_suffix(".fq.gz"))
        .or_else(|| name.strip_suffix(".fasta.gz"))
        .or_else(|| name.strip_suffix(".fa.gz"))
        .or_else(|| name.strip_suffix(".fastq"))
        .or_else(|| name.strip_suffix(".fq"))
        .or_else(|| name.strip_suffix(".fasta"))
        .or_else(|| name.strip_suffix(".fa"))
        .unwrap_or(name);
    sanitize_label(name)
}

fn sanitize_label(label: &str) -> String {
    label
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_') {
                ch
            } else {
                '_'
            }
        })
        .collect::<String>()
        .trim_matches('_')
        .to_string()
}

fn prefix_with_sample_label(prefix: &Path, label: &str) -> PathBuf {
    let sanitized = sanitize_label(label);
    let stem = prefix
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("fastcover");
    let mut out = prefix.to_path_buf();
    out.set_file_name(format!("{stem}.{sanitized}"));
    out
}

fn build_lookup(
    forward_seeds: &[Vec<Seed>],
    reverse_seeds: &[Vec<Seed>],
) -> FxHashMap<u64, Vec<IndexHit>> {
    let mut lookup: FxHashMap<u64, Vec<IndexHit>> = FxHashMap::default();
    for (target, seeds) in forward_seeds.iter().enumerate() {
        for seed in seeds {
            lookup.entry(seed.hash).or_default().push(IndexHit {
                target,
                pos: seed.pos,
                rev: false,
            });
        }
    }
    for (target, seeds) in reverse_seeds.iter().enumerate() {
        for seed in seeds {
            lookup.entry(seed.hash).or_default().push(IndexHit {
                target,
                pos: seed.pos,
                rev: true,
            });
        }
    }
    lookup
}

fn score_query(
    query: usize,
    seeds: &[Seed],
    lookup: &FxHashMap<u64, Vec<IndexHit>>,
    reads: &[ReadRecord],
    reverse_complements: &[Vec<u8>],
    forward_seeds: &[Vec<Seed>],
    reverse_seeds: &[Vec<Seed>],
    forward_nt4: &[Vec<u8>],
    reverse_nt4: &[Vec<u8>],
    args: &RunConfig,
    filter_params: &ResolvedFilterParams,
) -> (MateResult, RunStats) {
    let unique = unique_query_seeds(seeds);
    if unique.len() < filter_params.actual_min_shared_minimizers {
        return (
            MateResult {
                query,
                ..MateResult::default()
            },
            RunStats::default(),
        );
    }

    let candidates = candidates_for_query(query, &unique, lookup, args, filter_params);
    let mut stats = RunStats {
        candidate_edges: candidates.len(),
        ..RunStats::default()
    };
    let query_seq = &reads[query].seq;
    let mut passing_targets = Vec::new();
    let mut best: Option<BestAlignmentChoice> = None;
    let mut aligned_count = 0usize;
    let final_alignment_enabled = args.alignment_targets > 0;
    let mut scored_candidates = Vec::new();

    for candidate in &candidates {
        let target_seq = if candidate.rev {
            &reverse_complements[candidate.target]
        } else {
            &reads[candidate.target].seq
        };
        let target_seeds = if candidate.rev {
            &reverse_seeds[candidate.target]
        } else {
            &forward_seeds[candidate.target]
        };
        let (window, evaluated) =
            best_window(query_seq, target_seq, seeds, target_seeds, candidate, args);
        stats.evaluated_windows += evaluated;
        let Some(window) = window else {
            continue;
        };

        let cheap_passes = window.identity >= args.identity
            && window.alignment_ratio >= args.min_alignment_ratio
            && window.query_coverage >= args.min_query_coverage;
        let sketch_alignment = AlignmentStats {
            target: candidate.target,
            identity: window.identity,
            alignment_ratio: window.alignment_ratio,
            query_coverage: window.query_coverage,
            matches: window.shared,
            mismatches: window.denominator.saturating_sub(window.shared),
            insertions: 0,
            deletions: 0,
            score: (window.identity * 1_000_000.0).round() as i32
                + candidate.hits.min(10_000) as i32,
            passes: cheap_passes,
        };

        if final_alignment_enabled {
            if cheap_passes {
                scored_candidates.push(ScoredCandidate {
                    target: candidate.target,
                    rev: candidate.rev,
                    hits: candidate.hits,
                    window,
                });
            }
        } else {
            aligned_count += 1;
            let tie_key = alignment_tie_key(args.seed, query, candidate.target, candidate.rev);
            if best
                .as_ref()
                .map(|old| is_better_alignment_choice(&sketch_alignment, tie_key, old))
                .unwrap_or(true)
            {
                best = Some(BestAlignmentChoice {
                    alignment: sketch_alignment,
                    tie_key,
                });
            }
        }
    }

    if final_alignment_enabled {
        scored_candidates.sort_by(compare_scored_candidates);
        let final_limit = args.alignment_targets.min(args.prefilter_targets).min(16);
        let query_nt4 = &forward_nt4[query];
        for candidate in scored_candidates.iter().take(final_limit) {
            let target_nt4 = if candidate.rev {
                &reverse_nt4[candidate.target]
            } else {
                &forward_nt4[candidate.target]
            };
            aligned_count += 1;
            stats.final_alignments += 1;
            let Some(final_stats) = final_align::rammap_semiglobal_overlap_nt4(
                query_nt4,
                target_nt4,
                query_seq.len(),
                reads[candidate.target].len(),
                candidate.window.q_start,
                candidate.window.t_start,
                candidate.window.overlap_len,
                args.final_bandwidth,
            ) else {
                continue;
            };

            let passes = final_stats.identity >= args.identity
                && final_stats.alignment_ratio >= args.min_alignment_ratio
                && final_stats.query_coverage >= args.min_query_coverage;
            let alignment = AlignmentStats {
                target: candidate.target,
                identity: final_stats.identity,
                alignment_ratio: final_stats.alignment_ratio,
                query_coverage: final_stats.query_coverage,
                matches: final_stats.matches,
                mismatches: final_stats.mismatches,
                insertions: final_stats.insertions,
                deletions: final_stats.deletions,
                score: final_stats.score + candidate.hits.min(10_000) as i32,
                passes,
            };
            let tie_key = alignment_tie_key(args.seed, query, candidate.target, candidate.rev);
            if best
                .as_ref()
                .map(|old| is_better_alignment_choice(&alignment, tie_key, old))
                .unwrap_or(true)
            {
                best = Some(BestAlignmentChoice { alignment, tie_key });
            }
        }
    }

    let best = best.map(|choice| choice.alignment);
    if let Some(best_alignment) = &best {
        if best_alignment.passes {
            stats.passing_best_alignments += 1;
            passing_targets.push(best_alignment.target);
        }
    }
    (
        MateResult {
            query,
            mate_count: passing_targets.len() as u32,
            candidates: candidates.len(),
            aligned: aligned_count,
            best,
            passing_targets,
        },
        stats,
    )
}

fn candidates_for_query(
    query: usize,
    seeds: &[Seed],
    lookup: &FxHashMap<u64, Vec<IndexHit>>,
    args: &RunConfig,
    filter_params: &ResolvedFilterParams,
) -> Vec<SlideCandidate> {
    let mut bands: FxHashMap<BandKey, BandAccum> = FxHashMap::default();
    let diag_bin_size = args.diag_bin.max(1) as isize;
    for seed in seeds {
        let Some(hits) = lookup.get(&seed.hash) else {
            continue;
        };
        if hits.len() > args.max_hash_occ {
            continue;
        }
        for hit in hits {
            if hit.target == query {
                continue;
            }
            let diag = hit.pos as isize - seed.pos as isize;
            let key = BandKey {
                target: hit.target,
                rev: hit.rev,
                diag_bin: div_floor_isize(diag, diag_bin_size),
            };
            let band = bands.entry(key).or_insert_with(|| BandAccum {
                min_diag: diag,
                max_diag: diag,
                ..BandAccum::default()
            });
            band.hits += 1;
            band.sum_diag += diag as i128;
            band.min_diag = band.min_diag.min(diag);
            band.max_diag = band.max_diag.max(diag);
            band.query_positions.push(seed.pos);
        }
    }

    let mut best_by_target_orientation: FxHashMap<(usize, bool), SlideCandidate> =
        FxHashMap::default();
    for (key, mut band) in bands {
        band.query_positions.sort_unstable();
        band.query_positions.dedup();
        if band.query_positions.len() < filter_params.actual_min_shared_minimizers {
            continue;
        }
        let candidate = SlideCandidate {
            target: key.target,
            rev: key.rev,
            center_diag: (band.sum_diag / band.hits.max(1) as i128) as isize,
            min_diag: band.min_diag,
            max_diag: band.max_diag,
            hits: band.query_positions.len(),
        };
        best_by_target_orientation
            .entry((candidate.target, candidate.rev))
            .and_modify(|old| {
                if is_better_candidate(&candidate, old) {
                    *old = candidate.clone();
                }
            })
            .or_insert(candidate);
    }

    let mut out = best_by_target_orientation.into_values().collect::<Vec<_>>();
    out.sort_by(|a, b| {
        b.hits
            .cmp(&a.hits)
            .then_with(|| a.target.cmp(&b.target))
            .then_with(|| a.rev.cmp(&b.rev))
    });
    out.truncate(args.prefilter_targets);
    out
}

fn is_better_candidate(candidate: &SlideCandidate, current: &SlideCandidate) -> bool {
    if candidate.hits != current.hits {
        return candidate.hits > current.hits;
    }
    let cand_width = candidate.max_diag.abs_diff(candidate.min_diag);
    let curr_width = current.max_diag.abs_diff(current.min_diag);
    cand_width < curr_width
}

fn best_window(
    query_seq: &[u8],
    target_seq: &[u8],
    query_seeds: &[Seed],
    target_seeds: &[Seed],
    candidate: &SlideCandidate,
    args: &RunConfig,
) -> (Option<WindowScore>, usize) {
    let mut offsets = candidate_offsets(candidate, args);
    offsets.sort_unstable();
    offsets.dedup();

    let mut best = None;
    let mut evaluated = 0usize;
    for offset in offsets {
        let Some((q_start, t_start, overlap_len)) =
            overlap_for_offset(query_seq.len(), target_seq.len(), offset)
        else {
            continue;
        };
        let alignment_ratio =
            overlap_len as f64 / query_seq.len().min(target_seq.len()).max(1) as f64;
        let query_coverage = overlap_len as f64 / query_seq.len().max(1) as f64;
        if alignment_ratio < args.min_alignment_ratio || query_coverage < args.min_query_coverage {
            continue;
        }
        evaluated += 1;
        let sketch = minimizer_window_sketch_identity(
            query_seeds,
            target_seeds,
            q_start,
            t_start,
            overlap_len,
            args.kmer,
            args.sketch_size,
        );
        let window = WindowScore {
            q_start,
            t_start,
            offset,
            overlap_len,
            alignment_ratio,
            query_coverage,
            identity: sketch.identity,
            shared: sketch.shared,
            denominator: sketch.denominator,
        };
        if best
            .as_ref()
            .map(|old| is_better_window(&window, old))
            .unwrap_or(true)
        {
            best = Some(window);
        }
    }
    (best, evaluated)
}

fn candidate_offsets(candidate: &SlideCandidate, args: &RunConfig) -> Vec<isize> {
    let radius = args.slide_radius as isize;
    let step = args.slide_step.max(1) as isize;
    let start = candidate
        .min_diag
        .min(candidate.center_diag - radius)
        .min(candidate.max_diag);
    let end = candidate
        .max_diag
        .max(candidate.center_diag + radius)
        .max(candidate.min_diag);
    let mut offsets = vec![
        candidate.center_diag,
        candidate.min_diag,
        candidate.max_diag,
    ];
    let mut current = start;
    while current <= end {
        offsets.push(current);
        current = current.saturating_add(step);
        if current == isize::MAX {
            break;
        }
    }
    offsets
}

fn overlap_for_offset(
    query_len: usize,
    target_len: usize,
    offset: isize,
) -> Option<(usize, usize, usize)> {
    let q_start = if offset < 0 { offset.unsigned_abs() } else { 0 };
    let t_start = if offset > 0 { offset as usize } else { 0 };
    if q_start >= query_len || t_start >= target_len {
        return None;
    }
    let overlap_len = (query_len - q_start).min(target_len - t_start);
    (overlap_len > 0).then_some((q_start, t_start, overlap_len))
}

fn minimizer_window_sketch_identity(
    query_seeds: &[Seed],
    target_seeds: &[Seed],
    q_start: usize,
    t_start: usize,
    overlap_len: usize,
    k: usize,
    sketch_size: usize,
) -> SketchIdentity {
    if k == 0 || sketch_size == 0 {
        return SketchIdentity::default();
    }
    let query_hashes = sorted_window_hashes(query_seeds, q_start, q_start + overlap_len);
    let target_hashes = sorted_window_hashes(target_seeds, t_start, t_start + overlap_len);
    if query_hashes.is_empty() || target_hashes.is_empty() {
        return SketchIdentity::default();
    }

    let mut i = 0usize;
    let mut j = 0usize;
    let mut denominator = 0usize;
    let mut shared = 0usize;
    while denominator < sketch_size && (i < query_hashes.len() || j < target_hashes.len()) {
        denominator += 1;
        if i < query_hashes.len()
            && (j == target_hashes.len() || query_hashes[i] < target_hashes[j])
        {
            i += 1;
        } else if j < target_hashes.len()
            && (i == query_hashes.len() || target_hashes[j] < query_hashes[i])
        {
            j += 1;
        } else {
            shared += 1;
            i += 1;
            j += 1;
        }
    }

    let jaccard = if denominator == 0 {
        0.0
    } else {
        shared as f64 / denominator as f64
    };
    SketchIdentity {
        identity: jaccard_to_identity(jaccard, k),
        shared,
        denominator,
    }
}

fn sorted_window_hashes(seeds: &[Seed], start: usize, end: usize) -> Vec<u64> {
    let mut hashes = seeds
        .iter()
        .filter(|seed| seed.pos >= start && seed.pos < end)
        .map(|seed| seed.hash)
        .collect::<Vec<_>>();
    hashes.sort_unstable();
    hashes.dedup();
    hashes
}

fn is_better_window(candidate: &WindowScore, current: &WindowScore) -> bool {
    if candidate.identity != current.identity {
        return candidate.identity > current.identity;
    }
    if candidate.alignment_ratio != current.alignment_ratio {
        return candidate.alignment_ratio > current.alignment_ratio;
    }
    if candidate.overlap_len != current.overlap_len {
        return candidate.overlap_len > current.overlap_len;
    }
    candidate.offset.abs() < current.offset.abs()
}

fn compare_scored_candidates(a: &ScoredCandidate, b: &ScoredCandidate) -> std::cmp::Ordering {
    b.window
        .identity
        .total_cmp(&a.window.identity)
        .then_with(|| b.window.query_coverage.total_cmp(&a.window.query_coverage))
        .then_with(|| {
            b.window
                .alignment_ratio
                .total_cmp(&a.window.alignment_ratio)
        })
        .then_with(|| b.window.overlap_len.cmp(&a.window.overlap_len))
        .then_with(|| b.hits.cmp(&a.hits))
        .then_with(|| a.target.cmp(&b.target))
        .then_with(|| a.rev.cmp(&b.rev))
}

fn is_better_alignment_choice(
    candidate: &AlignmentStats,
    candidate_tie_key: u64,
    current: &BestAlignmentChoice,
) -> bool {
    let current_alignment = &current.alignment;
    if candidate.identity != current_alignment.identity {
        return candidate.identity > current_alignment.identity;
    }
    if candidate.query_coverage != current_alignment.query_coverage {
        return candidate.query_coverage > current_alignment.query_coverage;
    }
    if candidate.alignment_ratio != current_alignment.alignment_ratio {
        return candidate.alignment_ratio > current_alignment.alignment_ratio;
    }
    if candidate.matches != current_alignment.matches {
        return candidate.matches > current_alignment.matches;
    }
    candidate.score > current_alignment.score
        || (candidate.score == current_alignment.score
            && candidate.target != current_alignment.target
            && candidate_tie_key > current.tie_key)
}

fn alignment_tie_key(seed: u64, query: usize, target: usize, rev: bool) -> u64 {
    let mut state = seed ^ 0xD6E8_FD9D_31A4_82C1;
    state ^= (query as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    state ^= (target as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    state ^= if rev {
        0x94D0_49BB_1331_11EB
    } else {
        0x2545_F491_4F6C_DD1D
    };
    splitmix64_permute(state)
}

fn simd_twisted_minimizers(seq: &[u8], k: usize, w: usize, tab_hasher: &Tab64Twisted) -> Vec<Seed> {
    if seq.len() < k.saturating_add(w).saturating_sub(1) {
        return Vec::new();
    }

    let mut result = Vec::new();
    let mut run_start = 0usize;
    while run_start < seq.len() {
        while run_start < seq.len() && !is_acgt(seq[run_start]) {
            run_start += 1;
        }
        if run_start >= seq.len() {
            break;
        }

        let mut run_end = run_start;
        while run_end < seq.len() && is_acgt(seq[run_end]) {
            run_end += 1;
        }

        if run_end - run_start >= k + w - 1 {
            let packed = PackedSeqVec::from_ascii(&seq[run_start..run_end]);
            let mut minimizer_positions = Vec::new();
            let output = simd_minimizers::canonical_minimizers(k, w)
                .run(packed.as_slice(), &mut minimizer_positions);
            for (pos, canonical_value) in output.pos_and_values_u64() {
                result.push(Seed {
                    hash: minimizer_token(canonical_value, k, tab_hasher),
                    pos: run_start + pos as usize,
                });
            }
        }

        run_start = run_end;
    }

    result.sort_unstable_by_key(|seed| (seed.hash, seed.pos));
    result.dedup_by_key(|seed| (seed.hash, seed.pos));
    result
}

fn unique_query_seeds(seeds: &[Seed]) -> Vec<Seed> {
    let mut seen = FxHashSet::default();
    let mut unique = Vec::new();
    for seed in seeds {
        if seen.insert(seed.hash) {
            unique.push(seed.clone());
        }
    }
    unique
}

fn minimizer_token(canonical_kmer_value: u64, k: usize, tab_hasher: &Tab64Twisted) -> u64 {
    let key = canonical_kmer_value ^ ((k as u64) << 56) ^ 0xD1B5_4A32_D192_ED03;
    tab_hasher.hash(key)
}

fn deterministic_tab64_twisted(seed: u64) -> Tab64Twisted {
    let mut state = seed ^ 0xA076_1D64_78BD_642F;
    let mut table = [[0u128; 256]; 8];
    for row in &mut table {
        for value in row {
            let hi = splitmix64_next(&mut state) as u128;
            let lo = splitmix64_next(&mut state) as u128;
            *value = (hi << 64) | lo;
        }
    }
    Tab64Twisted::with_table(table)
}

fn splitmix64_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    splitmix64_permute(*state)
}

fn splitmix64_permute(x: u64) -> u64 {
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn jaccard_to_identity(jaccard: f64, k: usize) -> f64 {
    if jaccard <= 0.0 || k == 0 {
        return 0.0;
    }
    if jaccard >= 1.0 {
        return 1.0;
    }
    let shared_kmer_probability = (2.0 * jaccard) / (1.0 + jaccard);
    let distance = (-shared_kmer_probability.ln() / k as f64).clamp(0.0, 1.0);
    (1.0 - distance).clamp(0.0, 1.0)
}

fn reverse_complement(seq: &[u8]) -> Vec<u8> {
    seq.iter()
        .rev()
        .map(|&base| match base {
            b'A' | b'a' => b'T',
            b'C' | b'c' => b'G',
            b'G' | b'g' => b'C',
            b'T' | b't' | b'U' | b'u' => b'A',
            _ => b'N',
        })
        .collect()
}

fn is_acgt(base: u8) -> bool {
    matches!(
        base,
        b'A' | b'a' | b'C' | b'c' | b'G' | b'g' | b'T' | b't' | b'U' | b'u'
    )
}

fn simd_compatible_window_size(k: usize, w: usize) -> usize {
    if (k + w - 1) % 2 == 1 {
        w
    } else if w > 1 {
        w - 1
    } else {
        w + 1
    }
}

fn div_floor_isize(value: isize, divisor: isize) -> isize {
    let quotient = value / divisor;
    let remainder = value % divisor;
    if remainder != 0 && ((remainder > 0) != (divisor > 0)) {
        quotient - 1
    } else {
        quotient
    }
}

fn write_summary(
    path: &Path,
    summaries: &[SampleSummary],
    args: &RunConfig,
    sample_label: &str,
    filter_params: &ResolvedFilterParams,
    stats: &RunStats,
    total_reads: usize,
    total_bases: usize,
) -> Result<()> {
    let file =
        File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    let mut w = BufWriter::new(file);
    writeln!(w, "# @impl: FastCover")?;
    writeln!(w, "# @version: {}", env!("CARGO_PKG_VERSION"))?;
    writeln!(w, "# @sample: {sample_label}")?;
    writeln!(w, "# @reads: {total_reads}")?;
    writeln!(w, "# @bases: {total_bases}")?;
    writeln!(
        w,
        "# @candidate_source: shared_simd_minimizer_diagonal_bands"
    )?;
    writeln!(w, "# @overlap_source: de_novo_best_sliding_offset")?;
    writeln!(
        w,
        "# @prefilter_identity_source: twisted_tabulation_minimizer_window_bottom_k"
    )?;
    let identity_source = if args.alignment_targets > 0 {
        "rammap_core_dp_align_semiglobal"
    } else {
        "twisted_tabulation_minimizer_window_bottom_k"
    };
    writeln!(w, "# @identity_source: {identity_source}")?;
    writeln!(
        w,
        "# @final_alignment_source: {}",
        if args.alignment_targets > 0 {
            "rammap_core_dp_align_semiglobal"
        } else {
            "disabled"
        }
    )?;
    writeln!(
        w,
        "# @alignment_ratio_definition: overlap_len/min(query_len,target_len)"
    )?;
    writeln!(w, "# @query_coverage_definition: overlap_len/query_len")?;
    writeln!(w, "# @kmer: {}", args.kmer)?;
    writeln!(w, "# @p_value: {:.8e}", args.p_value)?;
    writeln!(
        w,
        "# @filter_fragment_len: {}",
        filter_params.filter_fragment_len
    )?;
    writeln!(w, "# @filter_fragment_len_source: mean_read_len")?;
    writeln!(w, "# @reference_size: {}", args.reference_size)?;
    writeln!(
        w,
        "# @filter_calibration_source: filter_fragment_len_and_reference_size"
    )?;
    writeln!(
        w,
        "# @pvalue_recommended_minimizer_window: {}",
        filter_params.recommended.minimizer_window
    )?;
    writeln!(
        w,
        "# @pvalue_recommended_sketch_size: {}",
        filter_params.recommended.sketch_size
    )?;
    writeln!(
        w,
        "# @pvalue_recommended_min_shared_minimizers: {}",
        filter_params.recommended.min_shared_minimizers
    )?;
    writeln!(
        w,
        "# @pvalue_at_recommended_sketch_size: {:.8e}",
        filter_params.recommended.p_value
    )?;
    let requested_window = filter_params
        .requested_minimizer_window
        .map(|value| value.to_string())
        .unwrap_or_else(|| "auto".to_string());
    writeln!(w, "# @requested_minimizer_window: {requested_window}")?;
    writeln!(
        w,
        "# @actual_minimizer_window: {}",
        filter_params.actual_minimizer_window
    )?;
    writeln!(w, "# @sketch_size: {}", args.sketch_size)?;
    writeln!(w, "# @identity_threshold: {:.6}", args.identity)?;
    writeln!(w, "# @min_alignment_ratio: {:.5}", args.min_alignment_ratio)?;
    writeln!(w, "# @min_query_coverage: {:.5}", args.min_query_coverage)?;
    writeln!(
        w,
        "# @requested_min_shared_minimizers: {}",
        filter_params
            .requested_min_shared_minimizers
            .map(|value| value.to_string())
            .unwrap_or_else(|| "auto".to_string())
    )?;
    writeln!(
        w,
        "# @actual_min_shared_minimizers: {}",
        filter_params.actual_min_shared_minimizers
    )?;
    writeln!(w, "# @prefilter_targets: {}", args.prefilter_targets)?;
    writeln!(w, "# @alignment_targets: {}", args.alignment_targets)?;
    writeln!(w, "# @alignment_targets_cap: 16")?;
    writeln!(w, "# @final_bandwidth: {}", args.final_bandwidth)?;
    writeln!(w, "# @max_hash_occ: {}", args.max_hash_occ)?;
    writeln!(w, "# @diag_bin: {}", args.diag_bin)?;
    writeln!(w, "# @slide_radius: {}", args.slide_radius)?;
    writeln!(w, "# @slide_step: {}", args.slide_step)?;
    writeln!(w, "# @tabulation: twisted")?;
    writeln!(w, "# @tab_hash_seed: {}", args.tab_hash_seed)?;
    writeln!(w, "# @candidate_edges: {}", stats.candidate_edges)?;
    writeln!(w, "# @evaluated_windows: {}", stats.evaluated_windows)?;
    writeln!(w, "# @final_alignments: {}", stats.final_alignments)?;
    writeln!(w, "# @best_mate_only: true")?;
    writeln!(
        w,
        "# @mate_count_definition: nonself_selected_best_mate_count"
    )?;
    writeln!(w, "# @resampling_effort_unit: bases")?;
    writeln!(w, "# @resampling_scheme: bernoulli_reads_by_base_fraction")?;
    writeln!(w, "# @redundancy_weighting: query_bases")?;
    writeln!(
        w,
        "# @passing_best_alignments: {}",
        stats.passing_best_alignments
    )?;
    writeln!(w, "# @replicates: {}", args.replicates)?;
    writeln!(w, "# @divide: {:.5}", args.divide)?;
    writeln!(w, "# @seed: {}", args.seed)?;
    writeln!(w, "reads\tbases\tportion\tmean\tsd\tq1\tmedian\tq3")?;
    for s in summaries {
        writeln!(
            w,
            "{}\t{}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}",
            s.reads, s.bases, s.portion, s.mean, s.sd, s.q1, s.median, s.q3
        )?;
    }
    Ok(())
}

fn write_all(path: &Path, all_values: &[(f64, u64, usize, f64)]) -> Result<()> {
    let file =
        File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    let mut w = BufWriter::new(file);
    writeln!(w, "portion\tbases\treplicate\tredundant_base_fraction")?;
    for (portion, bases, rep, value) in all_values {
        writeln!(w, "{portion:.8}\t{bases}\t{rep}\t{value:.8}")?;
    }
    Ok(())
}

fn write_mates(
    path: &Path,
    reads: &[ReadRecord],
    mates: &[MateResult],
    keep: &[bool],
) -> Result<()> {
    let file =
        File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    let mut w = BufWriter::new(file);
    writeln!(
        w,
        "query_id\tquery_len\tmate_count\tcandidate_count\taligned_count\tbest_target\tbest_identity\tbest_alignment_ratio\tbest_query_coverage\tbest_score\tbest_matches\tbest_mismatches\tnonredundant"
    )?;
    for mate in mates {
        let read = &reads[mate.query];
        if let Some(best) = &mate.best {
            writeln!(
                w,
                "{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{}\t{}\t{}\t{}",
                read.id,
                read.len(),
                mate.mate_count,
                mate.candidates,
                mate.aligned,
                reads[best.target].id,
                best.identity,
                best.alignment_ratio,
                best.query_coverage,
                best.score,
                best.matches,
                best.mismatches,
                keep[mate.query]
            )?;
        } else {
            writeln!(
                w,
                "{}\t{}\t{}\t{}\t{}\t.\t0\t0\t0\t0\t0\t0\t{}",
                read.id,
                read.len(),
                mate.mate_count,
                mate.candidates,
                mate.aligned,
                keep[mate.query]
            )?;
        }
    }
    Ok(())
}

fn write_ids<'a>(path: &Path, ids: impl Iterator<Item = &'a str>) -> Result<()> {
    let file =
        File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    let mut w = BufWriter::new(file);
    for id in ids {
        writeln!(w, "{id}")?;
    }
    Ok(())
}

fn ensure_parent(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlap_offset_handles_suffix_prefix() {
        let (q, t, len) = overlap_for_offset(100, 80, 50).unwrap();
        assert_eq!((q, t, len), (0, 50, 30));
        let (q, t, len) = overlap_for_offset(100, 80, -40).unwrap();
        assert_eq!((q, t, len), (40, 0, 60));
    }

    #[test]
    fn splitmix_is_deterministic() {
        assert_eq!(splitmix64_permute(42), splitmix64_permute(42));
    }

    #[test]
    fn simd_minimizers_use_positions_and_twisted_hashes() {
        let tab = deterministic_tab64_twisted(42);
        let seeds = simd_twisted_minimizers(b"ACGTGCTCAGAGACTCAGAGGA", 5, 7, &tab);
        let mut positions = seeds.iter().map(|seed| seed.pos).collect::<Vec<_>>();
        positions.sort_unstable();
        assert_eq!(positions, vec![0, 7, 9, 15]);
    }
}
