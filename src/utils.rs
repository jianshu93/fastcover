use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use crate::types::{MateResult, SampleSummary};

#[derive(Clone, Copy, Debug)]
struct ResampleValue {
    point_idx: usize,
    portion: f64,
    bases: u64,
    rep: usize,
    value: f64,
}

pub fn sampling_points(total_reads: usize, divide: f64) -> Vec<f64> {
    if total_reads <= 1 {
        return vec![1.0];
    }
    let divide = if divide > 0.0 && divide < 1.0 {
        divide
    } else {
        0.70
    };
    let n = ((2.0_f64.ln() - (total_reads as f64).ln()) / divide.ln()).ceil() as usize + 2;
    let mut points = Vec::with_capacity(n);
    for i in 0..n {
        let p = if i == 0 {
            0.0
        } else {
            divide.powi((n - i - 1) as i32)
        };
        if points
            .last()
            .map(|last: &f64| (p - last).abs() > 1e-12)
            .unwrap_or(true)
        {
            points.push(p.min(1.0));
        }
    }
    if *points.last().unwrap_or(&0.0) < 1.0 {
        points.push(1.0);
    }
    points
}

pub fn sample_curve(
    mates: &[MateResult],
    read_lengths: &[usize],
    replicates: usize,
    divide: f64,
    seed: u64,
) -> (Vec<SampleSummary>, Vec<(f64, u64, usize, f64)>) {
    let total_reads = read_lengths.len();
    let total_bases = read_lengths.iter().sum::<usize>();
    let points = sampling_points(total_reads, divide);
    let job_count = points.len().saturating_mul(replicates);
    let mut resampled = points
        .iter()
        .copied()
        .enumerate()
        .flat_map(|(point_idx, portion)| (0..replicates).map(move |rep| (point_idx, portion, rep)))
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|(point_idx, portion, rep)| ResampleValue {
            point_idx,
            portion,
            bases: (portion * total_bases as f64).round() as u64,
            rep,
            value: sample_once(
                mates,
                read_lengths,
                portion,
                seed ^ ((rep as u64) << 32) ^ portion.to_bits(),
            ),
        })
        .collect::<Vec<_>>();

    resampled.sort_by_key(|value| (value.point_idx, value.rep));

    let mut values_by_point = (0..points.len())
        .map(|_| Vec::with_capacity(replicates))
        .collect::<Vec<_>>();
    let mut all = Vec::with_capacity(job_count);
    for value in resampled {
        values_by_point[value.point_idx].push(value.value);
        all.push((value.portion, value.bases, value.rep, value.value));
    }

    let summaries = points
        .iter()
        .copied()
        .enumerate()
        .map(|(point_idx, portion)| {
            summarize(
                portion,
                total_reads,
                total_bases,
                &mut values_by_point[point_idx],
            )
        })
        .collect::<Vec<_>>();

    (summaries, all)
}

fn sample_once(mates: &[MateResult], read_lengths: &[usize], portion: f64, seed: u64) -> f64 {
    let total_reads = read_lengths.len();
    if portion <= 0.0 || mates.is_empty() || total_reads <= 1 {
        return 0.0;
    }

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut sampled = vec![false; total_reads];
    let mut sampled_bases = 0usize;
    let mut found_bases = 0usize;

    for (idx, &read_len) in read_lengths.iter().enumerate() {
        if rng.r#gen::<f64>() >= portion {
            continue;
        }
        sampled[idx] = true;
        sampled_bases = sampled_bases.saturating_add(read_len);
    }

    for mate in mates {
        if mate.query >= total_reads || !sampled[mate.query] {
            continue;
        }
        if mate
            .passing_targets
            .iter()
            .any(|&target| target < total_reads && sampled[target])
        {
            found_bases = found_bases.saturating_add(read_lengths[mate.query]);
        }
    }

    if sampled_bases == 0 {
        0.0
    } else {
        found_bases as f64 / sampled_bases as f64
    }
}

fn summarize(
    portion: f64,
    total_reads: usize,
    total_bases: usize,
    values: &mut [f64],
) -> SampleSummary {
    values.sort_by(|a, b| a.total_cmp(b));
    let mean = if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    };
    let sd = if values.is_empty() {
        0.0
    } else {
        let x2 = values.iter().map(|x| x * x).sum::<f64>() / values.len() as f64;
        (x2 - mean * mean).max(0.0).sqrt()
    };
    SampleSummary {
        reads: (portion * total_reads as f64).round() as u64,
        bases: (portion * total_bases as f64).round() as u64,
        portion,
        mean,
        sd,
        q1: quantile(values, 0.25),
        median: quantile(values, 0.50),
        q3: quantile(values, 0.75),
    }
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[idx]
}

pub fn greedy_representatives(mates: &[MateResult]) -> Vec<bool> {
    let mut keep = vec![false; mates.len()];
    for mate in mates {
        let covered_by_kept = mate
            .passing_targets
            .iter()
            .any(|&target| target < keep.len() && keep[target]);
        if !covered_by_kept {
            keep[mate.query] = true;
        }
    }
    keep
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn points_reach_one() {
        let points = sampling_points(1000, 0.7);
        assert_eq!(*points.first().unwrap(), 0.0);
        assert_eq!(*points.last().unwrap(), 1.0);
    }

    #[test]
    fn full_sample_recovers_binary_best_mates() {
        let mates = vec![
            MateResult {
                query: 0,
                mate_count: 1,
                passing_targets: vec![1],
                ..MateResult::default()
            },
            MateResult {
                query: 1,
                mate_count: 0,
                ..MateResult::default()
            },
        ];
        let read_lengths = vec![10, 30];
        let redundancy = sample_once(&mates, &read_lengths, 1.0, 7);
        assert!((redundancy - 0.25).abs() < f64::EPSILON);
    }

    #[test]
    fn parallel_sample_curve_is_deterministic() {
        let mates = (0..64)
            .map(|query| MateResult {
                query,
                mate_count: (query % 3 != 0) as u32,
                passing_targets: (query % 3 != 0)
                    .then_some((query + 1) % 64)
                    .into_iter()
                    .collect(),
                ..MateResult::default()
            })
            .collect::<Vec<_>>();
        let read_lengths = (0..64).map(|idx| 1000 + idx).collect::<Vec<_>>();

        let (summary_a, all_a) = sample_curve(&mates, &read_lengths, 8, 0.7, 11);
        let (summary_b, all_b) = sample_curve(&mates, &read_lengths, 8, 0.7, 11);

        assert_eq!(all_a, all_b);
        assert_eq!(summary_a.len(), summary_b.len());
        for (a, b) in summary_a.iter().zip(summary_b) {
            assert_eq!(a.reads, b.reads);
            assert_eq!(a.bases, b.bases);
            assert_eq!(a.portion, b.portion);
            assert_eq!(a.mean, b.mean);
            assert_eq!(a.sd, b.sd);
            assert_eq!(a.q1, b.q1);
            assert_eq!(a.median, b.median);
            assert_eq!(a.q3, b.q3);
        }
    }
}
