use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::types::{MateResult, SampleSummary};

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
    total_reads: usize,
    replicates: usize,
    divide: f64,
    seed: u64,
) -> (Vec<SampleSummary>, Vec<(f64, usize, f64)>) {
    let points = sampling_points(total_reads, divide);
    let mut summaries = Vec::with_capacity(points.len());
    let mut all = Vec::with_capacity(points.len() * replicates);

    for &portion in &points {
        let mut values = Vec::with_capacity(replicates);
        for rep in 0..replicates {
            let value = sample_once(
                mates,
                total_reads,
                portion,
                seed ^ ((rep as u64) << 32) ^ portion.to_bits(),
            );
            all.push((portion, rep, value));
            values.push(value);
        }
        summaries.push(summarize(portion, total_reads, &mut values));
    }

    (summaries, all)
}

fn sample_once(mates: &[MateResult], total_reads: usize, portion: f64, seed: u64) -> f64 {
    if portion <= 0.0 || mates.is_empty() || total_reads <= 1 {
        return 0.0;
    }

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut sample_size = 0usize;
    let mut found = 0usize;

    for mate in mates {
        if rng.r#gen::<f64>() >= portion {
            continue;
        }
        sample_size += 1;

        let p_gt_0 = 1.0 - (1.0 - portion).powi(mate.mate_count as i32);
        if rng.r#gen::<f64>() < p_gt_0 {
            found += 1;
        }
    }

    if sample_size == 0 {
        0.0
    } else {
        found as f64 / sample_size as f64
    }
}

fn summarize(portion: f64, total_reads: usize, values: &mut [f64]) -> SampleSummary {
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
                ..MateResult::default()
            },
            MateResult {
                query: 1,
                mate_count: 0,
                ..MateResult::default()
            },
        ];
        let redundancy = sample_once(&mates, 2, 1.0, 7);
        assert!((redundancy - 0.5).abs() < f64::EPSILON);
    }
}
