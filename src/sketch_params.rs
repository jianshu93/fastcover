#[derive(Clone, Copy, Debug)]
pub struct FilterParams {
    pub minimizer_window: usize,
    pub sketch_size: usize,
    pub min_shared_minimizers: usize,
    pub p_value: f64,
}

pub fn recommended_filter_params(
    pvalue_cutoff: f64,
    k: usize,
    identity_fraction: f64,
    fragment_len: usize,
    reference_size: u64,
) -> FilterParams {
    let identity_percent = identity_fraction.clamp(0.0, 1.0) * 100.0;
    let query_len = fragment_len.max(1);
    let reference_size = reference_size.max(1);
    let mut potential = vec![1usize, 2, 5];
    potential.extend((10..query_len).step_by(10));

    let mut best_sketch_size = query_len;
    let mut best_p_value = 1.0;
    for sketch_size in potential {
        let p_value = estimate_pvalue(
            sketch_size,
            k,
            4,
            identity_percent,
            query_len,
            reference_size,
        );
        if p_value <= pvalue_cutoff {
            best_sketch_size = sketch_size;
            best_p_value = p_value;
            break;
        }
    }

    let minimizer_window = ((2 * query_len) / best_sketch_size).clamp(1, query_len);
    let min_shared_minimizers =
        estimate_minimum_hits_relaxed(best_sketch_size, k, identity_percent);

    FilterParams {
        minimizer_window,
        sketch_size: best_sketch_size,
        min_shared_minimizers,
        p_value: best_p_value,
    }
}

pub fn estimate_minimum_hits(sketch_size: usize, k: usize, percent_identity: f64) -> usize {
    let mash_dist = 1.0 - percent_identity / 100.0;
    let jaccard = distance_to_jaccard(mash_dist, k);
    (sketch_size as f64 * jaccard).ceil() as usize
}

pub fn estimate_minimum_hits_relaxed(sketch_size: usize, k: usize, percent_identity: f64) -> usize {
    estimate_minimum_hits_relaxed_with_confidence(sketch_size, k, percent_identity, 0.9)
}

fn estimate_minimum_hits_relaxed_with_confidence(
    sketch_size: usize,
    k: usize,
    percent_identity: f64,
    confidence_interval: f64,
) -> usize {
    if sketch_size == 0 {
        return 0;
    }

    let strict = estimate_minimum_hits(sketch_size, k, percent_identity);
    let mut relaxed = strict;
    for i in (0..=strict).rev() {
        let jaccard = i as f64 / sketch_size as f64;
        let distance = jaccard_to_distance(jaccard, k);
        let distance_lower_bound =
            mash_distance_lower_bound(distance, sketch_size, k, confidence_interval);
        let upper_identity = 100.0 * (1.0 - distance_lower_bound);
        if upper_identity >= percent_identity {
            relaxed = i;
        } else {
            break;
        }
    }
    relaxed
}

pub fn estimate_pvalue(
    sketch_size: usize,
    k: usize,
    alphabet_size: usize,
    identity: f64,
    query_len: usize,
    reference_len: u64,
) -> f64 {
    let kmer_space = (alphabet_size as f64).powi(k as i32);
    let px = 1.0 / (1.0 + kmer_space / query_len.max(1) as f64);
    let py = px;
    let random_jaccard = px * py / (px + py - px * py);
    let x = estimate_minimum_hits_relaxed(sketch_size, k, identity);
    let sf = if x == 0 {
        1.0
    } else {
        binomial_sf(x, sketch_size, random_jaccard)
    };
    reference_len.max(1) as f64 * sf
}

pub fn identity_to_jaccard(identity_fraction: f64, k: usize) -> f64 {
    let distance = 1.0 - identity_fraction.clamp(0.0, 1.0);
    distance_to_jaccard(distance, k)
}

pub fn jaccard_to_identity(jaccard: f64, k: usize) -> f64 {
    1.0 - jaccard_to_distance(jaccard, k)
}

fn distance_to_jaccard(distance: f64, k: usize) -> f64 {
    let distance = distance.clamp(0.0, 1.0);
    let shared_kmer_probability = (-(k as f64) * distance).exp();
    shared_kmer_probability / (2.0 - shared_kmer_probability)
}

fn jaccard_to_distance(jaccard: f64, k: usize) -> f64 {
    if jaccard <= 0.0 {
        1.0
    } else if jaccard >= 1.0 {
        0.0
    } else {
        let shared_kmer_probability = (2.0 * jaccard) / (1.0 + jaccard);
        (-1.0 / k as f64) * shared_kmer_probability.ln()
    }
}

fn mash_distance_lower_bound(
    distance: f64,
    sketch_size: usize,
    k: usize,
    confidence_interval: f64,
) -> f64 {
    if sketch_size == 0 {
        return 1.0;
    }

    let q2 = (1.0 - confidence_interval) / 2.0;
    let p = distance_to_jaccard(distance, k).clamp(0.0, 1.0);
    let mut x = ((sketch_size as f64 * p).ceil() as usize).max(1);

    while x <= sketch_size {
        let sf = binomial_sf(x, sketch_size, p);
        if sf < q2 {
            x = x.saturating_sub(1);
            break;
        }
        x += 1;
    }

    x = x.clamp(1, sketch_size);
    jaccard_to_distance(x as f64 / sketch_size as f64, k)
}

fn binomial_sf(x: usize, n: usize, p: f64) -> f64 {
    if x == 0 {
        return 1.0;
    }
    if x > n || p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return 1.0;
    }

    let ln_p = p.ln();
    let ln_q = (1.0 - p).ln();
    let mut log_pmf = n as f64 * ln_q;
    let mut max_log = f64::NEG_INFINITY;
    let mut logs = Vec::with_capacity(n - x + 1);

    for i in 0..=n {
        if i >= x {
            max_log = max_log.max(log_pmf);
            logs.push(log_pmf);
        }

        if i < n {
            log_pmf += ((n - i) as f64).ln() - ((i + 1) as f64).ln() + ln_p - ln_q;
        }
    }

    let sum = logs
        .into_iter()
        .map(|value| (value - max_log).exp())
        .sum::<f64>();
    (max_log.exp() * sum).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn turboani_default_window_matches_reference_equation() {
        let params = recommended_filter_params(0.001, 16, 0.78, 3000, 5_000_000);
        assert_eq!(params.minimizer_window, 17);
    }

    #[test]
    fn fragment_len_controls_sampling_density() {
        let ten_k = recommended_filter_params(0.001, 16, 0.95, 10_000, 100_000);
        let hundred_k = recommended_filter_params(0.001, 16, 0.95, 100_000, 100_000);
        assert_eq!(ten_k.minimizer_window, 1000);
        assert_eq!(hundred_k.minimizer_window, 10000);
        assert_eq!(ten_k.sketch_size, hundred_k.sketch_size);
    }

    #[test]
    fn reference_size_adjusts_pvalue_without_changing_fragment_length() {
        let params = recommended_filter_params(0.001, 16, 0.95, 10_000, 100_000);
        assert_eq!(params.minimizer_window, 1000);
        assert_eq!(params.sketch_size, 20);
        assert_eq!(params.min_shared_minimizers, 3);
    }
}
