use std::cell::RefCell;

use rammap::align::dp::{self, APPROX_MAX, CIGAR_DEL, CIGAR_INS, CIGAR_MATCH, DpResult};
use rammap::align::extend::build_scoring_matrix;
use rammap::encode_nt4;

#[derive(Clone, Debug)]
pub struct FinalAlignmentStats {
    pub identity: f64,
    pub alignment_ratio: f64,
    pub query_coverage: f64,
    pub matches: usize,
    pub mismatches: usize,
    pub insertions: usize,
    pub deletions: usize,
    pub score: i32,
}

#[derive(Clone, Debug, Default)]
struct CigarCounts {
    matches: usize,
    mismatches: usize,
    insertions: usize,
    deletions: usize,
}

impl CigarCounts {
    fn block_len(&self) -> usize {
        self.matches + self.mismatches + self.insertions + self.deletions
    }

    fn query_aligned(&self) -> usize {
        self.matches + self.mismatches + self.insertions
    }

    fn target_aligned(&self) -> usize {
        self.matches + self.mismatches + self.deletions
    }
}

struct DpScratch {
    matrix: [i8; 25],
    result: DpResult,
}

impl DpScratch {
    fn new() -> Self {
        Self {
            matrix: build_scoring_matrix(2, 4),
            result: DpResult::default(),
        }
    }

    fn reset_result(&mut self) {
        let mut cigar = std::mem::take(&mut self.result.cigar);
        cigar.clear();
        self.result = DpResult::default();
        self.result.cigar = cigar;
    }
}

thread_local! {
    static DP_SCRATCH: RefCell<DpScratch> = RefCell::new(DpScratch::new());
}

pub fn encode_ascii_nt4(seq: &[u8]) -> Vec<u8> {
    encode_nt4(seq)
}

pub fn rammap_semiglobal_overlap(
    query_seq: &[u8],
    target_seq: &[u8],
    q_start: usize,
    t_start: usize,
    overlap_len: usize,
    bandwidth: i32,
) -> Option<FinalAlignmentStats> {
    let q_end = q_start.checked_add(overlap_len)?;
    let t_end = t_start.checked_add(overlap_len)?;
    if overlap_len == 0 || q_end > query_seq.len() || t_end > target_seq.len() {
        return None;
    }

    let query_nt4 = encode_nt4(&query_seq[q_start..q_end]);
    let target_nt4 = encode_nt4(&target_seq[t_start..t_end]);
    rammap_semiglobal_overlap_nt4(
        &query_nt4,
        &target_nt4,
        query_seq.len(),
        target_seq.len(),
        0,
        0,
        overlap_len,
        bandwidth,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn rammap_semiglobal_overlap_nt4(
    query_nt4: &[u8],
    target_nt4: &[u8],
    query_len: usize,
    target_len: usize,
    q_start: usize,
    t_start: usize,
    overlap_len: usize,
    bandwidth: i32,
) -> Option<FinalAlignmentStats> {
    let q_end = q_start.checked_add(overlap_len)?;
    let t_end = t_start.checked_add(overlap_len)?;
    if overlap_len == 0 || q_end > query_nt4.len() || t_end > target_nt4.len() {
        return None;
    }

    let query_slice = &query_nt4[q_start..q_end];
    let target_slice = &target_nt4[t_start..t_end];
    let (counts, score) = DP_SCRATCH.with(|scratch| {
        let mut scratch = scratch.borrow_mut();
        scratch.reset_result();
        let matrix = scratch.matrix;
        dp::extend_single_affine(
            query_slice,
            target_slice,
            5,
            &matrix,
            4,
            2,
            bandwidth,
            -1,
            0,
            APPROX_MAX,
            &mut scratch.result,
        );
        raw_cigar_counts(&scratch.result.cigar, query_slice, target_slice)
            .map(|counts| (counts, scratch.result.score))
    })?;
    let block_len = counts.block_len();
    if block_len == 0 {
        return None;
    }

    let query_aligned = counts.query_aligned();
    let target_aligned = counts.target_aligned();
    Some(FinalAlignmentStats {
        identity: counts.matches as f64 / block_len as f64,
        alignment_ratio: query_aligned.min(target_aligned) as f64
            / query_len.min(target_len).max(1) as f64,
        query_coverage: query_aligned as f64 / query_len.max(1) as f64,
        matches: counts.matches,
        mismatches: counts.mismatches,
        insertions: counts.insertions,
        deletions: counts.deletions,
        score,
    })
}

fn raw_cigar_counts(cigar: &[u32], query: &[u8], target: &[u8]) -> Option<CigarCounts> {
    let mut counts = CigarCounts::default();
    let mut q_pos = 0usize;
    let mut t_pos = 0usize;

    for &op_len in cigar {
        let len = (op_len >> 4) as usize;
        if len == 0 {
            return None;
        }

        match op_len & 0x0f {
            CIGAR_MATCH => {
                for _ in 0..len {
                    let q = *query.get(q_pos)?;
                    let t = *target.get(t_pos)?;
                    if q == t && q < 4 {
                        counts.matches += 1;
                    } else {
                        counts.mismatches += 1;
                    }
                    q_pos += 1;
                    t_pos += 1;
                }
            }
            CIGAR_INS => {
                q_pos = q_pos.checked_add(len)?;
                if q_pos > query.len() {
                    return None;
                }
                counts.insertions += len;
            }
            CIGAR_DEL => {
                t_pos = t_pos.checked_add(len)?;
                if t_pos > target.len() {
                    return None;
                }
                counts.deletions += len;
            }
            _ => return None,
        }
    }

    Some(counts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_overlap_has_full_identity() {
        let stats = rammap_semiglobal_overlap(b"ACGTACGT", b"ACGTACGT", 0, 0, 8, -1).unwrap();
        assert_eq!(stats.matches, 8);
        assert_eq!(stats.mismatches, 0);
        assert!((stats.identity - 1.0).abs() < f64::EPSILON);
        assert!((stats.query_coverage - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn mismatch_lowers_identity() {
        let stats = rammap_semiglobal_overlap(b"ACGTACGT", b"ACGTTCGT", 0, 0, 8, -1).unwrap();
        assert_eq!(stats.matches, 7);
        assert_eq!(stats.mismatches, 1);
        assert!((stats.identity - 0.875).abs() < f64::EPSILON);
    }
}
