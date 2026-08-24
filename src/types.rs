use std::path::PathBuf;

#[derive(Clone, Debug)]
pub struct ReadRecord {
    pub id: String,
    pub seq: Vec<u8>,
}

impl ReadRecord {
    pub fn len(&self) -> usize {
        self.seq.len()
    }
}

#[derive(Clone, Debug, Default)]
pub struct Candidate {
    pub target: usize,
    pub score: u32,
}

#[derive(Clone, Debug, Default)]
pub struct AlignmentStats {
    pub target: usize,
    pub identity: f64,
    pub alignment_ratio: f64,
    pub query_coverage: f64,
    pub matches: usize,
    pub mismatches: usize,
    pub insertions: usize,
    pub deletions: usize,
    pub score: i32,
    pub passes: bool,
}

#[derive(Clone, Debug, Default)]
pub struct MateResult {
    pub query: usize,
    pub mate_count: u32,
    pub candidates: usize,
    pub aligned: usize,
    pub best: Option<AlignmentStats>,
    pub passing_targets: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct SampleSummary {
    pub reads: u64,
    pub bases: u64,
    pub portion: f64,
    pub mean: f64,
    pub sd: f64,
    pub q1: f64,
    pub median: f64,
    pub q3: f64,
}

#[derive(Clone, Debug)]
pub struct OutputPaths {
    pub summary: PathBuf,
    pub all: PathBuf,
    pub model: PathBuf,
    pub mates: PathBuf,
    pub nonredundant_ids: PathBuf,
    pub representatives: PathBuf,
    pub svg: PathBuf,
    pub pdf: PathBuf,
}

impl OutputPaths {
    pub fn from_prefix(prefix: &std::path::Path) -> Self {
        let s = prefix.to_string_lossy();
        Self {
            summary: PathBuf::from(format!("{s}.summary.tsv")),
            all: PathBuf::from(format!("{s}.all.tsv")),
            model: PathBuf::from(format!("{s}.model.tsv")),
            mates: PathBuf::from(format!("{s}.mates.tsv")),
            nonredundant_ids: PathBuf::from(format!("{s}.nonredundant.ids")),
            representatives: PathBuf::from(format!("{s}.representatives.fastq")),
            svg: PathBuf::from(format!("{s}.svg")),
            pdf: PathBuf::from(format!("{s}.pdf")),
        }
    }
}
