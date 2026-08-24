use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result};
use needletail::{Sequence, parse_fastx_file};

use crate::types::ReadRecord;

pub fn read_fastx(path: &Path) -> Result<Vec<ReadRecord>> {
    let mut reader = parse_fastx_file(path)
        .with_context(|| format!("failed to open input {}", path.display()))?;
    let mut reads = Vec::new();

    while let Some(record) = reader.next() {
        let record =
            record.with_context(|| format!("invalid FASTA/FASTQ record in {}", path.display()))?;
        let id = String::from_utf8_lossy(record.id()).to_string();
        let seq = record.normalize(false).into_owned();
        if seq.is_empty() {
            continue;
        }
        reads.push(ReadRecord { id, seq });
    }

    anyhow::ensure!(!reads.is_empty(), "no reads found in {}", path.display());
    Ok(reads)
}

pub fn write_ids(path: &Path, ids: impl IntoIterator<Item = String>) -> Result<()> {
    let file =
        File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    for id in ids {
        writeln!(writer, "{id}")?;
    }
    Ok(())
}

pub fn write_representatives_fastq(path: &Path, reads: &[ReadRecord], keep: &[bool]) -> Result<()> {
    let file =
        File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    for (read, &is_kept) in reads.iter().zip(keep) {
        if !is_kept {
            continue;
        }
        writeln!(writer, "@{}", read.id)?;
        writeln!(writer, "{}", String::from_utf8_lossy(&read.seq))?;
        writeln!(writer, "+")?;
        writeln!(writer, "{}", "I".repeat(read.seq.len()))?;
    }
    Ok(())
}
