use anyhow::Result;

fn main() -> Result<()> {
    let config = fastcover::cli::parse_args()?;
    fastcover::coverage::run(config)
}
