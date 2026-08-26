use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use plotters::prelude::*;

use crate::coverage_model::{self, CoverageModel};

const MIN_PLOT_EFFORT_BP: f64 = 1.0e6;
const PLOT_WIDTH: u32 = 1040;
const PLOT_HEIGHT: u32 = 450;
const COVERAGE_PANEL_FRACTION: f64 = 0.60;

#[derive(Clone, Debug)]
pub struct PlotSample {
    pub label: String,
    pub model: CoverageModel,
}

#[derive(Clone, Copy, Debug)]
struct CoverageGuide {
    current_effort: f64,
    observed_coverage: f64,
    matched_coverage: f64,
    color: RGBColor,
}

pub fn write_coverage_plots(
    samples: &[PlotSample],
    svg_path: &Path,
    pdf_path: &Path,
) -> Result<()> {
    let svg = render_coverage_svg(samples)?;
    write_vector_plot(&svg, svg_path, pdf_path)
}

fn render_coverage_svg(samples: &[PlotSample]) -> Result<String> {
    anyhow::ensure!(!samples.is_empty(), "no samples to plot");
    let mut svg = String::new();
    {
        let root = SVGBackend::with_string(&mut svg, (PLOT_WIDTH, PLOT_HEIGHT)).into_drawing_area();
        root.fill(&WHITE)
            .map_err(|e| anyhow::anyhow!("failed to initialize SVG drawing area: {e:?}"))?;
        let coverage_panel_width = (PLOT_WIDTH as f64 * COVERAGE_PANEL_FRACTION).round() as u32;
        let (coverage_area, diversity_area) = root.split_horizontally(coverage_panel_width);

        let min_x = samples
            .iter()
            .flat_map(|sample| sample.model.points.iter())
            .filter(|p| p.adjusted_effort > 0.0)
            .map(|p| p.adjusted_effort)
            .fold(f64::INFINITY, f64::min)
            .max(MIN_PLOT_EFFORT_BP);
        let min_x = if min_x.is_finite() {
            min_x
        } else {
            MIN_PLOT_EFFORT_BP
        };
        let observed_max_x = samples
            .iter()
            .flat_map(|sample| sample.model.points.iter())
            .map(|p| p.adjusted_effort)
            .fold(1.0_f64, f64::max);
        let model_max_x = samples
            .iter()
            .flat_map(|sample| sample.model.curve.iter())
            .map(|p| p.adjusted_effort)
            .fold(1.0_f64, f64::max);
        let max_x = observed_max_x
            .max(model_max_x)
            .max(coverage_model::DEFAULT_MODEL_CURVE_MAX_EFFORT_BP)
            .max(min_x * 1.01);
        let guides = samples
            .iter()
            .enumerate()
            .map(|(idx, sample)| coverage_guide(sample, sample_color(idx), min_x, max_x))
            .collect::<Vec<_>>();

        let mut chart = ChartBuilder::on(&coverage_area)
            .margin(20)
            .caption("Stevens Coverage Curve", ("sans-serif", 26))
            .x_label_area_size(62)
            .y_label_area_size(78)
            .build_cartesian_2d((min_x..(max_x * 1.02)).log_scale(), 0f64..1.0)
            .map_err(|e| anyhow::anyhow!("failed to build chart: {e:?}"))?;

        chart
            .configure_mesh()
            .x_desc("Adjusted sequencing effort (bp)")
            .y_desc("Coverage")
            .x_label_formatter(&scientific_label)
            .axis_desc_style(("sans-serif", 24))
            .label_style(("sans-serif", 20))
            .x_label_offset(2)
            .disable_mesh()
            .draw()
            .map_err(|e| anyhow::anyhow!("failed to draw chart mesh: {e:?}"))?;

        let single_sample = samples.len() == 1;
        for (idx, sample) in samples.iter().enumerate() {
            let color = sample_color(idx);
            let positive_points = sample
                .model
                .points
                .iter()
                .filter(|p| p.adjusted_effort > 0.0)
                .collect::<Vec<_>>();
            if single_sample {
                let q1 = positive_points
                    .iter()
                    .map(|p| (p.adjusted_effort, p.q1_coverage))
                    .collect::<Vec<_>>();
                let q3 = positive_points
                    .iter()
                    .map(|p| (p.adjusted_effort, p.q3_coverage))
                    .collect::<Vec<_>>();
                chart
                    .draw_series(LineSeries::new(q1, color.mix(0.24)))
                    .map_err(|e| anyhow::anyhow!("failed to draw q1 series: {e:?}"))?;
                chart
                    .draw_series(LineSeries::new(q3, color.mix(0.24)))
                    .map_err(|e| anyhow::anyhow!("failed to draw q3 series: {e:?}"))?;
            }

            let mean = positive_points
                .iter()
                .map(|p| (p.adjusted_effort, p.coverage))
                .collect::<Vec<_>>();
            let empirical_style = ShapeStyle::from(&color).stroke_width(3);
            chart
                .draw_series(LineSeries::new(mean.clone(), empirical_style))?
                .label(coverage_legend_label(sample, &guides[idx]))
                .legend(move |(x, y)| {
                    PathElement::new(
                        [(x, y), (x + 22, y)],
                        ShapeStyle::from(&color).stroke_width(3),
                    )
                });
            chart
                .draw_series(mean.into_iter().map(|p| Circle::new(p, 3, color.filled())))
                .map_err(|e| anyhow::anyhow!("failed to draw empirical points: {e:?}"))?;

            if !sample.model.curve.is_empty() {
                let curve = sample
                    .model
                    .curve
                    .iter()
                    .map(|p| (p.adjusted_effort, p.coverage))
                    .collect::<Vec<_>>();
                chart
                    .draw_series(dashed_curve_segments(
                        &curve,
                        8,
                        5,
                        ShapeStyle::from(&color.mix(0.46)).stroke_width(1),
                    ))
                    .map_err(|e| anyhow::anyhow!("failed to draw gamma model curve: {e:?}"))?;
            }
        }

        if !guides.is_empty() {
            let guide_color = RGBColor(77, 77, 77);
            let guide_style = ShapeStyle::from(&guide_color.mix(0.72)).stroke_width(2);
            let max_matched_coverage = guides
                .iter()
                .map(|guide| guide.matched_coverage)
                .fold(0.0_f64, f64::max)
                .clamp(0.0, 1.0);
            let mut current_efforts = guides
                .iter()
                .map(|guide| guide.current_effort)
                .collect::<Vec<_>>();
            current_efforts.sort_by(|a, b| a.total_cmp(b));
            current_efforts
                .dedup_by(|a, b| (*a - *b).abs() <= a.abs().max(b.abs()).max(1.0) * 1e-9);

            for &current_effort in &current_efforts {
                chart
                    .draw_series(dashed_vertical_segments(
                        current_effort,
                        0.0,
                        max_matched_coverage,
                        18,
                        guide_style,
                    ))
                    .map_err(|e| anyhow::anyhow!("failed to draw current-effort guide: {e:?}"))?;
            }
            for guide in &guides {
                chart
                    .draw_series(dashed_horizontal_segments(
                        min_x,
                        guide.current_effort,
                        guide.matched_coverage,
                        28,
                        guide_style,
                    ))
                    .map_err(|e| anyhow::anyhow!("failed to draw matched-coverage guide: {e:?}"))?;
                chart
                    .draw_series(std::iter::once(Circle::new(
                        (guide.current_effort, guide.matched_coverage),
                        4,
                        guide.color.filled(),
                    )))
                    .map_err(|e| anyhow::anyhow!("failed to draw matched-coverage point: {e:?}"))?;
            }

            let x_arrow_tip = 0.01;
            let x_arrow_full_top = (max_matched_coverage * 0.55).clamp(0.12, 0.72);
            let x_arrow_height = (x_arrow_full_top - x_arrow_tip) * 0.20;
            let max_arrow_lanes = guides.len().clamp(1, 8);
            for (idx, guide) in guides.iter().enumerate() {
                let current_effort = guide.current_effort;
                let lane = current_effort_arrow_lane(idx, &guides, max_arrow_lanes);
                let lane_offset = lane as f64 * 0.018;
                let arrow_tip = x_arrow_tip + lane_offset;
                let arrow_top = arrow_tip + x_arrow_height;
                let arrow_head_height = (x_arrow_height * 0.45).max(0.018).min(x_arrow_height);
                let arrow_head_top = arrow_tip + arrow_head_height;
                let arrow_style = ShapeStyle::from(&guide.color).stroke_width(3);
                chart
                    .draw_series([
                        PathElement::new(
                            [(current_effort, arrow_top), (current_effort, arrow_tip)],
                            arrow_style,
                        ),
                        PathElement::new(
                            [
                                (current_effort / 1.18, arrow_head_top),
                                (current_effort, arrow_tip),
                            ],
                            arrow_style,
                        ),
                        PathElement::new(
                            [
                                (current_effort * 1.18, arrow_head_top),
                                (current_effort, arrow_tip),
                            ],
                            arrow_style,
                        ),
                    ])
                    .map_err(|e| anyhow::anyhow!("failed to draw current-effort arrow: {e:?}"))?;
            }
        }

        chart
            .configure_series_labels()
            .border_style(BLACK.mix(0.18))
            .background_style(WHITE.mix(0.88))
            .label_font(("sans-serif", 17))
            .draw()
            .map_err(|e| anyhow::anyhow!("failed to draw legend: {e:?}"))?;

        draw_diversity_bars(&diversity_area, samples)?;

        root.present()
            .map_err(|e| anyhow::anyhow!("failed to finalize SVG plot: {e:?}"))?;
    }
    Ok(svg)
}

fn draw_diversity_bars<DB: DrawingBackend>(
    area: &DrawingArea<DB, plotters::coord::Shift>,
    samples: &[PlotSample],
) -> Result<()>
where
    DB::ErrorType: std::fmt::Debug,
{
    let diversities = samples
        .iter()
        .map(|sample| {
            sample
                .model
                .diversity
                .filter(|value| value.is_finite() && *value > 0.0)
                .unwrap_or(0.0)
        })
        .collect::<Vec<_>>();
    let max_diversity = diversities.iter().copied().fold(0.0_f64, f64::max);
    let y_max = if max_diversity > 0.0 {
        (max_diversity * 1.12).max(max_diversity + 0.5)
    } else {
        1.0
    };
    let sample_count = samples.len();
    let x_end = sample_count as f64 - 0.5;
    let show_sample_labels = sample_count <= 6;

    let mut chart = ChartBuilder::on(area)
        .margin(20)
        .caption("Base-weighted Stevens Diversity", ("sans-serif", 24))
        .x_label_area_size(if show_sample_labels { 48 } else { 24 })
        .y_label_area_size(64)
        .build_cartesian_2d(-0.5f64..x_end, 0f64..y_max)
        .map_err(|e| anyhow::anyhow!("failed to build diversity bar chart: {e:?}"))?;

    let x_label = |x: &f64| -> String {
        let idx = x.round() as isize;
        if idx >= 0 && (idx as usize) < samples.len() {
            short_label(&samples[idx as usize].label)
        } else {
            String::new()
        }
    };
    let mut mesh = chart.configure_mesh();
    mesh.y_desc("Diversity (log-bp)")
        .axis_desc_style(("sans-serif", 18))
        .label_style(("sans-serif", 15))
        .disable_mesh();
    if show_sample_labels {
        mesh.x_desc("Sample")
            .x_labels(sample_count)
            .x_label_formatter(&x_label);
    } else {
        mesh.x_labels(0);
    }
    mesh.draw()
        .map_err(|e| anyhow::anyhow!("failed to draw diversity chart mesh: {e:?}"))?;

    chart
        .draw_series(diversities.iter().enumerate().map(|(idx, &diversity)| {
            let x = idx as f64;
            let color = sample_color(idx);
            Rectangle::new([(x - 0.34, 0.0), (x + 0.34, diversity)], color.filled())
        }))
        .map_err(|e| anyhow::anyhow!("failed to draw Stevens diversity bars: {e:?}"))?;

    let text_offset = y_max * 0.025;
    chart
        .draw_series(diversities.iter().enumerate().map(|(idx, &diversity)| {
            let color = sample_color(idx);
            Text::new(
                format!("{diversity:.2}"),
                (
                    idx as f64 - 0.22,
                    (diversity + text_offset).min(y_max * 0.98),
                ),
                ("sans-serif", 13).into_font().color(&color),
            )
        }))
        .map_err(|e| anyhow::anyhow!("failed to draw Stevens diversity labels: {e:?}"))?;

    Ok(())
}

fn coverage_guide(sample: &PlotSample, color: RGBColor, min_x: f64, max_x: f64) -> CoverageGuide {
    let latest = sample
        .model
        .points
        .iter()
        .rev()
        .find(|point| point.adjusted_effort > 0.0);
    let current_effort = latest
        .map(|point| point.adjusted_effort)
        .unwrap_or(sample.model.total_effort)
        .max(min_x)
        .min(max_x);
    let observed_coverage = latest
        .map(|point| point.coverage)
        .unwrap_or(sample.model.observed_coverage)
        .clamp(0.0, 1.0);
    let matched_coverage = latest
        .and_then(|point| point.fitted_coverage)
        .unwrap_or(observed_coverage)
        .clamp(0.0, 1.0);

    CoverageGuide {
        current_effort,
        observed_coverage,
        matched_coverage,
        color,
    }
}

fn current_effort_arrow_lane(
    guide_idx: usize,
    guides: &[CoverageGuide],
    max_lanes: usize,
) -> usize {
    let current = guides[guide_idx].current_effort;
    guides[..guide_idx]
        .iter()
        .filter(|guide| current_efforts_visually_overlap(current, guide.current_effort))
        .count()
        % max_lanes.max(1)
}

fn current_efforts_visually_overlap(a: f64, b: f64) -> bool {
    if a <= 0.0 || b <= 0.0 || !a.is_finite() || !b.is_finite() {
        return false;
    }
    (a.ln() - b.ln()).abs() <= 0.06
}

fn coverage_legend_label(sample: &PlotSample, guide: &CoverageGuide) -> String {
    format!(
        "{} C={:.3} fit={:.3}",
        sample.label, guide.observed_coverage, guide.matched_coverage
    )
}

fn short_label(label: &str) -> String {
    const MAX_CHARS: usize = 14;
    if label.chars().count() <= MAX_CHARS {
        return label.to_string();
    }
    let mut out = label.chars().take(MAX_CHARS - 3).collect::<String>();
    out.push_str("...");
    out
}

fn sample_color(index: usize) -> RGBColor {
    let hue = (202.0 + 137.507_764_05 * index as f64) % 360.0;
    let lightness = match index % 4 {
        0 => 0.31,
        1 => 0.43,
        2 => 0.36,
        _ => 0.48,
    };
    hsl_to_rgb(hue, 0.70, lightness)
}

fn hsl_to_rgb(hue_degrees: f64, saturation: f64, lightness: f64) -> RGBColor {
    let hue = (hue_degrees.rem_euclid(360.0)) / 360.0;
    let q = if lightness < 0.5 {
        lightness * (1.0 + saturation)
    } else {
        lightness + saturation - lightness * saturation
    };
    let p = 2.0 * lightness - q;
    RGBColor(
        (255.0 * hue_to_rgb(p, q, hue + 1.0 / 3.0)).round() as u8,
        (255.0 * hue_to_rgb(p, q, hue)).round() as u8,
        (255.0 * hue_to_rgb(p, q, hue - 1.0 / 3.0)).round() as u8,
    )
}

fn hue_to_rgb(p: f64, q: f64, mut t: f64) -> f64 {
    if t < 0.0 {
        t += 1.0;
    }
    if t > 1.0 {
        t -= 1.0;
    }
    if t < 1.0 / 6.0 {
        p + (q - p) * 6.0 * t
    } else if t < 1.0 / 2.0 {
        q
    } else if t < 2.0 / 3.0 {
        p + (q - p) * (2.0 / 3.0 - t) * 6.0
    } else {
        p
    }
}

fn dashed_curve_segments(
    points: &[(f64, f64)],
    on_points: usize,
    off_points: usize,
    style: ShapeStyle,
) -> Vec<PathElement<(f64, f64)>> {
    if points.len() < 2 || on_points == 0 {
        return Vec::new();
    }
    let period = on_points + off_points.max(1);
    let mut out = Vec::new();
    let mut start = 0usize;
    while start + 1 < points.len() {
        let end = (start + on_points).min(points.len());
        if end > start + 1 {
            out.push(PathElement::new(points[start..end].to_vec(), style));
        }
        start += period;
    }
    out
}

fn dashed_vertical_segments(
    x: f64,
    y0: f64,
    y1: f64,
    pieces: usize,
    style: ShapeStyle,
) -> Vec<PathElement<(f64, f64)>> {
    let pieces = pieces.max(2);
    let span = y1 - y0;
    (0..pieces)
        .step_by(2)
        .filter_map(|i| {
            let a = y0 + span * i as f64 / pieces as f64;
            let b = y0 + span * (i + 1).min(pieces) as f64 / pieces as f64;
            (b > a).then(|| PathElement::new([(x, a), (x, b)], style))
        })
        .collect()
}

fn dashed_horizontal_segments(
    x0: f64,
    x1: f64,
    y: f64,
    pieces: usize,
    style: ShapeStyle,
) -> Vec<PathElement<(f64, f64)>> {
    let pieces = pieces.max(2);
    if x0 <= 0.0 || x1 <= x0 {
        return Vec::new();
    }
    let start = x0.ln();
    let span = x1.ln() - start;
    (0..pieces)
        .step_by(2)
        .filter_map(|i| {
            let a = (start + span * i as f64 / pieces as f64).exp();
            let b = (start + span * (i + 1).min(pieces) as f64 / pieces as f64).exp();
            (b > a).then(|| PathElement::new([(a, y), (b, y)], style))
        })
        .collect()
}

fn write_vector_plot(svg: &str, svg_path: &Path, pdf_path: &Path) -> Result<()> {
    ensure_parent(svg_path)?;
    ensure_parent(pdf_path)?;
    fs::write(svg_path, svg)
        .with_context(|| format!("failed to write SVG {}", svg_path.display()))?;

    let mut options = svg2pdf::usvg::Options::default();
    options.fontdb_mut().load_system_fonts();
    let tree = svg2pdf::usvg::Tree::from_str(svg, &options)
        .map_err(|e| anyhow::anyhow!("failed to parse plot SVG before PDF conversion: {e}"))?;
    let pdf = svg2pdf::to_pdf(
        &tree,
        svg2pdf::ConversionOptions::default(),
        svg2pdf::PageOptions::default(),
    )
    .map_err(|e| anyhow::anyhow!("failed to convert SVG plot to PDF: {e:?}"))?;
    fs::write(pdf_path, pdf)
        .with_context(|| format!("failed to write PDF {}", pdf_path.display()))?;
    Ok(())
}

fn scientific_label(value: &f64) -> String {
    if !value.is_finite() || *value <= 0.0 {
        return "0".into();
    }
    format!("{value:.0e}")
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
