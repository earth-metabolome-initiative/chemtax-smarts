//! Small helpers shared across the crate.

use indicatif::ProgressStyle;

/// Convert a `usize` to `u64`, saturating at `u64::MAX` if it does not fit.
pub(crate) fn usize_to_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

/// Build a bar `ProgressStyle` from a template, falling back to the default bar
/// if the template is invalid. Every bar in the crate uses the `"=> "` progress
/// characters, so only the template differs between call sites.
pub(crate) fn progress_style(template: &str) -> ProgressStyle {
    match ProgressStyle::with_template(template) {
        Ok(style) => style,
        Err(_) => ProgressStyle::default_bar(),
    }
    .progress_chars("=> ")
}
