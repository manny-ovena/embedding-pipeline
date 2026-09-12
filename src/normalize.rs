#[derive(Debug, Clone, Copy)]
pub struct NormalizationOptions {
    pub collapse_whitespace: bool,
    pub strip_control_chars: bool,
    pub lowercase: bool,
}

impl Default for NormalizationOptions {
    fn default() -> Self {
        Self {
            collapse_whitespace: true,
            strip_control_chars: true,
            lowercase: false,
        }
    }
}

/// Normalizes raw input text by stripping control characters and collapsing whitespace.
pub fn normalize_text(raw: &str) -> String {
    normalize_text_with_options(raw, NormalizationOptions::default())
}

/// Normalizes text with configurable options.
pub fn normalize_text_with_options(raw: &str, options: NormalizationOptions) -> String {
    if raw.is_empty() {
        return String::new();
    }

    let mut result = String::with_capacity(raw.len());
    let mut in_whitespace = false;

    for ch in raw.chars() {
        if options.strip_control_chars && ch.is_control() && ch != '\n' && ch != '\t' {
            continue;
        }

        if options.collapse_whitespace && ch.is_whitespace() {
            if !in_whitespace {
                result.push(' ');
                in_whitespace = true;
            }
        } else {
            in_whitespace = false;
            if options.lowercase {
                for lower_ch in ch.to_lowercase() {
                    result.push(lower_ch);
                }
            } else {
                result.push(ch);
            }
        }
    }

    result.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_string() {
        assert_eq!(normalize_text(""), "");
    }

    #[test]
    fn test_whitespace_collapsing() {
        let input = "  The   quick\t\t \n\n brown  fox  ";
        assert_eq!(normalize_text(input), "The quick brown fox");
    }

    #[test]
    fn test_control_character_stripping() {
        let input = "Hello\u{0000}\u{0007} World\u{001B}!";
        assert_eq!(normalize_text(input), "Hello World!");
    }

    #[test]
    fn test_unicode_preservation() {
        let input = "  Rust 🦀   高性能  embeddings  ";
        assert_eq!(normalize_text(input), "Rust 🦀 高性能 embeddings");
    }

    #[test]
    fn test_lowercase_option() {
        let options = NormalizationOptions {
            lowercase: true,
            ..Default::default()
        };
        assert_eq!(
            normalize_text_with_options("Hello WORLD 123", options),
            "hello world 123"
        );
    }
}
