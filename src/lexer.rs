use logos::Logos;
use std::fmt;

use crate::Span;

#[derive(Logos, Debug, Clone, PartialEq)]
#[logos(skip r"[ \t\n\r]+")] // Skip whitespace
#[logos(skip r";[^\n\r]+")] // Skip comments
#[logos(error = LexerErrorKind)]
pub enum TokenKind {
    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    #[token(".")]
    Dot,
    #[token("'")]
    Quote,
    #[token("`")]
    QuasiQuote,
    #[token(",")]
    Unquote,
    #[token(",@")]
    UnquoteSplicing,
    #[regex(r"[\p{Extended_Pictographic}.a-zA-Z0-9!#$%&*/:<=>?~_^+-]*", |lex| lex.slice().to_string())]
    Symbol(String),
    #[regex(r"[-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?", |lex| {
        let slice = lex.slice();
        slice
            .parse::<f64>()
            .map_err(|_| LexerErrorKind::InvalidNumberFormat(slice.to_string()))
    })]
    Number(f64),
    #[token("#t", |_| true)]
    #[token("#f", |_| false)]
    Boolean(bool), // #t, #f
    #[regex(r#""([^"\\]|\\.)*.?"#, |lex| {
        let slice = lex.slice();
        let len = slice.len();
        // make sure string was terminated
        if len == 1 || &slice[len-1..] != "\"" {
            return Err(LexerErrorKind::UnterminatedString);
        }
        // Use a helper to unescape; return Err on failure
        unescape::unescape(&slice[1..slice.len()-1])
    })]
    String(String),
    // TODO: Characters #\a, #\b, etc.
    // TODO: add Dot '.' for improper lists
    // TODO: add Comment ';' #\a
}

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

mod unescape {
    use super::{LexerErrorKind, LexerResult};
    // Basic unescape logic
    pub fn unescape(s: &str) -> LexerResult<String> {
        // un-escaping should only ever reduce the length of the string.
        let mut result = String::with_capacity(s.len());
        let mut chars = s.chars();
        while let Some(c) = chars.next() {
            if c == '\\' {
                match chars.next() {
                    Some('n') => result.push('\n'),
                    Some('r') => result.push('\r'),
                    Some('t') => result.push('\t'),
                    Some('\\') => result.push('\\'),
                    Some('"') => result.push('"'),
                    // Add other escapes if needed (\r, \', etc.)
                    Some(c) => return Err(LexerErrorKind::UnknownEscapeSequence(c)),
                    None => return Err(LexerErrorKind::UnterminatedString),
                }
            } else {
                result.push(c);
            }
        }
        Ok(result)
    }
}

// Implement Display for easy printing
impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenKind::LParen => write!(f, "("),
            TokenKind::RParen => write!(f, ")"),
            TokenKind::Dot => write!(f, "."),
            TokenKind::Quote => write!(f, "'"),
            TokenKind::QuasiQuote => write!(f, "`"),
            TokenKind::Unquote => write!(f, ","),
            TokenKind::UnquoteSplicing => write!(f, ",@"),
            TokenKind::Symbol(s) => write!(f, "{}", s),
            TokenKind::Number(n) => write!(f, "{}", n),
            TokenKind::Boolean(b) => write!(f, "{}", if *b { "#t" } else { "#f" }),
            TokenKind::String(s) => write!(f, "\"{}\"", s), // Display with quotes for clarity
        }
    }
}

#[derive(Default, Debug, Clone, PartialEq)]
pub enum LexerErrorKind {
    UnexpectedEof,
    UnterminatedString,
    InvalidNumberFormat(String),
    InvalidCharacter(char),
    UnknownEscapeSequence(char),
    InvalidBooleanLiteral(String),
    #[default]
    InvalidToken,
    // Add more specific errors as needed
}

#[derive(Debug, Clone, PartialEq)]
pub struct LexerError {
    pub error: LexerErrorKind,
    pub span: Span,
}

impl fmt::Display for LexerErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LexerErrorKind::UnexpectedEof => write!(f, "Unexpected end of input"),
            LexerErrorKind::UnterminatedString => write!(f, "Unterminated string literal"),
            LexerErrorKind::InvalidNumberFormat(s) => write!(f, "Invalid number format: '{}'", s),
            LexerErrorKind::InvalidCharacter(c) => {
                write!(f, "Invalid character encountered: '{}'", c)
            }
            LexerErrorKind::UnknownEscapeSequence(c) => {
                write!(f, "Unknown escape sequence: '\\{}'", c)
            }
            LexerErrorKind::InvalidBooleanLiteral(s) => {
                write!(f, "Invalid boolean literal: '#{}'", s)
            }
            LexerErrorKind::InvalidToken => write!(f, "Invalid Token"),
        }
    }
}

impl fmt::Display for LexerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.error.fmt(f)
    }
}

// std::error::Error implementation allows using `?` with other error types
impl std::error::Error for LexerErrorKind {}

impl std::error::Error for LexerError {}

// Result type alias for convenience
type LexerResult<T> = Result<T, LexerErrorKind>;

// Result type alias for convenience
type LexerRangedResult<T> = Result<T, LexerError>;

// Helper function to tokenize a string directly (useful for tests and parser)
pub fn tokenize(input: &str) -> LexerRangedResult<Vec<Token>> {
    TokenKind::lexer(input)
        .spanned() // This yields Result<(TokenKind, Range<usize>), LexerError>
        .map(|(result, range)| match result {
            Ok(kind) => Ok(Token {
                kind,
                span: Span {
                    start: range.start,
                    end: range.end,
                },
            }),
            Err(error) => Err(LexerError {
                error,
                span: Span {
                    start: range.start,
                    end: range.end,
                },
            }),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Keep assert_tokens and assert_lexer_error helpers
    // Helper to simplify testing token sequences
    fn assert_tokens(input: &str, expected: Vec<TokenKind>) {
        match tokenize(input) {
            Ok(tokens) => {
                let kinds: Vec<TokenKind> = tokens.into_iter().map(|t| t.kind).collect();
                assert_eq!(kinds, expected, "Input: '{}'", input);
            }
            Err(e) => panic!("Lexing failed for input '{}': {}", input, e.error),
        }
    }

    // Helper to simplify testing for lexer errors
    fn assert_lexer_error(input: &str, expected_error_variant: LexerErrorKind) {
        match tokenize(input) {
            Ok(tokens) => panic!(
                "Expected lexing to fail for input '{}', but got tokens: {:?}",
                input, tokens
            ),
            Err(e) => {
                // Compare enum variants, ignoring specific content like the exact invalid char or number string for simplicity
                // Use std::mem::discriminant or match if more specific error checks are needed
                assert_eq!(
                    std::mem::discriminant(&e.error),
                    std::mem::discriminant(&expected_error_variant),
                    "Input: '{}', Expected error variant like {:?}, got: {:?}",
                    input,
                    expected_error_variant,
                    e
                );
            }
        }
    }

    #[test]
    fn test_empty_input() {
        assert_tokens("", vec![]);
    }

    #[test]
    fn test_parentheses_and_quote() {
        assert_tokens("()", vec![TokenKind::LParen, TokenKind::RParen]);
        assert_tokens("( )", vec![TokenKind::LParen, TokenKind::RParen]);
        assert_tokens(" ' ", vec![TokenKind::Quote]);
        assert_tokens(
            "(')",
            vec![TokenKind::LParen, TokenKind::Quote, TokenKind::RParen],
        );
        assert_tokens(" ` ", vec![TokenKind::QuasiQuote]);
        assert_tokens(" , ", vec![TokenKind::Unquote]);
        assert_tokens(" ,@ ", vec![TokenKind::UnquoteSplicing]);
        assert_tokens(
            "`(,@(1 2) ,x)",
            vec![
                TokenKind::QuasiQuote,
                TokenKind::LParen,
                TokenKind::UnquoteSplicing,
                TokenKind::LParen,
                TokenKind::Number(1.0),
                TokenKind::Number(2.0),
                TokenKind::RParen,
                TokenKind::Unquote,
                TokenKind::Symbol("x".to_string()),
                TokenKind::RParen,
            ],
        );
    }

    #[test]
    fn test_numbers() {
        assert_tokens("123", vec![TokenKind::Number(123.0)]);
        assert_tokens("-45", vec![TokenKind::Number(-45.0)]);
        assert_tokens("6.78", vec![TokenKind::Number(6.78)]);
        assert_tokens("-0.9", vec![TokenKind::Number(-0.9)]);
        assert_tokens(".5", vec![TokenKind::Number(0.5)]);
        assert_tokens("-.5", vec![TokenKind::Number(-0.5)]);
        assert_tokens("1.", vec![TokenKind::Number(1.0)]); // f64::parse("1.") works
        assert_tokens("+10", vec![TokenKind::Number(10.0)]); // f64::parse("+10") works
        assert_tokens("-1e-5", vec![TokenKind::Number(-1e-5)]); // Scientific notation
    }

    #[test]
    fn test_booleans() {
        assert_tokens("#t", vec![TokenKind::Boolean(true)]);
        assert_tokens("#f", vec![TokenKind::Boolean(false)]);
        assert_tokens(
            "(#t)",
            vec![
                TokenKind::LParen,
                TokenKind::Boolean(true),
                TokenKind::RParen,
            ],
        );
    }

    #[test]
    fn test_symbols() {
        assert_tokens("foo", vec![TokenKind::Symbol("foo".to_string())]);
        assert_tokens("+", vec![TokenKind::Symbol("+".to_string())]);
        assert_tokens("-", vec![TokenKind::Symbol("-".to_string())]);
        assert_tokens("*", vec![TokenKind::Symbol("*".to_string())]);
        assert_tokens("/", vec![TokenKind::Symbol("/".to_string())]);
        assert_tokens("<=?", vec![TokenKind::Symbol("<=?".to_string())]);
        assert_tokens("🍕+☕", vec![TokenKind::Symbol("🍕+☕".to_string())]);
        assert_tokens("...", vec![TokenKind::Symbol("...".to_string())]);
        assert_tokens(
            "a-symbol-with-hyphens",
            vec![TokenKind::Symbol("a-symbol-with-hyphens".to_string())],
        );
        assert_tokens("sym123", vec![TokenKind::Symbol("sym123".to_string())]);
    }

    #[test]
    fn test_dot_symbol() {
        // A lone dot is typically special in lists, but lexes as a symbol for now
        assert_tokens(".", vec![TokenKind::Dot]);
        assert_tokens(
            "(.)",
            vec![TokenKind::LParen, TokenKind::Dot, TokenKind::RParen],
        );
        assert_tokens(
            " a . b ",
            vec![
                TokenKind::Symbol("a".to_string()),
                TokenKind::Dot,
                TokenKind::Symbol("b".to_string()),
            ],
        );
        // Ensure dot within symbols/numbers still works
        assert_tokens("1.2", vec![TokenKind::Number(1.2)]);
        assert_tokens(".5", vec![TokenKind::Number(0.5)]);
        assert_tokens("sym.bol", vec![TokenKind::Symbol("sym.bol".to_string())]);
    }

    #[test]
    fn test_strings() {
        assert_tokens(r#""hello""#, vec![TokenKind::String("hello".to_string())]);
        assert_tokens(
            r#""with space""#,
            vec![TokenKind::String("with space".to_string())],
        );
        assert_tokens(
            r#""esc \" \n \t \\""#,
            vec![TokenKind::String("esc \" \n \t \\".to_string())],
        );
    }

    #[test]
    fn test_sequences_and_whitespace() {
        assert_tokens(
            "(+ 1 2)",
            vec![
                TokenKind::LParen,
                TokenKind::Symbol("+".to_string()),
                TokenKind::Number(1.0),
                TokenKind::Number(2.0),
                TokenKind::RParen,
            ],
        );
        assert_tokens(
            "  ( define x 10 )  ",
            vec![
                TokenKind::LParen,
                TokenKind::Symbol("define".to_string()),
                TokenKind::Symbol("x".to_string()),
                TokenKind::Number(10.0),
                TokenKind::RParen,
            ],
        );
    }

    #[test]
    fn test_comments() {
        let input = "
            (define x 10) ; Define x
            ; Another comment line
              (+ x 5)  ; Add 5 to x
              ; Final comment";
        assert_tokens(
            input,
            vec![
                TokenKind::LParen,
                TokenKind::Symbol("define".to_string()),
                TokenKind::Symbol("x".to_string()),
                TokenKind::Number(10.0),
                TokenKind::RParen,
                TokenKind::LParen,
                TokenKind::Symbol("+".to_string()),
                TokenKind::Symbol("x".to_string()),
                TokenKind::Number(5.0),
                TokenKind::RParen,
            ],
        );
        assert_tokens("; only comment", vec![]);
        assert_tokens(
            "token ; then comment",
            vec![TokenKind::Symbol("token".to_string())],
        );
    }

    #[test]
    fn test_mixed_types() {
        assert_tokens(
            "(list 'foo (bar 1 #t \"str\"))",
            vec![
                TokenKind::LParen,
                TokenKind::Symbol("list".to_string()),
                TokenKind::Quote,
                TokenKind::Symbol("foo".to_string()),
                TokenKind::LParen,
                TokenKind::Symbol("bar".to_string()),
                TokenKind::Number(1.0),
                TokenKind::Boolean(true),
                TokenKind::String("str".to_string()),
                TokenKind::RParen,
                TokenKind::RParen,
            ],
        );
    }

    // --- Tests for cases that should now be SYMBOLS ---

    #[test]
    fn test_number_like_symbols() {
        // These failed f64::parse and become symbols
        assert_tokens("1-2", vec![TokenKind::Symbol("1-2".to_string())]); // Correct!
        assert_tokens("+-", vec![TokenKind::Symbol("+-".to_string())]);
        assert_tokens("1.2.3", vec![TokenKind::Symbol("1.2.3".to_string())]); // Correct! (f64::parse fails)
        assert_tokens("--5", vec![TokenKind::Symbol("--5".to_string())]);
        assert_tokens("1e", vec![TokenKind::Symbol("1e".to_string())]);
        assert_tokens("1e-", vec![TokenKind::Symbol("1e-".to_string())]);
        assert_tokens(".+", vec![TokenKind::Symbol(".+".to_string())]);
        assert_tokens("-.", vec![TokenKind::Symbol("-.".to_string())]); // Correct!
    }

    #[test]
    fn test_boolean_like_symbols() {
        // These are not *exactly* #t or #f
        assert_tokens("#true", vec![TokenKind::Symbol("#true".to_string())]); // Correct!
        assert_tokens("#false", vec![TokenKind::Symbol("#false".to_string())]); // Correct!
        assert_tokens("#t1", vec![TokenKind::Symbol("#t1".to_string())]); // Correct!
        assert_tokens("#f?", vec![TokenKind::Symbol("#f?".to_string())]); // Correct!
        assert_tokens("#<foo>", vec![TokenKind::Symbol("#<foo>".to_string())]); // Often used for unreadable objects
    }

    // --- Error Condition Tests ---

    #[test]
    fn test_unterminated_string() {
        assert_lexer_error(r#""hellox"#, LexerErrorKind::UnterminatedString);
        assert_lexer_error(r#""hello"#, LexerErrorKind::UnterminatedString);
        assert_lexer_error(r#""hello\""#, LexerErrorKind::UnterminatedString);
        assert_lexer_error(r#"""#, LexerErrorKind::UnterminatedString);
    }

    #[test]
    fn test_invalid_escape() {
        assert_lexer_error(r#""hello \a""#, LexerErrorKind::UnknownEscapeSequence('a'));
    }

    #[test]
    fn test_bench_code() {
        let input = r#"
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
        "#;

        match tokenize(input) {
            Ok(tokens) => assert_eq!(tokens.len(), 91, "Input: '{}'", input),
            Err(e) => panic!("Lexing failed for input '{}': {}", input, e),
        }
    }
    #[test]
    fn test_simple_tokens() {
        assert_tokens(
            "( ) '",
            vec![TokenKind::LParen, TokenKind::RParen, TokenKind::Quote],
        );
    }

    #[test]
    fn test_atoms() {
        assert_tokens(
            "foo 123 -4.5 #t \"hello\"",
            vec![
                TokenKind::Symbol("foo".to_string()),
                TokenKind::Number(123.0),
                TokenKind::Number(-4.5),
                TokenKind::Boolean(true),
                TokenKind::String("hello".to_string()),
            ],
        );
    }

    #[test]
    fn test_string_escapes() {
        assert_tokens(
            r#""\n\t\r\"\\""#,
            vec![TokenKind::String("\n\t\r\"\\".to_string())],
        );
    }

    #[test]
    fn test_error_invalid_token() {
        // Example: Add a character Logos is configured to error on (if any)
        // If everything invalid is skipped or matched, this might be hard to hit directly.
        // Let's assume '^' is not skipped and not matched by any rule:
        // assert_lexer_error("^", std::mem::discriminant(&LexerError::InvalidToken));
        // Requires configuring Logos or the regexes such that '^' causes an error.
        // Currently, it might be skipped or become part of a symbol depending on regex.
        // Test based on your actual Logos setup.
    }

    #[test]
    fn test_error_invalid_escape() {
        assert_lexer_error(r#""hello \x""#, LexerErrorKind::UnknownEscapeSequence('x'));
        assert_lexer_error(r#""hello \"#, LexerErrorKind::UnterminatedString);
    }

    #[test]
    fn test_error_invalid_number() {
        // Regex might be too permissive, but if parse::<f64> fails within callback:
        // Example: A number format that the regex might allow but parse fails on (unlikely with current regex)
        // assert_lexer_error("1.2.3", std::mem::discriminant(&LexerError::InvalidNumberFormat("".into())));
        // Current Symbol regex likely grabs "1.2.3" before Number regex does.
        // Need a specific case where regex passes but parse fails. Maybe scientific notation issues?
        assert_tokens("1e", vec![TokenKind::Symbol("1e".to_string())]); // This becomes a symbol currently
        // If the regex was different, e.g., `\d+e`, it might fail parsing:
        // assert_lexer_error("1e", std::mem::discriminant(&LexerError::InvalidNumberFormat("".into())));
    }

    #[test]
    fn test_tokenize_spans() {
        // Verify spans manually for a simple case
        let input = "(+ 1)";
        let tokens = tokenize(input).expect("Should tokenize successfully");

        assert_eq!(tokens.len(), 4);

        assert_eq!(tokens[0].kind, TokenKind::LParen);
        assert_eq!(tokens[0].span, Span { start: 0, end: 1 });

        assert_eq!(tokens[1].kind, TokenKind::Symbol("+".to_string()));
        assert_eq!(tokens[1].span, Span { start: 1, end: 2 }); // Assumes no skip between ( and +

        assert_eq!(tokens[2].kind, TokenKind::Number(1.0));
        assert_eq!(tokens[2].span, Span { start: 3, end: 4 }); // Assumes space is skipped

        assert_eq!(tokens[3].kind, TokenKind::RParen);
        assert_eq!(tokens[3].span, Span { start: 4, end: 5 });
    }
}
