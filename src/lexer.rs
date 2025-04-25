use logos::{Logos, Source};
use std::fmt;
use std::iter::Peekable;
use std::num::ParseFloatError;
use std::str::Chars;

impl From<ParseFloatError> for LexerError {
    fn from(err: ParseFloatError) -> Self {
        LexerError::InvalidNumberFormat(err.to_string())
    }
}

#[derive(Logos, Debug, Clone, PartialEq)]
#[logos(skip r"[ \t\n\r]+")] // Skip whitespace
#[logos(skip r";[^\n\r]+")] // Skip comments
#[logos(error = LexerError)]
pub enum Token {
    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    #[token(".")]
    Dot,
    #[token("'")]
    Quote,
    #[regex(r"[.a-zA-Z0-9!#$%&*/:<=>?~_^+-]*", |lex| lex.slice().to_string())]
    Symbol(String),
    #[regex(r"[-+]?[0-9]+\.?", |lex| lex.slice().parse())]
    #[regex(r"[-+]?[0-9]*\.[0-9]+", |lex| lex.slice().parse())]
    #[regex(r"[-+]?[0-9]*[eE][-+]?[0-9]+", |lex| lex.slice().parse())]
    Number(f64),
    #[token("#t", |_| true)]
    #[token("#f", |_| false)]
    Boolean(bool), // #t, #f
    #[token("\"", string_lexer)]
    String(String),
    // TODO: Characters #\a, #\b, etc.
    // TODO: add Dot '.' for improper lists
    // TODO: add Comment ';' #\a
}

#[derive(Logos, Debug, PartialEq, Clone)]
#[logos()]
enum StringContext {
    #[token("\"")]
    Quote,
    #[regex(r#"\\[ntr\\"]"#)]
    EscapedChar,
    #[regex(r#"\\[^ntr\\"]"#)]
    UnknownEscapedChar,
    #[regex(r#"[^"\\]"#)]
    Content,
}

// TODO: Is it faster to match thw whole string and then run a mini lexer?
fn string_lexer(lex: &mut logos::Lexer<Token>) -> LexerResult<String> {
    let mut result = String::new();
    let mut string_lexer = lex.clone().morph::<StringContext>();
    while let Some(Ok(token)) = string_lexer.next() {
        match token {
            StringContext::Quote => {
                *lex = string_lexer.morph();
                return Ok(result);
            }
            StringContext::EscapedChar => {
                result.push(match string_lexer.slice().chars().nth(1).unwrap() {
                    '"' => '"',
                    'n' => '\n',
                    'r' => '\r',
                    't' => '\t',
                    c => c, // should never happen
                })
            }
            StringContext::UnknownEscapedChar => {
                if let Some(c) = string_lexer.slice().chars().nth(1) {
                    return Err(LexerError::UnknownEscapeSequence(c));
                }
            }
            StringContext::Content => result.push_str(string_lexer.slice()),
        }
    }
    Err(LexerError::UnterminatedString)
}

// Implement Display for easy printing
impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Token::LParen => write!(f, "("),
            Token::RParen => write!(f, ")"),
            Token::Dot => write!(f, "."),
            Token::Quote => write!(f, "'"),
            Token::Symbol(s) => write!(f, "{}", s),
            Token::Number(n) => write!(f, "{}", n),
            Token::Boolean(b) => write!(f, "{}", if *b { "#t" } else { "#f" }),
            Token::String(s) => write!(f, "\"{}\"", s), // Display with quotes for clarity
        }
    }
}

#[derive(Default, Debug, Clone, PartialEq)]
pub enum LexerError {
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

impl fmt::Display for LexerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LexerError::UnexpectedEof => write!(f, "Unexpected end of input"),
            LexerError::UnterminatedString => write!(f, "Unterminated string literal"),
            LexerError::InvalidNumberFormat(s) => write!(f, "Invalid number format: '{}'", s),
            LexerError::InvalidCharacter(c) => write!(f, "Invalid character encountered: '{}'", c),
            LexerError::UnknownEscapeSequence(c) => write!(f, "Unknown escape sequence: '\\{}'", c),
            LexerError::InvalidBooleanLiteral(s) => write!(f, "Invalid boolean literal: '#{}'", s),
            LexerError::InvalidToken => write!(f, "Invalid Token"),
        }
    }
}

// std::error::Error implementation allows using `?` with other error types
impl std::error::Error for LexerError {}

// Result type alias for convenience
type LexerResult<T> = Result<T, LexerError>;

// Helper function to tokenize a string directly (useful for tests and parser)
pub fn tokenize(input: &str) -> LexerResult<Vec<Token>> {
    Token::lexer(input).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Keep assert_tokens and assert_lexer_error helpers
    // Helper to simplify testing token sequences
    fn assert_tokens(input: &str, expected: Vec<Token>) {
        match tokenize(input) {
            Ok(tokens) => assert_eq!(tokens, expected, "Input: '{}'", input),
            Err(e) => panic!("Lexing failed for input '{}': {}", input, e),
        }
    }

    // Helper to simplify testing for lexer errors
    fn assert_lexer_error(input: &str, expected_error_variant: LexerError) {
        match tokenize(input) {
            Ok(tokens) => panic!(
                "Expected lexing to fail for input '{}', but got tokens: {:?}",
                input, tokens
            ),
            Err(e) => {
                // Compare enum variants, ignoring specific content like the exact invalid char or number string for simplicity
                // Use std::mem::discriminant or match if more specific error checks are needed
                assert_eq!(
                    std::mem::discriminant(&e),
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
        assert_tokens("()", vec![Token::LParen, Token::RParen]);
        assert_tokens("( )", vec![Token::LParen, Token::RParen]);
        assert_tokens(" ' ", vec![Token::Quote]);
        assert_tokens("(')", vec![Token::LParen, Token::Quote, Token::RParen]);
    }

    #[test]
    fn test_numbers() {
        assert_tokens("123", vec![Token::Number(123.0)]);
        assert_tokens("-45", vec![Token::Number(-45.0)]);
        assert_tokens("6.78", vec![Token::Number(6.78)]);
        assert_tokens("-0.9", vec![Token::Number(-0.9)]);
        assert_tokens(".5", vec![Token::Number(0.5)]);
        assert_tokens("-.5", vec![Token::Number(-0.5)]);
        assert_tokens("1.", vec![Token::Number(1.0)]); // f64::parse("1.") works
        assert_tokens("+10", vec![Token::Number(10.0)]); // f64::parse("+10") works
        assert_tokens("-1e-5", vec![Token::Number(-1e-5)]); // Scientific notation
    }

    #[test]
    fn test_booleans() {
        assert_tokens("#t", vec![Token::Boolean(true)]);
        assert_tokens("#f", vec![Token::Boolean(false)]);
        assert_tokens(
            "(#t)",
            vec![Token::LParen, Token::Boolean(true), Token::RParen],
        );
    }

    #[test]
    fn test_symbols() {
        assert_tokens("foo", vec![Token::Symbol("foo".to_string())]);
        assert_tokens("+", vec![Token::Symbol("+".to_string())]);
        assert_tokens("-", vec![Token::Symbol("-".to_string())]);
        assert_tokens("*", vec![Token::Symbol("*".to_string())]);
        assert_tokens("/", vec![Token::Symbol("/".to_string())]);
        assert_tokens("<=?", vec![Token::Symbol("<=?".to_string())]);
        assert_tokens("...", vec![Token::Symbol("...".to_string())]);
        assert_tokens(
            "a-symbol-with-hyphens",
            vec![Token::Symbol("a-symbol-with-hyphens".to_string())],
        );
        assert_tokens("sym123", vec![Token::Symbol("sym123".to_string())]);
    }

    #[test]
    fn test_dot_symbol() {
        // A lone dot is typically special in lists, but lexes as a symbol for now
        assert_tokens(".", vec![Token::Dot]);
        assert_tokens("(.)", vec![Token::LParen, Token::Dot, Token::RParen]);
    }

    #[test]
    fn test_strings() {
        assert_tokens(r#""hello""#, vec![Token::String("hello".to_string())]);
        assert_tokens(
            r#""with space""#,
            vec![Token::String("with space".to_string())],
        );
        assert_tokens(
            r#""esc \" \n \t \\""#,
            vec![Token::String("esc \" \n \t \\".to_string())],
        );
    }

    #[test]
    fn test_sequences_and_whitespace() {
        assert_tokens(
            "(+ 1 2)",
            vec![
                Token::LParen,
                Token::Symbol("+".to_string()),
                Token::Number(1.0),
                Token::Number(2.0),
                Token::RParen,
            ],
        );
        assert_tokens(
            "  ( define x 10 )  ",
            vec![
                Token::LParen,
                Token::Symbol("define".to_string()),
                Token::Symbol("x".to_string()),
                Token::Number(10.0),
                Token::RParen,
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
                Token::LParen,
                Token::Symbol("define".to_string()),
                Token::Symbol("x".to_string()),
                Token::Number(10.0),
                Token::RParen,
                Token::LParen,
                Token::Symbol("+".to_string()),
                Token::Symbol("x".to_string()),
                Token::Number(5.0),
                Token::RParen,
            ],
        );
        assert_tokens("; only comment", vec![]);
        assert_tokens(
            "token ; then comment",
            vec![Token::Symbol("token".to_string())],
        );
    }

    #[test]
    fn test_mixed_types() {
        assert_tokens(
            "(list 'foo (bar 1 #t \"str\"))",
            vec![
                Token::LParen,
                Token::Symbol("list".to_string()),
                Token::Quote,
                Token::Symbol("foo".to_string()),
                Token::LParen,
                Token::Symbol("bar".to_string()),
                Token::Number(1.0),
                Token::Boolean(true),
                Token::String("str".to_string()),
                Token::RParen,
                Token::RParen,
            ],
        );
    }

    // --- Tests for cases that should now be SYMBOLS ---

    #[test]
    fn test_number_like_symbols() {
        // These failed f64::parse and become symbols
        assert_tokens("1-2", vec![Token::Symbol("1-2".to_string())]); // Correct!
        assert_tokens("+-", vec![Token::Symbol("+-".to_string())]);
        assert_tokens("1.2.3", vec![Token::Symbol("1.2.3".to_string())]); // Correct! (f64::parse fails)
        assert_tokens("--5", vec![Token::Symbol("--5".to_string())]);
        assert_tokens("1e", vec![Token::Symbol("1e".to_string())]);
        assert_tokens("1e-", vec![Token::Symbol("1e-".to_string())]);
        assert_tokens(".+", vec![Token::Symbol(".+".to_string())]);
        assert_tokens("-.", vec![Token::Symbol("-.".to_string())]); // Correct!
    }

    #[test]
    fn test_boolean_like_symbols() {
        // These are not *exactly* #t or #f
        assert_tokens("#true", vec![Token::Symbol("#true".to_string())]); // Correct!
        assert_tokens("#false", vec![Token::Symbol("#false".to_string())]); // Correct!
        assert_tokens("#t1", vec![Token::Symbol("#t1".to_string())]); // Correct!
        assert_tokens("#f?", vec![Token::Symbol("#f?".to_string())]); // Correct!
        assert_tokens("#<foo>", vec![Token::Symbol("#<foo>".to_string())]); // Often used for unreadable objects
    }

    // --- Error Condition Tests ---

    #[test]
    fn test_unterminated_string() {
        assert_lexer_error(r#""hello"#, LexerError::UnterminatedString);
        assert_lexer_error(r#""hello\""#, LexerError::UnterminatedString);
    }

    #[test]
    fn test_invalid_escape() {
        assert_lexer_error(r#""hello \a""#, LexerError::UnknownEscapeSequence('a'));
    }

    // Invalid characters might be harder to hit now, as most things become symbols.
    // Depends if you want *any* character restrictions on symbols.
    // Standard scheme symbols are quite permissive. Let's remove invalid char test for now.
    /*
    #[test]
    fn test_invalid_character() {
        // Need to define what's truly invalid if not part of string/comment/token
        // Perhaps control characters? Or characters explicitly disallowed in symbols?
        // For now, most things become symbols or part of strings/comments.
        // assert_lexer_error("`", LexerError::InvalidCharacter('`')); // Example if backtick is invalid
    }
    */

    // Invalid number format errors are less likely now, as failures result in Symbols.
    // Errors would primarily come from `read_string` or maybe EOF issues.
}
