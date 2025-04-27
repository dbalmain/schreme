use crate::lexer::{LexerError, TokenKind}; // Assuming Token is in lexer
use crate::types::Sexpr; // Assuming Sexpr is in types
use std::fmt;
use std::iter::Peekable;
use std::vec::IntoIter; // To iterate over Vec<Token>

#[derive(Debug, Clone, PartialEq)]
pub enum ParseError {
    UnexpectedToken(Option<TokenKind>, String), // Found token, Expected description
    UnexpectedEof,
    LexerError(LexerError), // Propagate lexer errors if parsing directly from string later
    InvalidDotSyntax,       // For improper lists later
                            // Add more specific errors as needed
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::UnexpectedToken(token_opt, expected) => {
                if let Some(token) = token_opt {
                    write!(
                        f,
                        "Parse Error: Unexpected token '{:?}', expected {}",
                        token, expected
                    )
                } else {
                    write!(
                        f,
                        "Parse Error: Unexpected end of input, expected {}",
                        expected
                    )
                }
            }
            ParseError::UnexpectedEof => {
                write!(f, "Parse Error: Unexpected end of input during parsing")
            }
            ParseError::LexerError(lex_err) => write!(f, "Lexer Error during parse: {}", lex_err),
            ParseError::InvalidDotSyntax => {
                write!(f, "Parse Error: Invalid syntax for dotted pair")
            }
        }
    }
}

// Allow ParseError to be treated as a standard Error
impl std::error::Error for ParseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ParseError::LexerError(lex_err) => Some(lex_err),
            _ => None,
        }
    }
}

// Allow converting LexerError into ParseError easily
impl From<LexerError> for ParseError {
    fn from(err: LexerError) -> Self {
        ParseError::LexerError(err)
    }
}

// Result type alias for convenience
type ParseResult<T> = Result<T, ParseError>;

pub struct Parser {
    // We iterate over owned Tokens, consuming them.
    tokens: Peekable<IntoIter<TokenKind>>,
}

impl Parser {
    pub fn new(tokens: Vec<TokenKind>) -> Self {
        Parser {
            tokens: tokens.into_iter().peekable(),
        }
    }

    // Consumes the next token if available.
    fn next_token(&mut self) -> Option<TokenKind> {
        self.tokens.next()
    }

    // Peeks at the next token without consuming.
    fn peek_token(&mut self) -> Option<&TokenKind> {
        self.tokens.peek()
    }

    /// Parses a single S-expression from the token stream.
    pub fn parse_expr(&mut self) -> ParseResult<Sexpr> {
        match self.next_token() {
            Some(TokenKind::LParen) => self.parse_list(),
            Some(TokenKind::Quote) => self.parse_quote(),
            Some(atom) => self.parse_atom(atom), // Handle atoms if not '(' or '''
            None => Err(ParseError::UnexpectedEof), // No tokens left
        }
    }

    /// Parses an atomic expression (symbol, number, boolean, string).
    fn parse_atom(&mut self, token: TokenKind) -> ParseResult<Sexpr> {
        match token {
            TokenKind::Symbol(s) => Ok(Sexpr::Symbol(s)),
            TokenKind::Number(n) => Ok(Sexpr::Number(n)),
            TokenKind::Boolean(b) => Ok(Sexpr::Boolean(b)),
            TokenKind::String(s) => Ok(Sexpr::String(s)),
            other_token => Err(ParseError::UnexpectedToken(
                Some(other_token),
                "an atom (symbol, number, boolean, string)".to_string(),
            )),
        }
    }

    /// Parses a list expression `(...)`.
    fn parse_list(&mut self) -> ParseResult<Sexpr> {
        let mut elements: Vec<Sexpr> = Vec::new();

        // Loop until we find the closing parenthesis ')'
        loop {
            match self.peek_token() {
                Some(TokenKind::RParen) => {
                    // Found the closing parenthesis
                    self.next_token(); // Consume ')'
                    // Handle empty list '()' -> Sexpr::Nil
                    if elements.is_empty() {
                        return Ok(Sexpr::Nil);
                    } else {
                        return Ok(Sexpr::List(elements));
                    }
                }
                Some(_) => {
                    // Parse the next element inside the list
                    let expr = self.parse_expr()?; // Recursive call
                    elements.push(expr);
                }
                None => {
                    // Reached EOF before finding ')'
                    return Err(ParseError::UnexpectedToken(None, "')'".to_string()));
                }
            }
        }
    }

    /// Parses a quoted expression `'expr`.
    fn parse_quote(&mut self) -> ParseResult<Sexpr> {
        // Parse the expression immediately following the quote
        let quoted_expr = self.parse_expr()?;

        // Construct the equivalent (quote expr) S-expression
        Ok(Sexpr::List(vec![
            Sexpr::Symbol("quote".to_string()),
            quoted_expr,
        ]))
    }

    /// Parses the entire sequence of tokens, expecting potentially multiple top-level expressions.
    /// For now, let's assume we want to parse just ONE top-level expression from the input tokens.
    /// A full program might be a sequence, often wrapped in a `begin`.
    pub fn parse(mut self) -> ParseResult<Sexpr> {
        // Expect exactly one top-level expression for now
        let expr = self.parse_expr()?;

        // Check if there are any tokens left - shouldn't be for a single expression parse
        if self.peek_token().is_some() {
            Err(ParseError::UnexpectedToken(
                self.peek_token().cloned(),
                "end of input".to_string(),
            ))
        } else {
            Ok(expr)
        }
        // Later: To parse multiple expressions (like a file), loop `parse_expr` until EOF
        // and maybe wrap them in a (begin ...)?
        /*
        let mut expressions = Vec::new();
        while self.peek_token().is_some() {
            expressions.push(self.parse_expr()?);
        }
        // Decide how to return multiple expressions, e.g., wrap in begin or return Vec<Sexpr>
        if expressions.is_empty() {
             Err(ParseError::UnexpectedEof) // Or Ok(Sexpr::Void)?
        } else if expressions.len() == 1 {
             Ok(expressions.remove(0))
        } else {
             Ok(Sexpr::List(std::iter::once(Sexpr::Symbol("begin".to_string())).chain(expressions).collect()))
        }
        */
    }
}

// Helper function to lex and parse a string directly (useful for tests and REPL)
pub fn parse_str(input: &str) -> ParseResult<Sexpr> {
    let tokens = crate::lexer::tokenize(input)?
        .into_iter()
        .map(|token| token.kind)
        .collect(); // Use tokenize from lexer module
    Parser::new(tokens).parse()
}

#[cfg(test)]
mod tests {
    use super::*; // Import items from parent module (Parser, ParseError, parse_str)
    use crate::Span;
    use crate::lexer::LexerErrorKind;
    use crate::types::Sexpr; // Import Sexpr too

    // Helper for asserting successful parsing
    fn assert_parse(input: &str, expected: Sexpr) {
        match parse_str(input) {
            Ok(result) => assert_eq!(result, expected, "Input: '{}'", input),
            Err(e) => panic!("Parsing failed for input '{}': {}", input, e),
        }
    }

    // Helper for asserting parse errors
    fn assert_parse_error(input: &str, expected_error_variant: ParseError) {
        match parse_str(input) {
            Ok(result) => panic!(
                "Expected parsing to fail for input '{}', but got: {:?}",
                input, result
            ),
            Err(e) => {
                // Compare enum variants, ignoring specific content for simplicity
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
    fn test_parse_atoms() {
        assert_parse("123", Sexpr::Number(123.0));
        assert_parse("-4.5", Sexpr::Number(-4.5));
        assert_parse("symbol", Sexpr::Symbol("symbol".to_string()));
        assert_parse("+", Sexpr::Symbol("+".to_string()));
        assert_parse("#t", Sexpr::Boolean(true));
        assert_parse("#f", Sexpr::Boolean(false));
        assert_parse(r#""hello world""#, Sexpr::String("hello world".to_string()));
        assert_parse(
            r#""with \"quotes\"""#,
            Sexpr::String("with \"quotes\"".to_string()),
        );
    }

    #[test]
    fn test_parse_empty_list() {
        assert_parse("()", Sexpr::Nil);
        assert_parse("( )", Sexpr::Nil); // With space
    }

    #[test]
    fn test_parse_simple_list() {
        assert_parse(
            "(1 2 3)",
            Sexpr::List(vec![
                Sexpr::Number(1.0),
                Sexpr::Number(2.0),
                Sexpr::Number(3.0),
            ]),
        );
        assert_parse(
            "(+ 10 20)",
            Sexpr::List(vec![
                Sexpr::Symbol("+".to_string()),
                Sexpr::Number(10.0),
                Sexpr::Number(20.0),
            ]),
        );
        assert_parse(
            "(list #t \"hello\")",
            Sexpr::List(vec![
                Sexpr::Symbol("list".to_string()),
                Sexpr::Boolean(true),
                Sexpr::String("hello".to_string()),
            ]),
        );
    }

    #[test]
    fn test_parse_nested_list() {
        assert_parse(
            "(a (b c) d)",
            Sexpr::List(vec![
                Sexpr::Symbol("a".to_string()),
                Sexpr::List(vec![
                    Sexpr::Symbol("b".to_string()),
                    Sexpr::Symbol("c".to_string()),
                ]),
                Sexpr::Symbol("d".to_string()),
            ]),
        );
        assert_parse("(()())", Sexpr::List(vec![Sexpr::Nil, Sexpr::Nil]));
    }

    #[test]
    fn test_parse_quote_sugar() {
        assert_parse(
            "'a",
            Sexpr::List(vec![
                Sexpr::Symbol("quote".to_string()),
                Sexpr::Symbol("a".to_string()),
            ]),
        );
        assert_parse(
            "'123",
            Sexpr::List(vec![
                Sexpr::Symbol("quote".to_string()),
                Sexpr::Number(123.0),
            ]),
        );
        assert_parse(
            "'()",
            Sexpr::List(vec![Sexpr::Symbol("quote".to_string()), Sexpr::Nil]),
        );
        assert_parse(
            "'(1 2)",
            Sexpr::List(vec![
                Sexpr::Symbol("quote".to_string()),
                Sexpr::List(vec![Sexpr::Number(1.0), Sexpr::Number(2.0)]),
            ]),
        );
        assert_parse(
            "(list 'a 'b)",
            Sexpr::List(vec![
                Sexpr::Symbol("list".to_string()),
                Sexpr::List(vec![
                    Sexpr::Symbol("quote".to_string()),
                    Sexpr::Symbol("a".to_string()),
                ]),
                Sexpr::List(vec![
                    Sexpr::Symbol("quote".to_string()),
                    Sexpr::Symbol("b".to_string()),
                ]),
            ]),
        );
    }

    #[test]
    fn test_parse_errors_unexpected_token() {
        // Using dummy content for the expected error variant
        assert_parse_error("(1 2", ParseError::UnexpectedToken(None, "')'".to_string()));
        assert_parse_error(
            "(1 . 2)",
            ParseError::UnexpectedToken(
                Some(TokenKind::Symbol(".".to_string())),
                "')'".to_string(),
            ),
        ); // Assuming dot not handled yet
        assert_parse_error(
            ")",
            ParseError::UnexpectedToken(
                Some(TokenKind::RParen),
                "an atom or '(' or '''".to_string(),
            ),
        ); // Simplified expected msg
        assert_parse_error(
            "(1))",
            ParseError::UnexpectedToken(Some(TokenKind::RParen), "end of input".to_string()),
        );
        assert_parse_error(
            "(')",
            ParseError::UnexpectedToken(
                Some(TokenKind::RParen),
                "an atom or '(' or '''".to_string(),
            ),
        ); // After quote, expects expression
        assert_parse_error("(", ParseError::UnexpectedToken(None, "')'".to_string())); // EOF inside list
    }

    #[test]
    fn test_parse_errors_eof() {
        assert_parse_error("", ParseError::UnexpectedEof);
        assert_parse_error("'", ParseError::UnexpectedEof); // EOF after quote
    }

    #[test]
    fn test_parse_lexer_error_propagation() {
        assert_parse_error(
            "\"",
            ParseError::LexerError(LexerError {
                error: LexerErrorKind::UnterminatedString,
                span: Span { start: 0, end: 1 },
            }),
        ); // Propagates lexer error
        assert_parse_error(
            "(1 \"abc",
            ParseError::LexerError(LexerError {
                error: LexerErrorKind::UnterminatedString,
                span: Span { start: 0, end: 6 },
            }),
        );
    }

    #[test]
    fn test_whitespace_and_comments_parsing() {
        // Parser operates on tokens; whitespace/comments are handled by lexer
        assert_parse(
            " ( + 1 2 ) ; comment",
            Sexpr::List(vec![
                Sexpr::Symbol("+".to_string()),
                Sexpr::Number(1.0),
                Sexpr::Number(2.0),
            ]),
        );
        assert_parse(
            " ; comment at start\n   'symbol   ; comment at end\n ",
            Sexpr::List(vec![
                Sexpr::Symbol("quote".to_string()),
                Sexpr::Symbol("symbol".to_string()),
            ]),
        );
    }
}
