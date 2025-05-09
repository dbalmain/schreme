use crate::Span; // Assuming Sexpr is in types
use crate::lexer::{LexerError, Token, TokenKind}; // Assuming Token is in lexer
use crate::types::{Node, Sexpr};
use std::fmt;
use std::iter::Peekable;
use std::rc::Rc;
use std::vec::IntoIter; // To iterate over Vec<Token>

#[derive(Debug, Clone, PartialEq)]
pub enum ParseError {
    UnexpectedToken { found: Token, expected: String }, // Found token, Expected description
    UnexpectedEof(String),
    LexerError(LexerError), // Propagate lexer errors if parsing directly from string later
    InvalidDotSyntax(Span), // For improper lists later
                            // Add more specific errors as needed
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::UnexpectedToken { found, expected } => {
                write!(
                    f,
                    "Parse Error [at {}..{}]: Unexpected token '{:?}', expected {}",
                    found.span.start, found.span.end, found, expected
                )
            }
            ParseError::UnexpectedEof(expected) => {
                write!(
                    f,
                    "Parse Error: Unexpected end of input during parsing. Expected {}",
                    expected
                )
            }
            ParseError::LexerError(lex_err) => write!(f, "Lexer Error during parse: {}", lex_err),
            ParseError::InvalidDotSyntax(span) => {
                write!(
                    f,
                    "Parse Error: Invalid syntax for dotted pair at [{}]",
                    span
                )
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
    tokens: Peekable<IntoIter<Token>>,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Parser {
            tokens: tokens.into_iter().peekable(),
        }
    }

    // Consumes the next token if available.
    fn next_token(&mut self) -> Option<Token> {
        self.tokens.next()
    }

    // Peeks at the next token without consuming.
    fn peek_token(&mut self) -> Option<&Token> {
        self.tokens.peek()
    }

    /// Parses a single S-expression from the token stream.
    pub fn parse_expr_with_token(&mut self, token: Option<Token>) -> ParseResult<Node> {
        match token {
            Some(Token {
                kind: TokenKind::LParen,
                span,
            }) => {
                match self.next_token() {
                    Some(Token {
                        kind: TokenKind::RParen,
                        span: _rparen_span,
                    }) => {
                        // Empty list case
                        self.next_token(); // Consume ')'
                        return Ok(Node::new(Sexpr::Nil, span.merge(&span)));
                    }
                    Some(token) => {
                        let car_node = self.parse_expr_with_token(Some(token))?;
                        self.parse_list_recursive(&span, car_node)
                    }
                    _ => Err(ParseError::UnexpectedEof("')'".to_string())),
                }
            }
            Some(Token {
                kind: TokenKind::Quote,
                span,
            }) => self.parse_quote(span),
            Some(atom) => self.parse_atom(atom), // Handle atoms if not '(' or '''
            None => Err(ParseError::UnexpectedEof(")".to_string())), // No tokens left
        }
    }

    pub fn parse_expr(&mut self) -> ParseResult<Node> {
        let token = self.next_token();
        self.parse_expr_with_token(token)
    }

    /// Parses an atomic expression (symbol, number, boolean, string).
    fn parse_atom(&mut self, token: Token) -> ParseResult<Node> {
        Ok(Node::new(
            match token.kind {
                TokenKind::Symbol(s) => Sexpr::Symbol(s),
                TokenKind::Number(n) => Sexpr::Number(n),
                TokenKind::Boolean(b) => Sexpr::Boolean(b),
                TokenKind::String(s) => Sexpr::String(s),
                other_token => Err(ParseError::UnexpectedToken {
                    found: Token {
                        kind: other_token,
                        span: token.span,
                    },
                    expected: "an atom (symbol, number, boolean, string)".to_string(),
                })?,
            },
            token.span,
        ))
    }

    /// Parses a list expression `(...)`.
    fn parse_list_recursive(&mut self, start_span: &Span, car_node: Node) -> ParseResult<Node> {
        match self.next_token() {
            Some(Token {
                kind: TokenKind::Dot,
                span: _dot_span,
            }) => {
                let cdr_node = self.parse_expr()?;

                match self.next_token() {
                    Some(Token {
                        kind: TokenKind::RParen,
                        span: rparen_span,
                    }) => {
                        let final_span = start_span.merge(&rparen_span);
                        Ok(Node::new(
                            // Wrap nodes in Rc
                            Sexpr::Pair(car_node, cdr_node),
                            final_span,
                        ))
                    }
                    // ... (error handling) ...
                    Some(t) => Err(ParseError::UnexpectedToken {
                        found: t,
                        expected: "')' after dotted pair".to_string(),
                    }),
                    None => Err(ParseError::UnexpectedEof(
                        "')' after dotted pair".to_string(),
                    )),
                }
            }
            Some(Token {
                kind: TokenKind::RParen,
                span: rparen_span,
            }) => {
                // Consume ')'
                let final_span = start_span.merge(&rparen_span);
                // Use a shared Nil node if desired, or create new one
                let nil_node = Node::new(Sexpr::Nil, rparen_span);
                Ok(Node::new(Sexpr::Pair(car_node, nil_node), final_span))
            }
            Some(token) => {
                let next_car_node = self.parse_expr_with_token(Some(token))?;
                let cdr_node = self.parse_list_recursive(start_span, next_car_node)?;
                let final_span = start_span.merge(&cdr_node.span); // Calculate span covering whole list
                Ok(Node::new(Sexpr::Pair(car_node, cdr_node), final_span))
            }
            None => {
                // Reached EOF before finding ')'
                Err(ParseError::UnexpectedEof("')'".to_string()))
            }
        }
    }

    /// Parses a quoted expression `'expr`.
    fn parse_quote(&mut self, quote_span: Span) -> ParseResult<Node> {
        // Parse the expression immediately following the quote
        let quoted_expr = self.parse_expr()?;
        let span = quote_span.merge(&quoted_expr.span);

        // Construct the equivalent (quote expr) S-expression
        Ok(Node::new(
            Sexpr::Pair(
                Node::new(Sexpr::Symbol("quote".to_string()), quote_span),
                quoted_expr,
            ),
            span,
        ))
    }

    /// Parses the entire sequence of tokens, expecting potentially multiple top-level expressions.
    /// For now, let's assume we want to parse just ONE top-level expression from the input tokens.
    /// A full program might be a sequence, often wrapped in a `begin`.
    pub fn parse(mut self) -> ParseResult<Node> {
        // Expect exactly one top-level expression for now
        let expr = self.parse_expr()?;

        // Check if there are any tokens left - shouldn't be for a single expression parse
        if let Some(found) = self.next_token() {
            Err(ParseError::UnexpectedToken {
                found,
                expected: "end of input".to_string(),
            })
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
pub fn parse_str(input: &str) -> ParseResult<Node> {
    let tokens = crate::lexer::tokenize(input)?;
    Parser::new(tokens).parse()
}

#[cfg(test)]
mod tests {
    use super::*; // Import items from parent module (Parser, ParseError, parse_str)
    use crate::Span;
    use crate::lexer::LexerErrorKind;
    use crate::types::Sexpr; // Import Sexpr too

    // Helper for asserting successful parsing
    fn assert_parse(input: &str, expected: Node) {
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

    fn node(sexpr: Sexpr, start: usize, end: usize) -> Node {
        Node::new(sexpr, Span::new(start, end))
    }

    fn list(nodes: &[Node], start: usize, end: usize) -> Node {
        match nodes {
            [] => Node::new(Sexpr::Nil, Span::new(start, end)),
            [first, rest @ ..] => Node::new(
                Sexpr::Pair(first.clone(), list(rest, start, end)),
                Span::new(start, end),
            ),
        }
    }

    fn unexpected_token(kind: TokenKind, start: usize, end: usize, expected: String) -> ParseError {
        ParseError::UnexpectedToken {
            found: Token {
                kind,
                span: Span::new(start, end),
            },
            expected,
        }
    }

    #[test]
    fn test_parse_atoms() {
        assert_parse("123", node(Sexpr::Number(123.0), 0, 3));
        assert_parse("-4.5", node(Sexpr::Number(-4.5), 0, 4));
        assert_parse("symbol", node(Sexpr::Symbol("symbol".to_string()), 0, 6));
        assert_parse("+", node(Sexpr::Symbol("+".to_string()), 0, 1));
        assert_parse("#t", node(Sexpr::Boolean(true), 0, 2));
        assert_parse("#f", node(Sexpr::Boolean(false), 0, 2));
        assert_parse(
            r#""hello world""#,
            node(Sexpr::String("hello world".to_string()), 0, 13),
        );
        assert_parse(
            r#""with \"quotes\"""#,
            node(Sexpr::String("with \"quotes\"".to_string()), 0, 17),
        );
    }

    #[test]
    fn test_parse_empty_list() {
        assert_parse("()", node(Sexpr::Nil, 0, 2));
        assert_parse("( )", node(Sexpr::Nil, 0, 3)); // With space
    }

    #[test]
    fn test_parse_simple_list() {
        assert_parse(
            "(1 2 3)",
            list(
                &[
                    node(Sexpr::Number(1.0), 1, 2),
                    node(Sexpr::Number(2.0), 3, 4),
                    node(Sexpr::Number(3.0), 5, 6),
                ],
                0,
                7,
            ),
        );
        assert_parse(
            "(+ 10 20)",
            list(
                &[
                    node(Sexpr::Symbol("+".to_string()), 1, 2),
                    node(Sexpr::Number(10.0), 3, 5),
                    node(Sexpr::Number(20.0), 6, 8),
                ],
                0,
                9,
            ),
        );
        assert_parse(
            "(list #t \"hello\")",
            list(
                &[
                    node(Sexpr::Symbol("list".to_string()), 1, 5),
                    node(Sexpr::Boolean(true), 6, 8),
                    node(Sexpr::String("hello".to_string()), 9, 16),
                ],
                0,
                17,
            ),
        );
    }

    #[test]
    fn test_parse_nested_list() {
        assert_parse(
            "(a (b c) d)",
            list(
                &[
                    node(Sexpr::Symbol("a".to_string()), 1, 2),
                    list(
                        &[
                            node(Sexpr::Symbol("b".to_string()), 4, 5),
                            node(Sexpr::Symbol("c".to_string()), 6, 7),
                        ],
                        3,
                        8,
                    ),
                    node(Sexpr::Symbol("d".to_string()), 9, 10),
                ],
                0,
                11,
            ),
        );
        assert_parse(
            "(()())",
            list(&[node(Sexpr::Nil, 1, 3), node(Sexpr::Nil, 3, 5)], 0, 6),
        );
    }

    #[test]
    fn test_parse_quote_sugar() {
        assert_parse(
            "'a",
            list(
                &[
                    node(Sexpr::Symbol("quote".to_string()), 0, 1),
                    node(Sexpr::Symbol("a".to_string()), 1, 2),
                ],
                0,
                2,
            ),
        );
        assert_parse(
            "'123",
            list(
                &[
                    node(Sexpr::Symbol("quote".to_string()), 0, 1),
                    node(Sexpr::Number(123.0), 1, 4),
                ],
                0,
                4,
            ),
        );
        assert_parse(
            "'()",
            list(
                &[
                    node(Sexpr::Symbol("quote".to_string()), 0, 1),
                    node(Sexpr::Nil, 1, 3),
                ],
                0,
                3,
            ),
        );
        assert_parse(
            "'(1 2)",
            list(
                &[
                    node(Sexpr::Symbol("quote".to_string()), 0, 1),
                    list(
                        &[
                            node(Sexpr::Number(1.0), 2, 3),
                            node(Sexpr::Number(2.0), 4, 5),
                        ],
                        1,
                        6,
                    ),
                ],
                0,
                6,
            ),
        );
        assert_parse(
            "(list 'a 'b)",
            list(
                &[
                    node(Sexpr::Symbol("list".to_string()), 1, 5),
                    list(
                        &[
                            node(Sexpr::Symbol("quote".to_string()), 6, 7),
                            node(Sexpr::Symbol("a".to_string()), 7, 8),
                        ],
                        6,
                        8,
                    ),
                    list(
                        &[
                            node(Sexpr::Symbol("quote".to_string()), 9, 10),
                            node(Sexpr::Symbol("b".to_string()), 10, 11),
                        ],
                        9,
                        11,
                    ),
                ],
                0,
                12,
            ),
        );
    }

    #[test]
    fn test_parse_errors_unexpected_token() {
        // Using dummy content for the expected error variant
        assert_parse_error("(1 2", ParseError::UnexpectedEof("')'".to_string()));
        assert_parse_error(
            "(1 . 2)",
            unexpected_token(TokenKind::Symbol(".".to_string()), 3, 3, "')'".to_string()),
        ); // Assuming dot not handled yet
        assert_parse_error(
            ")",
            unexpected_token(TokenKind::RParen, 0, 0, "an atom or '(' or '''".to_string()),
        ); // Simplified expected msg
        assert_parse_error(
            "(1))",
            unexpected_token(TokenKind::RParen, 4, 4, "end of input".to_string()),
        );
        assert_parse_error(
            "(')",
            unexpected_token(TokenKind::RParen, 2, 2, "an atom or '(' or '''".to_string()),
        ); // After quote, expects expression
        assert_parse_error("(", ParseError::UnexpectedEof("')'".to_string())); // EOF inside list
    }

    #[test]
    fn test_parse_errors_eof() {
        assert_parse_error("", ParseError::UnexpectedEof("".to_string()));
        assert_parse_error("'", ParseError::UnexpectedEof("".to_string())); // EOF after quote
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
            list(
                &[
                    node(Sexpr::Symbol("+".to_string()), 3, 4),
                    node(Sexpr::Number(1.0), 5, 6),
                    node(Sexpr::Number(2.0), 7, 8),
                ],
                1,
                10,
            ),
        );
        assert_parse(
            " ; comment at start\n   'symbol   ; comment at end\n ",
            list(
                &[
                    node(Sexpr::Symbol("quote".to_string()), 23, 24),
                    node(Sexpr::Symbol("symbol".to_string()), 24, 30),
                ],
                23,
                30,
            ),
        );
    }
}
