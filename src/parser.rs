use crate::Span; // Assuming Sexpr is in types
use crate::lexer::{LexerError, Token, TokenKind}; // Assuming Token is in lexer
use crate::types::{Node, Sexpr};
use std::fmt;
use std::iter::Peekable;
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
    // TODO: can we remove Peekable here?
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
    //fn peek_token(&mut self) -> Option<&Token> {
    //    self.tokens.peek()
    //}

    /// Parses a single S-expression from the token stream.
    pub fn parse_expr_with_token(&mut self, token: Option<Token>) -> ParseResult<Node> {
        match token {
            Some(Token {
                kind: TokenKind::LParen,
                span,
            }) => match self.next_token() {
                Some(Token {
                    kind: TokenKind::RParen,
                    span: rparen_span,
                }) => Ok(Node::new_nil(span.merge(&span.merge(&rparen_span)))),
                Some(token) => {
                    let car_node = self.parse_expr_with_token(Some(token))?;
                    self.parse_list_recursive(span.start, car_node)
                }
                _ => Err(ParseError::UnexpectedEof("')'".to_string())),
            },
            Some(Token {
                kind: TokenKind::Quote,
                span,
            }) => self.parse_quoted_expr("quote", span),
            Some(Token {
                kind: TokenKind::QuasiQuote,
                span,
            }) => self.parse_quoted_expr("quasiquote", span),
            Some(Token {
                kind: TokenKind::Unquote,
                span,
            }) => self.parse_quoted_expr("unquote", span),
            Some(Token {
                kind: TokenKind::UnquoteSplicing,
                span,
            }) => self.parse_quoted_expr("unquote-splicing", span),
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
    fn parse_list_recursive(&mut self, start: usize, car_node: Node) -> ParseResult<Node> {
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
                    }) => Ok(Node::new_pair(
                        car_node,
                        cdr_node,
                        Span::new(start, rparen_span.end),
                    )),
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
                let nil_node = Node::new_nil(rparen_span);
                Ok(Node::new_pair(
                    car_node,
                    nil_node,
                    Span::new(start, rparen_span.end),
                ))
            }
            Some(token) => {
                let next_car_node = self.parse_expr_with_token(Some(token))?;
                let cdr_node =
                    self.parse_list_recursive(next_car_node.span.start, next_car_node)?;
                let end = cdr_node.span.end;
                Ok(Node::new_pair(car_node, cdr_node, Span::new(start, end)))
            }
            None => {
                // Reached EOF before finding ')'
                Err(ParseError::UnexpectedEof("')'".to_string()))
            }
        }
    }

    /// Parses a quoted expression `'expr`.
    fn parse_quoted_expr(&mut self, quote_symbol: &str, quote_span: Span) -> ParseResult<Node> {
        // Parse the expression immediately following the quote
        let quoted_expr = self.parse_expr()?;

        // Construct the equivalent (quote expr) S-expression
        Ok(Node::new_quoted_expr(
            quoted_expr.clone(),
            quote_symbol,
            quote_span,
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

    // Helper function to tokenize and parse a single expression, then get its string representation.
    // This simplifies the test assertions.
    fn assert_parsed_sexpr_string(input: &str, expected_output: &str) {
        let node = match parse_str(input) {
            Ok(result) => result,
            Err(e) => panic!("Parsing failed for input '{}': {}", input, e),
        };
        assert_eq!(node.to_string(), expected_output, "Input: '{}'", input);
    }

    fn node(sexpr: Sexpr, start: usize, end: usize) -> Node {
        Node::new(sexpr, Span::new(start, end))
    }

    fn node_number(n: f64, start: usize, end: usize) -> Node {
        Node::new_number(n, Span::new(start, end))
    }

    fn node_bool(b: bool, start: usize, end: usize) -> Node {
        Node::new_bool(b, Span::new(start, end))
    }

    fn node_string(s: &str, start: usize, end: usize) -> Node {
        Node::new_string(s, Span::new(start, end))
    }

    fn node_nil(start: usize, end: usize) -> Node {
        Node::new_nil(Span::new(start, end))
    }

    fn node_symbol(s: &str, start: usize, end: usize) -> Node {
        Node::new_symbol(s.to_string(), Span::new(start, end))
    }

    fn node_pair(car: Node, cdr: Node, start: usize, end: usize) -> Node {
        Node::new_pair(car, cdr, Span::new(start, end))
    }

    fn node_list(nodes: &[Node], start: usize, end: usize) -> Node {
        match nodes {
            [] => node_nil(end - 1, end),
            [last] => node_pair(last.clone(), node_nil(end - 1, end), start, end),
            [first, rest @ ..] => node_pair(
                first.clone(),
                node_list(rest, rest[0].span.start, end),
                start,
                end,
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
        assert_parse("123", node_number(123.0, 0, 3));
        assert_parse("-4.5", node_number(-4.5, 0, 4));
        assert_parse("symbol", node_symbol("symbol", 0, 6));
        assert_parse("+", node_symbol("+", 0, 1));
        assert_parse("#t", node_bool(true, 0, 2));
        assert_parse("#f", node_bool(false, 0, 2));
        assert_parse(r#""hello world""#, node_string("hello world", 0, 13));
        assert_parse(
            r#""with \"quotes\"""#,
            node_string("with \"quotes\"", 0, 17),
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
            node_list(
                &[
                    node_number(1.0, 1, 2),
                    node_number(2.0, 3, 4),
                    node_number(3.0, 5, 6),
                ],
                0,
                7,
            ),
        );
        assert_parse(
            "(+ 10 20)",
            node_list(
                &[
                    node_symbol("+", 1, 2),
                    node_number(10.0, 3, 5),
                    node_number(20.0, 6, 8),
                ],
                0,
                9,
            ),
        );
        assert_parse(
            "(list #t \"hello\")",
            node_list(
                &[
                    node_symbol("list", 1, 5),
                    node_bool(true, 6, 8),
                    node_string("hello", 9, 16),
                ],
                0,
                17,
            ),
        );
    }

    #[test]
    fn test_parse_dotted_list() {
        assert_parse(
            "(1 . 2)",
            node_pair(node_number(1.0, 1, 2), node_number(2.0, 5, 6), 0, 7),
        );

        assert_parse(
            "(1 2 . 3)",
            node_pair(
                node_number(1.0, 1, 2),
                node_pair(node_number(2.0, 3, 4), node_number(3.0, 7, 8), 3, 9),
                0,
                9,
            ),
        );
    }

    #[test]
    fn test_parse_nested_list() {
        assert_parse(
            "(a (b c) d)",
            node_list(
                &[
                    node_symbol("a", 1, 2),
                    node_list(&[node_symbol("b", 4, 5), node_symbol("c", 6, 7)], 3, 8),
                    node_symbol("d", 9, 10),
                ],
                0,
                11,
            ),
        );
        assert_parse(
            "(()())",
            node_list(&[node(Sexpr::Nil, 1, 3), node(Sexpr::Nil, 3, 5)], 0, 6),
        );
    }

    #[test]
    fn test_parse_quote_sugar() {
        assert_parse(
            "'a",
            Node::new_quote(node_symbol("a", 1, 2), Span::new(0, 1)),
        );
        assert_parse(
            "'123",
            Node::new_quote(node_number(123.0, 1, 4), Span::new(0, 1)),
        );
        assert_parse("'()", Node::new_quote(node_nil(1, 3), Span::new(0, 1)));
        assert_parse(
            "'(1 2)",
            Node::new_quote(
                node_list(&[node_number(1.0, 2, 3), node_number(2.0, 4, 5)], 1, 6),
                Span::new(0, 1),
            ),
        );
        assert_parse(
            "(list 'a 'b)",
            node_list(
                &[
                    node_symbol("list", 1, 5),
                    Node::new_quote(node_symbol("a", 7, 8), Span::new(6, 7)),
                    Node::new_quote(node_symbol("b", 10, 11), Span::new(9, 10)),
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
            node_list(
                &[
                    node_symbol("+", 3, 4),
                    node_number(1.0, 5, 6),
                    node_number(2.0, 7, 8),
                ],
                1,
                10,
            ),
        );
        assert_parse(
            " ; comment at start\n   'symbol   ; comment at end\n ",
            Node::new_quote(node_symbol("symbol", 24, 30), Span::new(23, 24)),
        );
    }

    // --- Tests for Quasiquote (`), Unquote (,), Unquote-Splicing (,@) Parsing ---

    #[test]
    fn test_parse_quasiquote_simple_symbol() {
        assert_parsed_sexpr_string("`foo", "(quasiquote foo)");
    }

    #[test]
    fn test_parse_quasiquote_simple_number() {
        assert_parsed_sexpr_string("`123", "(quasiquote 123)");
    }

    #[test]
    fn test_parse_quasiquote_simple_list() {
        assert_parsed_sexpr_string("`(a b c)", "(quasiquote (a b c))");
    }

    #[test]
    fn test_parse_unquote_simple_symbol() {
        // This would be inside a quasiquote in valid Scheme, but parser should still handle it.
        assert_parsed_sexpr_string(",foo", "(unquote foo)");
    }

    #[test]
    fn test_parse_unquote_simple_list() {
        assert_parsed_sexpr_string(",(a b)", "(unquote (a b))");
    }

    #[test]
    fn test_parse_unquote_splicing_simple_symbol() {
        // This also would be inside a quasiquote.
        assert_parsed_sexpr_string(",@foo", "(unquote-splicing foo)");
    }

    #[test]
    fn test_parse_unquote_splicing_simple_list() {
        assert_parsed_sexpr_string(",@(a b)", "(unquote-splicing (a b))");
    }

    // --- Combinations and Nesting ---

    #[test]
    fn test_parse_quasiquote_with_unquote() {
        assert_parsed_sexpr_string("`(a ,b c)", "(quasiquote (a (unquote b) c))");
    }

    #[test]
    fn test_parse_quasiquote_with_unquote_splicing() {
        assert_parsed_sexpr_string("`(a ,@b c)", "(quasiquote (a (unquote-splicing b) c))");
    }

    #[test]
    fn test_parse_quasiquote_with_multiple_unquotes() {
        assert_parsed_sexpr_string(
            "`(,a ,b ,@c ,d)",
            "(quasiquote ((unquote a) (unquote b) (unquote-splicing c) (unquote d)))",
        );
    }

    #[test]
    fn test_parse_nested_quasiquote() {
        assert_parsed_sexpr_string("``foo", "(quasiquote (quasiquote foo))");
        assert_parsed_sexpr_string(
            "`(a `(b ,c) d)",
            "(quasiquote (a (quasiquote (b (unquote c))) d))",
        );
    }

    #[test]
    fn test_parse_quasiquote_unquote_nested_quasiquote() {
        // `(a ,`(b ,c) d) -> (quasiquote (a (unquote (quasiquote (b (unquote c)))) d))
        assert_parsed_sexpr_string(
            "`(a ,`(b ,c) d)",
            "(quasiquote (a (unquote (quasiquote (b (unquote c)))) d))",
        );
    }

    #[test]
    fn test_parse_quasiquote_with_unquote_splicing_at_start_of_list() {
        assert_parsed_sexpr_string("`(,@a b c)", "(quasiquote ((unquote-splicing a) b c))");
    }

    #[test]
    fn test_parse_quasiquote_with_unquote_splicing_at_end_of_list() {
        assert_parsed_sexpr_string("`(a b ,@c)", "(quasiquote (a b (unquote-splicing c)))");
    }

    #[test]
    fn test_parse_quasiquote_with_only_unquote_splicing() {
        assert_parsed_sexpr_string("`(,@a)", "(quasiquote ((unquote-splicing a)))");
    }

    #[test]
    fn test_parse_quasiquote_dotted_list_with_unquote() {
        assert_parsed_sexpr_string("`(a . ,b)", "(quasiquote (a unquote b))");
    }

    #[test]
    fn test_parse_quasiquote_dotted_list_with_unquote_splicing_in_car() {
        // This is unusual but syntactically parsable by the reader macro rules
        assert_parsed_sexpr_string("`(,@a . b)", "(quasiquote ((unquote-splicing a) . b))");
    }

    #[test]
    fn test_parse_quasiquote_unquote_splicing_in_dotted_cdr_is_error_for_eval_but_parser_might_allow()
     {
        // `(a . ,@b)` is typically an error at evaluation time for unquote-splicing,
        // because ,@ needs to splice into a list context.
        // The parser, however, might just see `,@` and then `b` and form `(unquote-splicing b)`.
        // So the parsed structure could be `(quasiquote (a unquote-splicing b))`.
        // This test checks if the parser produces this structure. The eval error is separate.
        assert_parsed_sexpr_string("`(a . ,@b)", "(quasiquote (a unquote-splicing b))");
    }

    // --- Potential Parser Error Cases (depending on your parser's strictness) ---

    #[test]
    fn test_parse_dangling_backtick() {
        assert!(
            parse_str("`").is_err(),
            "Dangling backtick should be a parser error"
        );
    }

    #[test]
    fn test_parse_dangling_comma() {
        assert!(
            parse_str(",").is_err(),
            "Dangling comma should be a parser error"
        );
    }

    #[test]
    fn test_parse_dangling_comma_at() {
        assert!(
            parse_str(",@").is_err(),
            "Dangling comma-at should be a parser error"
        );
    }
}
