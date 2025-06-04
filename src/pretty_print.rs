use crate::{EnvError, EvalError, ParseError};
use ariadne::{Label, Report, ReportKind, Source};

impl EvalError {
    pub fn pretty_print(&self, input: &str) {
        let report = match self {
            EvalError::EnvError(env_error) => match env_error {
                EnvError::UnboundVariable(symbol, span) => {
                    Report::build(ReportKind::Error, ("REPL", span.to_range()))
                        .with_message(format!("Unbound symbol `{}`", symbol))
                        .with_label(
                            Label::new(("REPL", span.to_range()))
                                .with_message("This symbol is not defined in the current scope"),
                        )
                }
            },
            EvalError::NotAProcedure(sexpr, span) => {
                Report::build(ReportKind::Error, ("REPL", span.to_range()))
                    .with_message(format!("Not a procedure: {}", sexpr))
                    .with_label(
                        Label::new(("REPL", span.to_range()))
                            .with_message("This expression cannot be called as a procedure"),
                    )
            }
            EvalError::InvalidArguments(message, span) => {
                Report::build(ReportKind::Error, ("REPL", span.to_range()))
                    .with_message("Invalid arguments:")
                    .with_label(Label::new(("REPL", span.to_range())).with_message(message))
            }
            EvalError::NotASymbol(sexpr, span) => {
                Report::build(ReportKind::Error, ("REPL", span.to_range()))
                    .with_message(format!("Not a symbol: {}", sexpr))
                    .with_label(Label::new(("REPL", span.to_range())).with_message(format!(
                        "Expected a symbol but found a {}",
                        sexpr.type_name()
                    )))
            }
            EvalError::InvalidSpecialForm(message, span) => {
                Report::build(ReportKind::Error, ("REPL", span.to_range()))
                    .with_message(format!("Invalid special form: {}", message))
                    .with_label(
                        Label::new(("REPL", span.to_range()))
                            .with_message("This special form is malformed or incomplete"),
                    )
            }
            EvalError::UnexpectedError(sexpr, span, description) => {
                Report::build(ReportKind::Error, ("REPL", span.to_range()))
                    .with_message(format!("Unexpected error: {}", description))
                    .with_label(
                        Label::new(("REPL", span.to_range()))
                            .with_message(format!("Error occurred while processing: {}", sexpr)),
                    )
            }
            EvalError::TypeMismatch {
                expected,
                found,
                span,
            } => Report::build(ReportKind::Error, ("REPL", span.to_range()))
                .with_message(format!("Type mismatch: {}", span))
                .with_label(Label::new(("REPL", span.to_range())).with_message(format!(
                    "Expected {}, found {}",
                    expected,
                    found.type_name()
                ))),
        };
        report
            .finish()
            .print(("REPL", Source::from(input)))
            .unwrap();
    }
}

impl ParseError {
    pub fn pretty_print(&self, input: &str) {
        let report = match self {
            ParseError::UnexpectedToken { found, expected } => {
                Report::build(ReportKind::Error, ("REPL", found.span.to_range()))
                    .with_message(format!("Unexpected token: {}", found.kind))
                    .with_label(
                        Label::new(("REPL", found.span.to_range()))
                            .with_message(format!("Expected {expected}")),
                    )
            }
            ParseError::UnexpectedEof(expected) => {
                let idx = input.len();
                Report::build(ReportKind::Error, ("REPL", idx..=idx))
                    .with_message("Unexpected EOF")
                    .with_label(Label::new(("REPL", idx..=idx)).with_message(expected))
            }
            ParseError::LexerError(lex_err) => {
                Report::build(ReportKind::Error, ("REPL", lex_err.span.to_range()))
                    .with_message("Lexer Error")
                    .with_label(
                        Label::new(("REPL", lex_err.span.to_range()))
                            .with_message(lex_err.error.to_string()),
                    )
            }
            ParseError::InvalidDotSyntax(span) => {
                Report::build(ReportKind::Error, ("REPL", span.to_range()))
                    .with_message("Invalid Dot Syntax")
                    .with_label(
                        Label::new(("REPL", span.to_range())).with_message("Unexpected dot"),
                    )
            }
        };
        report
            .finish()
            .print(("REPL", Source::from(input)))
            .unwrap();
    }
}
