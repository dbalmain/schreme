use crate::{evaluator::EvalResult, source::Span};
use std::fmt; // For custom display formatting

#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    pub kind: Sexpr, // The actual S-expression data
    pub span: Span,  // The source span it covers
}

impl Node {
    pub fn new(kind: Sexpr, span: Span) -> Self {
        Node { kind, span }
    }
}

impl fmt::Display for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Delegate to Sexpr's Display implementation
        write!(f, "{}", self.kind)
        // Or potentially include span info in debug formats, but not default Display
    }
}

/// Represents a Scheme S-expression (Symbolic Expression).
/// This enum will be the core data structure for both code (AST) and data.
#[derive(Debug, Clone, PartialEq)] // Add traits for easy debugging, copying, and comparison
pub enum Sexpr {
    Symbol(String),  // e.g., +, variable-name, quote
    Number(f64),     // Using f64 for simplicity for now
    Boolean(bool),   // #t or #f
    String(String),  // For string literals "hello\n"
    List(Vec<Node>), // e.g., (+ 1 2), (define x 10)
    Nil,             // Represents the empty list '()
    Procedure(Procedure),
    // --- Future additions ---
    // Pair(Box<Sexpr>, Box<Sexpr>), // For dotted pairs, alternative list rep
    // Primitive(PrimitiveFunc),
    // Lambda(LambdaExpr),
}

// Implement Display trait for pretty printing the Sexpr values
impl fmt::Display for Sexpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Sexpr::Symbol(s) => write!(f, "{}", s),
            Sexpr::Number(n) => write!(f, "{}", n),
            Sexpr::Boolean(b) => write!(f, "{}", if *b { "#t" } else { "#f" }),
            Sexpr::List(list) => {
                let mut first = true;
                for expr in list {
                    if !first {
                        write!(f, " ")?;
                    }
                    write!(f, "{}", expr)?;
                    first = false;
                }
                write!(f, ")")
            }
            Sexpr::Nil => write!(f, "()"),
            Sexpr::String(str) => {
                write!(
                    f,
                    "\"{}\"",
                    str.chars().fold(String::new(), |mut acc, char| {
                        match char {
                            '"' => acc.push_str("\\\""),
                            '\n' => acc.push_str("\\n"),
                            '\r' => acc.push_str("\\r"),
                            '\t' => acc.push_str("\\t"),
                            c => acc.push(c),
                        }
                        acc
                    })
                )
            }
            Sexpr::Procedure(procedure) => match procedure {
                Procedure::Primitive(_, name) => write!(f, "#<primitive:{}>", name),
            },
        }
    }
}

pub type PrimitiveFunc = fn(Vec<Node>, Span) -> EvalResult;

#[derive(Clone)] // Need Clone for Sexpr::Procedure
pub enum Procedure {
    Primitive(PrimitiveFunc, String), // The function pointer and its name (for display/debug)
                                      // Lambda(LambdaData), // Placeholder for user-defined functions
}

impl fmt::Debug for Procedure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Procedure::Primitive(_, name) => write!(f, "Primitive({})", name),
            // Procedure::Lambda(_) => write!(f, "Lambda(...)"), // Later
        }
    }
}

// Procedures should be comparable based on identity or name if needed,
// but function pointers don't implement PartialEq directly.
// We might need custom PartialEq later if exact procedure comparison is needed.
impl PartialEq for Procedure {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Procedure::Primitive(_f1, n1), Procedure::Primitive(_f2, n2)) => {
                // Compare function pointers for identity and maybe names
                n1 == n2
            } // Add Lambda comparison later
              // _ => false,
        }
    }
}
