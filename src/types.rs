use crate::{evaluator::EvalResult, source::Span};
use std::{
    cell::{Ref, RefCell},
    fmt,
    rc::Rc,
}; // For custom display formatting

#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    pub kind: Rc<RefCell<Sexpr>>, // The actual S-expression data
    pub span: Span,               // The source span it covers
}

impl Node {
    pub fn new(kind: Sexpr, span: Span) -> Self {
        Node {
            kind: Rc::new(RefCell::new(kind)),
            span,
        }
    }
}

impl fmt::Display for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Delegate to Sexpr's Display implementation
        write!(f, "{}", self.kind.borrow())
        // Or potentially include span info in debug formats, but not default Display
    }
}

/// Represents a Scheme S-expression (Symbolic Expression).
/// This enum will be the core data structure for both code (AST) and data.
#[derive(Debug, Clone, PartialEq)] // Add traits for easy debugging, copying, and comparison
pub enum Sexpr {
    Symbol(String),   // e.g., +, variable-name, quote
    Number(f64),      // Using f64 for simplicity for now
    Boolean(bool),    // #t or #f
    String(String),   // For string literals "hello\n"
    Pair(Node, Node), // e.g., (+ 1 2), (define x 10)
    Nil,              // Represents the empty list '()
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
            Sexpr::Pair(head, tail) => {
                write!(f, "({}", head)?;

                // Start iterating with the first tail Node
                let mut current_tail_for_loop: Node = tail.clone();

                loop {
                    // Borrow the Sexpr kind from the current_tail_for_loop's Rc<RefCell<Sexpr>>
                    let kind_of_current_tail: Ref<Sexpr> = current_tail_for_loop.kind.borrow();

                    match &*kind_of_current_tail {
                        Sexpr::Pair(elem_of_pair, next_tail_node) => {
                            write!(f, " {}", elem_of_pair)?;
                            // Update current_tail_for_loop for the next iteration by cloning the next Node.
                            // The borrow `kind_of_current_tail` is *not* on `next_tail_node` yet.
                            // And `current_tail_for_loop` is separate from `next_tail_node` until assignment.
                            let temp_next_tail = next_tail_node.clone();
                            // Explicitly drop the borrow before reassigning to the loop variable
                            // that was involved in the borrow.
                            drop(kind_of_current_tail);
                            current_tail_for_loop = temp_next_tail;
                        }
                        Sexpr::Nil => {
                            // End of a proper list
                            break;
                        }
                        _ => {
                            // Dotted list: the cdr is not a Pair and not Nil.
                            // `current_tail_for_loop` is the Node containing this final Sexpr.
                            write!(f, " . {}", current_tail_for_loop)?;
                            break;
                        }
                    }
                    // If we didn't break, kind_of_current_tail (the Ref guard) is dropped here naturally
                    // before the next loop iteration starts.
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

pub type PrimitiveFunc = fn(Node, Span) -> EvalResult;

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
