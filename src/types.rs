use crate::{Environment, evaluate, evaluator::EvalResult, source::Span};
use std::{cell::RefCell, fmt, rc::Rc};

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

    /// Helper to create a Nil node with a given span.
    pub fn new_nil(span: Span) -> Self {
        Node {
            kind: Rc::new(RefCell::new(Sexpr::Nil)),
            span,
        }
    }

    /// Helper to create a Pair node. The span for the new Pair node itself
    /// will be synthetic or based on one of its elements.
    pub fn new_pair(car: Node, cdr: Node, pair_span: Span) -> Self {
        Node {
            kind: Rc::new(RefCell::new(Sexpr::Pair(car, cdr))),
            span: pair_span,
        }
    }

    pub fn new_bool(val: bool, span: Span) -> Self {
        Node {
            kind: Rc::new(RefCell::new(Sexpr::Boolean(val))),
            span,
        }
    }

    pub fn new_string(val: &str, span: Span) -> Self {
        Node {
            kind: Rc::new(RefCell::new(Sexpr::String(val.to_owned()))),
            span,
        }
    }

    pub fn new_number(num: f64, span: Span) -> Self {
        Node {
            kind: Rc::new(RefCell::new(Sexpr::Number(num))),
            span,
        }
    }

    pub fn new_symbol(symbol: String, span: Span) -> Self {
        Node {
            kind: Rc::new(RefCell::new(Sexpr::Symbol(symbol))),
            span,
        }
    }

    pub fn new_quote(node: Node, quote_span: Span) -> Self {
        let span = quote_span.merge(&node.span);
        Node::new_pair(
            Node::new_symbol("quote".to_owned(), quote_span),
            Node::new_pair(node, Node::new_nil(span), span),
            span,
        )
    }

    pub fn new_primitive(primitive: PrimitiveFunc, name: &str, span: Span) -> Self {
        Node {
            kind: Rc::new(RefCell::new(Sexpr::Procedure(Procedure::Primitive(
                primitive,
                name.to_owned(),
            )))),
            span,
        }
    }

    /// Creates a new iterator for lazily evaluating arguments from a Scheme list.
    ///
    /// # Arguments
    /// * `env`: The environment in which to evaluate each argument.
    pub fn into_eval_iter(self: Node, env: Rc<RefCell<Environment>>) -> EvaluatedNodeIterator {
        EvaluatedNodeIterator {
            current_node: self,
            env,
        }
    }

    pub fn into_iter(self: Node) -> NodeIterator {
        NodeIterator { current_node: self }
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
}

pub struct NodeIterator {
    current_node: Node, // Current part of the argument list (Pair or Nil)
}

impl Iterator for EvaluatedNodeIterator {
    type Item = EvalResult; // Each item is the result of evaluating an argument expression

    fn next(&mut self) -> Option<Self::Item> {
        let sexpr = self.current_node.kind.borrow();
        match &*sexpr {
            Sexpr::Pair(car_node, cdr_node) => {
                // This is the argument expression we need to evaluate
                let result = evaluate(car_node.clone(), self.env.clone());

                let node = cdr_node.clone();
                drop(sexpr);
                // Advance the iterator's state to the rest of the list for the *next* call
                self.current_node = node;

                // Lazily evaluate the current argument expression
                Some(result)
            }
            Sexpr::Nil => {
                // We've reached the end of a proper list
                None
            }
            _ => {
                let result = evaluate(self.current_node.clone(), self.env.clone());
                // we've encountered a dotted pair (e.g., (a b . c) so return c)
                // It doesn't matter what the span is here, because it's never returned
                drop(sexpr);
                self.current_node = Node::new_nil(Span::default());
                Some(result)
            }
        }
    }
}

impl FromIterator<Node> for Node {
    /// Creates a Scheme list (a `Node` that is `Pair` or `Nil`) from an iterator of `Node`s.
    /// The `Node`s in the iterator are used as the elements of the list.
    /// Spans for the newly created `Pair` structures and the final `Nil` will be synthetic.
    fn from_iter<I: IntoIterator<Item = Node>>(iter: I) -> Self {
        let items: Vec<Node> = iter.into_iter().collect();

        // The span for synthetically created list structure nodes.
        // You might want a more specific span if available (e.g., from the `(list ...)` call itself)
        // but FromIterator is generic and doesn't have that context.
        let span = Span::default();

        let mut current_list_node = Node::new_nil(span.clone());

        // Iterate in reverse to build the list: (cons item (cons item ... nil))
        for item_node in items.into_iter().rev() {
            // The `item_node` already has its own span from its source or evaluation.
            // The new `Pair` node we're creating needs a span.
            // Using a synthetic span for the pair structure itself is common.
            // Alternatively, you could try to use `item_node.span.clone()`, but
            // `synthetic_span` is cleaner for the structural part.
            current_list_node = Node::new_pair(item_node, current_list_node, span.clone());
        }

        // If items was empty, current_list_node is still the initial Nil node.
        // If items was not empty, the outermost pair's span is `synthetic_span`.
        // If you want the overall list node to have a span derived from its contents
        // or a call site, that would need to be adjusted *after* collection.
        // For example, if `items` is not empty, you could set:
        // if let Some(first_item) = items.first() {
        //     current_list_node.span = first_item.span.clone(); // Or some combination
        // }
        // However, for a generic FromIterator, keeping synthetic_span for the structure is fine.

        current_list_node
    }
}

pub struct EvaluatedNodeIterator {
    current_node: Node,            // Current part of the argument list (Pair or Nil)
    env: Rc<RefCell<Environment>>, // Environment for evaluation
}

impl Iterator for NodeIterator {
    type Item = Node; // Each item is the result of evaluating an argument expression

    fn next(&mut self) -> Option<Self::Item> {
        let sexpr = self.current_node.kind.borrow();
        match &*sexpr {
            Sexpr::Pair(car_node, cdr_node) => {
                let node = car_node.clone();
                let next_node = cdr_node.clone();
                drop(sexpr);
                // Advance the iterator's state to the rest of the list for the *next* call
                self.current_node = next_node;

                // Lazily evaluate the current argument expression
                Some(node)
            }
            Sexpr::Nil => {
                // We've reached the end of a proper list
                None
            }
            _ => {
                let node = self.current_node.clone();
                // we've encountered a dotted pair (e.g., (a b . c) so return c)
                // It doesn't matter what the span is here, because it's never returned
                drop(sexpr);
                self.current_node = Node::new_nil(Span::default());
                Some(node)
            }
        }
    }
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

                for node in tail.clone().into_iter() {
                    write!(f, " {}", node)?;
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
                Procedure::Lambda(lambda) => write!(f, "#<lambda:{:?}>", lambda),
            },
        }
    }
}

pub type PrimitiveFunc = fn(EvaluatedNodeIterator, Span) -> EvalResult;

#[derive(Clone, Debug, PartialEq)] // Ensure PartialEq if needed for Procedure
pub struct Lambda {
    pub params: Vec<String>,
    pub variadic_param: Option<String>,
    pub body: Vec<Node>, // list of body expressions e.g. ((+ x y)) or ((expr1) (expr2))
    pub env: Rc<RefCell<Environment>>, // The *captured* lexical environment
}

#[derive(Clone)] // Need Clone for Sexpr::Procedure
pub enum Procedure {
    Primitive(PrimitiveFunc, String), // The function pointer and its name (for display/debug)
    Lambda(Rc<Lambda>),               // New: Rc for cheap cloning/sharing of lambda objects
}

impl fmt::Debug for Procedure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Procedure::Primitive(_, name) => write!(f, "Primitive({})", name),
            procedure => write!(f, "Lambda({:?})", procedure),
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
            }
            (Procedure::Lambda(lambda1), Procedure::Lambda(lambda2)) => lambda1 == lambda2,
            _ => false,
        }
    }
}
