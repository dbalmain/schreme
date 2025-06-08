use crate::{Environment, evaluate, evaluator::EvalResult, source::Span};
use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    fmt,
    rc::Rc,
};

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

    /// Helper to create a tuple node, which is a Pair(X, Pair(Y, Nil)).
    pub fn new_tuple(car: Node, cdr: Node, pair_span: Span) -> Self {
        Node {
            kind: Rc::new(RefCell::new(Sexpr::Pair(
                car,
                Node::new_pair(cdr, Node::new_nil(pair_span.clone()), pair_span),
            ))),
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
        Node::new_quoted_expr(node, "quote", quote_span)
    }

    pub fn new_quoted_expr(node: Node, quote_symbol: &str, quote_span: Span) -> Self {
        let span = quote_span.merge(&node.span);
        Node::new_pair(
            Node::new_symbol(quote_symbol.to_owned(), quote_span),
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

    pub fn into_dotted_iter(self: Node) -> DottedNodeIterator {
        DottedNodeIterator { current_node: self }
    }

    pub fn is_list(self: &Node) -> bool {
        match &*self.kind.borrow() {
            Sexpr::Nil => true,
            Sexpr::Pair(_, _) => true,
            _ => false,
        }
    }

    pub fn is_pair(self: &Node) -> bool {
        match &*self.kind.borrow() {
            Sexpr::Pair(_, _) => true,
            _ => false,
        }
    }

    pub fn is_nil(self: &Node) -> bool {
        match &*self.kind.borrow() {
            Sexpr::Nil => true,
            _ => false,
        }
    }

    pub fn tuple(self: &Node) -> Option<(Node, Node)> {
        match &*self.kind.borrow() {
            Sexpr::Pair(first, rest) => match &*rest.kind.borrow() {
                Sexpr::Pair(second, must_be_nil) => match &*must_be_nil.kind.borrow() {
                    Sexpr::Nil => Some((first.clone(), second.clone())),
                    _ => None,
                },
                _ => None,
            },
            _ => None,
        }
    }

    pub fn pair(self: &Node) -> Option<(Node, Node)> {
        match &*self.kind.borrow() {
            Sexpr::Pair(first, rest) => Some((first.clone(), rest.clone())),
            _ => None,
        }
    }

    pub fn singleton(self: &Node) -> Option<Node> {
        match &*self.kind.borrow() {
            Sexpr::Pair(inner_expr_node, rest) => {
                if !rest.is_nil() {
                    None
                } else {
                    Some(inner_expr_node.clone())
                }
            }
            _ => None,
        }
    }

    pub fn symbol(self: &Node) -> Option<String> {
        match &*self.kind.borrow() {
            Sexpr::Symbol(symbol) => Some(symbol.clone()),
            _ => None,
        }
    }

    pub fn call_name(self: &Node) -> Option<String> {
        match &*self.kind.borrow() {
            Sexpr::Pair(head_node, _rest) => head_node.symbol(),
            _ => None,
        }
    }

    pub fn from_iter_with_dotted_tail<T: Iterator<Item = Node>>(
        nodes: T,
        dotted_tail: Node,
    ) -> Self {
        let nodes: Vec<Node> = nodes.collect();
        let mut current_span = dotted_tail.span.clone();
        let mut current_list_node = dotted_tail;
        // Iterate in reverse to build the list: (cons item (cons item ... nil))
        for item_node in nodes.into_iter().rev() {
            // The `item_node` already has its own span from its source or evaluation.
            // The new `Pair` node we're creating needs a span.
            // Using a synthetic span for the pair structure itself is common.
            current_span = item_node.span.merge(&current_span);
            current_list_node = Node::new_pair(item_node.clone(), current_list_node, current_span);
        }
        current_list_node
    }

    pub fn scheme_eq(&self, other: &Node) -> bool {
        // For eq?, the primary check for heap-allocated Sexprs (via Rc<RefCell>)
        // is whether they point to the same allocation.
        if Rc::ptr_eq(&self.kind, &other.kind) {
            return true;
        }

        let left_kind = self.kind.borrow();
        let right_kind = other.kind.borrow();

        match (&*left_kind, &*right_kind) {
            (Sexpr::String(a), Sexpr::String(b)) => a == b,
            (Sexpr::Symbol(a), Sexpr::Symbol(b)) => a == b,
            (Sexpr::Number(a), Sexpr::Number(b)) => a == b,
            (Sexpr::Boolean(a), Sexpr::Boolean(b)) => a == b,
            (Sexpr::Nil, Sexpr::Nil) => true,
            // For Pairs, Procedures, they must be Rc::ptr_eq to be eq?, which was handled above.
            // If not Rc::ptr_eq, they are not eq?.
            (Sexpr::Pair(_, _), Sexpr::Pair(_, _)) => false,
            (Sexpr::Procedure(_), Sexpr::Procedure(_)) => false,
            _ => false,
        }
    }

    pub fn scheme_eqv(&self, other: &Node) -> bool {
        // for schreme at the moment, eqv? is the same as eq?. eq? will match 100.0 and 100.0
        // even if they're not in the smae memory location.
        self.scheme_eq(other)
    }

    pub fn scheme_equal(&self, other: &Node) -> bool {
        if self.scheme_eqv(other) {
            return true;
        }
        let left_kind = self.kind.borrow();
        let right_kind = other.kind.borrow();

        match (&*left_kind, &*right_kind) {
            (Sexpr::Pair(_, _), Sexpr::Pair(_, _)) => {
                let mut seen: HashSet<(*const RefCell<Sexpr>, *const RefCell<Sexpr>)> =
                    HashSet::new();
                self.schreme_equal_recursive(other, &mut seen)
            }
            // eqv? would have handled any other case
            _ => false,
        }
    }

    fn schreme_equal_recursive(
        &self,
        other: &Node,
        seen: &mut HashSet<(*const RefCell<Sexpr>, *const RefCell<Sexpr>)>,
    ) -> bool {
        if self.scheme_eqv(other) {
            return true;
        }

        let left_kind = self.kind.borrow();
        let right_kind = other.kind.borrow();

        match (&*left_kind, &*right_kind) {
            (Sexpr::Pair(self_car, self_cdr), Sexpr::Pair(other_car, other_cdr)) => {
                // Cycle detection:
                // Get pointers to the Rc<RefCell<Sexpr>> data for self and other.
                let self_ptr = Rc::as_ptr(&self.kind);
                let other_ptr = Rc::as_ptr(&other.kind);

                // The reverse check is not strictly necessary but reduces the risk of stack overflow
                // e.g. `(,@long-a ,@long-b) and `(,@long-b ,@long-a)
                if seen.contains(&(self_ptr, other_ptr)) || seen.contains(&(other_ptr, self_ptr)) {
                    // This means we've encountered a cycle
                    return true;
                }

                // Add this pair of objects to the 'seen' set before recursing
                seen.insert((self_ptr, other_ptr));

                self_car.schreme_equal_recursive(other_car, seen)
                    && self_cdr.schreme_equal_recursive(other_cdr, seen)
            }
            // Add Sexpr::Vector comparison here when you have vectors
            // (Sexpr::Vector(v1), Sexpr::Vector(v2)) => { ... similar logic to pairs ... }

            // If types differ or they are not compound types that equal? handles specially,
            // and they weren't eqv?, then they are not equal?.
            _ => false,
        }
    }
}

impl fmt::Display for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn recursive_pair_fmt(
            node: &Node,
            f: &mut fmt::Formatter<'_>,
            index: isize,
            is_first: bool,
            seen: &mut HashMap<*const RefCell<Sexpr>, isize>,
        ) -> fmt::Result {
            if is_first {}
            let ptr = Rc::as_ptr(&node.kind);
            if let Some(seen_index) = seen.get(&ptr) {
                if !is_first {
                    write!(f, " ")?;
                }
                return write!(f, ". #{}#", seen_index - index);
            }
            match &*node.kind.borrow() {
                Sexpr::Pair(head, tail) => {
                    seen.insert(ptr, index + 1);
                    if !is_first {
                        write!(f, " ")?;
                    }
                    if head.is_list() {
                        if let Some(seen_index) = seen.get(&Rc::as_ptr(&head.kind)) {
                            write!(f, "#{}#", seen_index - index - 1)?;
                        } else {
                            write!(f, "(")?;
                            recursive_pair_fmt(head, f, index + 1, true, seen)?;
                            write!(f, ")")?;
                        }
                    } else {
                        write!(f, "{}", head)?;
                    }
                    recursive_pair_fmt(tail, f, index + 1, false, seen)?;
                    seen.remove(&ptr);
                    Ok(())
                }
                Sexpr::Nil => Ok(()),
                node => write!(f, " . {}", node),
            }
        }

        if self.is_list() {
            write!(f, "(")?;
            recursive_pair_fmt(self, f, 0, true, &mut HashMap::new())?;
            write!(f, ")")
        } else {
            // Delegate to Sexpr's Display implementation
            write!(f, "{}", self.kind.borrow())
        }
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

impl Sexpr {
    pub fn type_name(&self) -> &'static str {
        match self {
            Sexpr::Number(_) => "number",
            Sexpr::Symbol(_) => "symbol",
            Sexpr::Boolean(_) => "boolean",
            Sexpr::String(_) => "string",
            Sexpr::Pair(_, _) => "pair",
            Sexpr::Nil => "nil",
            Sexpr::Procedure(_) => "procedure",
        }
    }
}

pub struct NodeIterator {
    current_node: Node, // Current part of the argument list (Pair or Nil)
}

pub struct DottedNodeIterator {
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

impl Iterator for DottedNodeIterator {
    type Item = (Node, bool); // Each item is the result of evaluating an argument expression

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
                Some((node, false))
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
                Some((node, true))
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
                // note this doesn't handle cycles but the Node fmt function does
                write!(f, "({}", head)?;

                for (node, dotted) in tail.clone().into_dotted_iter() {
                    if dotted {
                        write!(f, " .")?;
                    }
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
