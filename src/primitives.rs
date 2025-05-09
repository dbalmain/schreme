use std::rc::Rc;

use crate::{EvalError, EvalResult, Node, Sexpr, Span};

fn arity_error(name: &str, expected: usize, actual: usize, span: Span) -> EvalResult {
    Err(EvalError::InvalidArguments(
        format!(
            "Primitive '{}' expects exactly {} arguments, got {}",
            name, expected, actual
        ),
        span,
    ))
}

fn old_numbers<F: Fn(f64, f64) -> f64>(
    args: Node,
    span: Span,
    start: f64,
    func: F,
    operator: &str,
) -> EvalResult {
    match args.kind.into_inner() {
        Sexpr::Pair(car, cdr) => match car.kind.into_inner() {
            Sexpr::Number(num) => fold_numbers_inner(cdr, span, func(num, start), func, operator),
            _ => invalid_arg_error(&cdr, operator),
        },
        Sexpr::Number(num) => Ok(Node::new(Sexpr::Number(func(start, num)), span)),
        _ => invalid_arg_error(&args, operator),
    }
}

fn fold_numbers_inner<F: Fn(f64, f64) -> f64>(
    args: Node,
    span: Span,
    current: f64,
    func: F,
    operator: &str,
) -> EvalResult {
    match args.kind.into_inner() {
        Sexpr::Pair(car, cdr) => match car.kind.into_inner() {
            Sexpr::Number(num) => fold_numbers_inner(cdr, span, func(current, num), func, operator),
            _ => invalid_arg_error(&cdr, operator),
        },
        Sexpr::Number(num) => Ok(Node::new(Sexpr::Number(func(current, num)), span)),
        Sexpr::Nil => Ok(Node::new(Sexpr::Number(current), span)), // Handle the end of the list
        _ => invalid_arg_error(&args, operator),
    }
}

/*
// Checks the number of arguments
macro_rules! check_arity {
    ($args:expr, $expected:expr, $span:expr, $name:expr) => {
        if $args.len() != $expected {
            return arity_error($name, $expected, $args.len(), $span);
        }
    };
    // Variant for minimum number of args
    ($args:expr, min $expected:expr, $span:expr, $name:expr) => {
        if $args.len() < $expected {
            return Err(EvalError::InvalidArguments(
                format!(
                    "Primitive '{}' expects at least {} arguments, got {}",
                    $name,
                    $expected,
                    $args.len()
                ),
                $span,
            ));
        }
    };
    // Variant for range of args (inclusive)
    ($args:expr, $min:expr, $max:expr, $span:expr, $name:expr) => {
        if !($min..=$max).contains(&$args.len()) {
            return Err(EvalError::InvalidArguments(
                format!(
                    "Primitive '{}' expects between {} and {} arguments, got {}",
                    $name,
                    $min,
                    $max,
                    $args.len()
                ),
                $span,
            ));
        }
    };
}

// Extracts a number from a Node or returns WrongType error
macro_rules! expect_number {
    ($node:expr, $span:expr, $name:expr, $arg_pos:expr) => {
        match $node.kind {
            Sexpr::Number(n) => n,
            _ => {
                return Err(EvalError::InvalidArguments(
                    // Or define a dedicated WrongType error
                    format!(
                        "Primitive '{}' expects a number for argument {}, got {}",
                        $name,
                        $arg_pos,
                        $node.kind.type_name()
                    ),
                    $span, // Use call span for arg type errors
                ));
            }
        }
    };
    // Variant without arg_pos specified
    ($node:expr, $span:expr, $name:expr) => {
        match $node.kind {
            Sexpr::Number(n) => n,
            _ => {
                return Err(EvalError::InvalidArguments(
                    format!(
                        "Primitive '{}' expects number arguments, got {}",
                        $name,
                        $node.kind.type_name()
                    ),
                    $span,
                ))
            }
        }
    };
}

// Helper to check if a Node is a list (Sexpr::List or Sexpr::Nil)
// Returns EvalError if not.
// We might refine this later if we introduce proper Pairs vs Lists vs Nil distinction.
fn expect_list_or_nil(
    node: &Node,
    span: Span,
    prim_name: &str,
    arg_pos: usize,
) -> Result<(), EvalError> {
    match node.kind {
        Sexpr::List(_) | Sexpr::Nil => Ok(()),
        _ => Err(EvalError::InvalidArguments(
            format!(
                "Primitive '{}' expects argument {} to be a list or nil, got {}",
                prim_name,
                arg_pos,
                node.kind.type_name()
            ),
            span, // Use call span
        )),
    }
}
*/

// TODO add arg index
fn invalid_arg_error(node: &Node, operator: &str) -> EvalResult {
    Err(EvalError::InvalidArguments(
        format!(
            "Primitive '{}' expects a list of numbers, got {}",
            operator,
            node.kind.borrow().type_name()
        ),
        node.span,
    ))
}

fn fold_numbers<F: Fn(f64, f64) -> f64>(
    args: Node,
    span: Span,
    start: f64,
    func: F,
    operator: &str,
) -> EvalResult {
    match &*args.kind.borrow() {
        Sexpr::Pair(car, cdr) => match *car.kind.borrow() {
            Sexpr::Number(num) => fold_numbers(cdr.clone(), span, func(num, start), func, operator),
            _ => invalid_arg_error(&cdr, operator),
        },
        Sexpr::Number(num) => Ok(Node::new(Sexpr::Number(func(start, *num)), span)),
        _ => invalid_arg_error(&args, operator),
    }
}

fn compare_numbers_with<F: Fn(f64, f64) -> bool>(
    args: &Node,
    span: Span,
    compare: F,
    operator: &str,
    current: f64,
) -> EvalResult {
    match &*args.kind.borrow() {
        Sexpr::Pair(first, rest) => {
            if let Sexpr::Number(num) = *first.kind.borrow() {
                if !compare(current, num) {
                    Ok(Node::new(Sexpr::Boolean(false), span))
                } else {
                    compare_numbers_with(&rest, span, compare, operator, num)
                }
            } else {
                return invalid_arg_error(&args, operator);
            }
        }
        Sexpr::Nil => Ok(Node::new(Sexpr::Boolean(false), span)),
        Sexpr::Number(num) => Ok(Node::new(Sexpr::Boolean(compare(current, *num)), span)),
        _ => invalid_arg_error(&args, operator),
    }
}

// TODO add arg index
fn compare_numbers<F: Fn(f64, f64) -> bool>(
    args: &Node,
    span: Span,
    compare: F,
    operator: &str,
) -> EvalResult {
    // (= n1 n2 ...) -> boolean
    match &*args.kind.borrow() {
        Sexpr::Pair(first, rest) => {
            if let Sexpr::Number(num) = *first.kind.borrow() {
                compare_numbers_with(rest, span, compare, operator, num)
            } else {
                return invalid_arg_error(&args, operator);
            }
        }
        Sexpr::Number(_) => return arity_error(operator, 2, 1, span),
        _ => {
            return invalid_arg_error(&args, operator);
        }
    }
}

fn any_node<F: Fn(Node) -> bool>(args: Node, predicate: F) -> Option<Span> {
    match *args.kind.borrow() {
        Sexpr::Pair(ref car, ref cdr) => {
            if predicate(car.clone()) {
                Some(car.span)
            } else {
                any_node(cdr.clone(), predicate)
            }
        }
        Sexpr::Nil => None,
        _ => {
            if predicate(args.clone()) {
                Some(args.span)
            } else {
                None
            }
        }
    }
}

pub fn prim_add(args: Node, span: Span) -> EvalResult {
    // (+) -> 0
    // (+ 1 2 3) -> 6
    fold_numbers(args, span, 0.0, |acc, val| acc + val, "+")
}

pub fn prim_sub(args: Node, span: Span) -> EvalResult {
    // (- x) -> -x
    // (- x y z) -> x - y - z
    match *args.kind.borrow() {
        Sexpr::Number(num) => Ok(Node::new(Sexpr::Number(-num), span)),
        Sexpr::Pair(car, cdr) => match *car.kind.borrow() {
            Sexpr::Number(num) => fold_numbers(cdr, span, num, |acc, val| acc - val, operator),
            _ => invalid_arg_error(&cdr, "-"),
        },
        _ => invalid_arg_error(&args, "-"),
    }
}

pub fn prim_mul(args: Node, span: Span) -> EvalResult {
    // (*) -> 1
    // (* 1 2 3) -> 6
    fold_numbers(args, span, 1.0, |acc, val| acc * val, "*")
}

pub fn prim_div(args: Node, span: Span) -> EvalResult {
    let div_by_zero_error = |span| {
        Err(EvalError::InvalidArguments(
            "Division by zero: (/ 0)".to_string(),
            span,
        ))
    };
    match &*args.kind.borrow() {
        Sexpr::Number(num) if *num == 0.0 => div_by_zero_error(span),
        Sexpr::Number(num) => Ok(Node::new(Sexpr::Number(1.0 / num), span)),
        Sexpr::Pair(car, cdr) => match *car.kind.borrow() {
            Sexpr::Number(num) => {
                match any_node(cdr.clone(), |node| {
                    matches!(*node.kind.borrow(), Sexpr::Number(0.0))
                }) {
                    Some(span) => div_by_zero_error(span),
                    None => fold_numbers(args.clone(), span, num, |acc, val| acc / val, "/"),
                }
            }
            _ => invalid_arg_error(&cdr, "-"),
        },
        _ => invalid_arg_error(&args, "-"),
    }
}

pub fn prim_equals(args: &Node, span: Span) -> EvalResult {
    compare_numbers(args, span, |left, right| left == right, "=")
}

pub fn prim_less_than(args: &Node, span: Span) -> EvalResult {
    compare_numbers(args, span, |left, right| left < right, "<")
}

pub fn prim_less_than_or_equals(args: &Node, span: Span) -> EvalResult {
    compare_numbers(args, span, |left, right| left <= right, "<=")
}

pub fn prim_greater_than(args: &Node, span: Span) -> EvalResult {
    compare_numbers(args, span, |left, right| left > right, ">")
}

pub fn prim_greater_than_or_equals(args: &Node, span: Span) -> EvalResult {
    compare_numbers(args, span, |left, right| left >= right, ">=")
}

// --- List Primitives ---

pub fn prim_cons(args: &Node, span: Span) -> EvalResult {
    // (cons item list) -> [item, ..list]
    if let Sexpr::Pair(head, tail) = &*args.kind.borrow() {
        if let Sexpr::Pair(tail, should_be_nil) = &*tail.kind.borrow() {
            if let Sexpr::Nil = *should_be_nil.kind.borrow() {
                // Create a new list with head as the first element and tail as the rest
                Ok(Node::new(Sexpr::Pair(head.clone(), tail.clone()), span))
            } else {
                // TODO: get the actual number of arguments
                arity_error("cons", 2, 3, span)
            }
        } else {
            // (cons a b) => (a . b)
            Ok(Node::new(Sexpr::Pair(head.clone(), tail.clone()), span))
        }
    } else {
        arity_error("cons", 2, 1, span)
    }
}

pub fn prim_car(args: &Node, span: Span) -> EvalResult {
    // (car list) -> first item
    // Check the argment was a pair
    match &*args.kind.borrow() {
        Sexpr::Pair(arg, should_be_nil) => {
            if let Sexpr::Nil = *should_be_nil.kind.borrow() {
                // Check the argment was a pair
                if let Sexpr::Pair(car, _) = &*arg.kind.borrow() {
                    Ok(car.clone())
                } else {
                    Err(EvalError::InvalidArguments(
                        format!(
                            "car: Expected a list or pair, got {}",
                            args.kind.borrow().type_name()
                        ),
                        args.span, // Span of the incorrect argument
                    ))
                }
            } else {
                // TODO: get the actual number of arguments
                arity_error("car", 1, 2, span)
            }
        }
        Sexpr::Nil => arity_error("car", 1, 0, span),
        _ => {
            Err(EvalError::InvalidArguments(
                format!(
                    "car: Expected a list or pair, got {}",
                    args.kind.borrow().type_name()
                ),
                args.span, // Span of the incorrect argument
            ))
        }
    }
}

pub fn prim_cdr(args: Vec<Node>, span: Span) -> EvalResult {
    // (cdr list) -> rest of list
    if let [list] = &args[..] {
        expect_list_or_nil(list, span, "car", 1)?;
        match &list.kind {
            Sexpr::List(elements) => {
                if elements.is_empty() {
                    // Scheme standard: Error on cdr of empty list
                    Err(EvalError::InvalidArguments(
                        "cdr: Cannot take cdr of empty list".to_string(),
                        list.span,
                    ))
                } else if elements.len() == 1 {
                    // cdr of a single-element list is the empty list '()
                    Ok(Node {
                        kind: Sexpr::Nil,
                        span,
                    }) // Result span is call span
                } else {
                    // Create a new Vec containing elements from the second onwards
                    let rest_elements: Vec<Node> = elements[1..].to_vec();
                    Ok(Node {
                        kind: Sexpr::List(rest_elements),
                        span, // Result span is call span
                    })
                }
            }
            // R5RS/R6RS error on cdr of non-pair (which includes Nil for proper lists)
            Sexpr::Nil => Err(EvalError::InvalidArguments(
                "cdr: Expected a pair/non-empty list, got ()".to_string(),
                list.span,
            )),
            _ => Err(EvalError::InvalidArguments(
                format!(
                    "cdr: Expected a list or pair, got {}",
                    list.kind.type_name()
                ),
                list.span,
            )),
        }
    } else {
        arity_error("car", 2, args.len(), span)
    }
}

pub fn prim_list(args: Vec<Node>, span: Span) -> EvalResult {
    // (list item1 item2 ...) -> new list containing items
    // Args are already evaluated nodes
    // We just need to collect the cloned nodes into a new Sexpr::List
    // If args is empty, result is '() -> Sexpr::Nil
    if args.is_empty() {
        Ok(Node {
            kind: Sexpr::Nil,
            span,
        })
    } else {
        Ok(Node {
            kind: Sexpr::List(args), // args is already Vec<Node>, directly use it
            span,                    // Result span is call span
        })
    }
}

// --- Type Predicates ---

pub fn prim_is_null(args: &Node, span: Span) -> EvalResult {
    // (null? obj) -> boolean
    let is_null = matches!(args.kind, Sexpr::Nil);
    Ok(Node {
        kind: Sexpr::Boolean(is_null),
        span,
    })
}

// Add number?, boolean?, symbol?, string?, procedure? similarly

fn expect_one_arg(args: &Node, span: Span, operator: &str) -> EvalResult<&Node> {
    match args.kind {
        Sexpr::Pair(first, should_be_nil) => match should_be_nil.kind {
            Sexpr::Nil => Ok(first),
            _ => arity_error(operator, 1, 2, span),
        },
        _ => Ok(args),
    }
}
macro_rules! is_type {
    ($arg:expr, $type:pat, $name:expr, $span:expr) => {
        match $arg.kind {
            Sexpr::Pair(first, should_be_nil) => {
                match should_be_nil
            }
            kind => Ok(Node {
                kind: Sexpr::Boolean(matches!(kind, $type)),
                span: $span,
            })
        }
    };
}

pub fn prim_is_pair(args: &Node, span: Span) -> EvalResult {
    // (pair? obj) -> boolean
    is_type!(args, Sexpr::Pair(_, _), "pair?", span)
}

pub fn prim_is_number(args: Vec<Node>, span: Span) -> EvalResult {
    is_type!(args, Sexpr::Number(_), "number?", span)
}

pub fn prim_is_boolean(args: Vec<Node>, span: Span) -> EvalResult {
    is_type!(args, Sexpr::Boolean(_), "boolean?", span)
}

pub fn prim_is_symbol(args: Vec<Node>, span: Span) -> EvalResult {
    is_type!(args, Sexpr::Symbol(_), "symbol?", span)
}

pub fn prim_is_string(args: Vec<Node>, span: Span) -> EvalResult {
    is_type!(args, Sexpr::String(_), "string?", span)
}

pub fn prim_is_procedure(args: Vec<Node>, span: Span) -> EvalResult {
    is_type!(args, Sexpr::Procedure(_), "procedure?", span)
}
