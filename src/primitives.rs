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

pub fn prim_fold_numbers<F: Fn(f64, f64) -> f64>(
    args: Vec<Node>,
    span: Span,
    start: f64,
    func: F,
    operator: &str,
) -> EvalResult {
    let mut acc = start;
    for (i, node) in args.iter().enumerate() {
        let num = expect_number!(node, span, operator, i + 1);
        acc = func(acc, num);
    }
    // Result needs to be a Node with a span. Let's use the call span.
    Ok(Node {
        kind: Sexpr::Number(acc),
        span,
    })
}

pub fn prim_add(args: Vec<Node>, span: Span) -> EvalResult {
    // (+) -> 0
    // (+ 1 2 3) -> 6
    prim_fold_numbers(args, span, 0.0, |acc, val| acc + val, "+")
}

pub fn prim_sub(args: Vec<Node>, span: Span) -> EvalResult {
    // (- x) -> -x
    // (- x y z) -> x - y - z
    check_arity!(args, min 1, span, "-");
    let first_num = expect_number!(&args[0], span, "-", 1);

    if args.len() == 1 {
        Ok(Node {
            kind: Sexpr::Number(-first_num),
            span,
        })
    } else {
        let mut result = first_num;
        for (i, node) in args.iter().skip(1).enumerate() {
            let num = expect_number!(node, span, "-", i + 2);
            result -= num;
        }
        Ok(Node {
            kind: Sexpr::Number(result),
            span,
        })
    }
}

pub fn prim_mul(args: Vec<Node>, span: Span) -> EvalResult {
    // (*) -> 1
    // (* 1 2 3) -> 6
    prim_fold_numbers(args, span, 1.0, |acc, val| acc * val, "*")
}

pub fn prim_div(args: Vec<Node>, span: Span) -> EvalResult {
    // (/ x) -> 1/x
    // (/ x y z) -> x / y / z
    check_arity!(args, min 1, span, "/");
    let first_num = expect_number!(&args[0], span, "/", 1);
    if first_num == 0.0 && args.len() > 1 {
        // Check potential division by zero if it's not the only arg
        // Or let Rust panic/return Inf/NaN? Let's allow Inf/NaN for now.
    }

    if args.len() == 1 {
        if first_num == 0.0 {
            return Err(EvalError::InvalidArguments(
                "Division by zero: (/ 0)".to_string(),
                span,
            ));
        }
        Ok(Node {
            kind: Sexpr::Number(1.0 / first_num),
            span,
        })
    } else {
        let mut result = first_num;
        for (i, node) in args.iter().skip(1).enumerate() {
            let num = expect_number!(node, span, "/", i + 2);
            if num == 0.0 {
                return Err(EvalError::InvalidArguments(
                    "Division by zero".to_string(),
                    span,
                ));
            }
            result /= num;
        }
        Ok(Node {
            kind: Sexpr::Number(result),
            span,
        })
    }
}

pub fn prim_all_numbers<F: Fn(f64, f64) -> bool>(
    args: Vec<Node>,
    span: Span,
    compare: F,
    operator: &str,
) -> EvalResult {
    // (= n1 n2 ...) -> boolean
    check_arity!(args, min 2, span, operator);
    let mut last_val = expect_number!(&args[0], span, operator, 1);
    let mut result = true;
    for (index, arg) in args.iter().enumerate().skip(1) {
        let val = expect_number!(arg, span, "=", index + 1);
        result = result && compare(last_val, val);
        last_val = val;
    }
    Ok(Node {
        kind: Sexpr::Boolean(result),
        span,
    })
}

pub fn prim_equals(args: Vec<Node>, span: Span) -> EvalResult {
    prim_all_numbers(args, span, |left, right| left == right, "=")
}

pub fn prim_less_than(args: Vec<Node>, span: Span) -> EvalResult {
    prim_all_numbers(args, span, |left, right| left < right, "<")
}

pub fn prim_less_than_or_equals(args: Vec<Node>, span: Span) -> EvalResult {
    prim_all_numbers(args, span, |left, right| left <= right, "<=")
}

pub fn prim_greater_than(args: Vec<Node>, span: Span) -> EvalResult {
    prim_all_numbers(args, span, |left, right| left > right, ">")
}

pub fn prim_greater_than_or_equals(args: Vec<Node>, span: Span) -> EvalResult {
    prim_all_numbers(args, span, |left, right| left >= right, ">=")
}

// --- List Primitives ---

pub fn prim_cons(args: Vec<Node>, span: Span) -> EvalResult {
    // (cons item list) -> new list
    if let [item, list] = &args[..] {
        expect_list_or_nil(list, span, "cons", 2)?;

        let mut new_list_elements: Vec<Node> = vec![item.clone()];

        // Append elements from the existing list_node
        if let Sexpr::List(existing_elements) = &list.kind {
            new_list_elements.extend_from_slice(existing_elements);
        }
        // If list_node.kind was Sexpr::Nil, we just have the list with 'item'

        // Return the new list node
        Ok(Node {
            kind: Sexpr::List(new_list_elements),
            span, // Result span is the call span
        })
    } else {
        arity_error("cons", 2, args.len(), span)
    }
}

pub fn prim_car(args: Vec<Node>, span: Span) -> EvalResult {
    // (car list) -> first item
    if let [list] = &args[..] {
        expect_list_or_nil(list, span, "car", 1)?;
        match &list.kind {
            Sexpr::List(elements) => {
                if elements.is_empty() {
                    // Scheme standard: Error on car of empty list
                    Err(EvalError::InvalidArguments(
                        "car: Cannot take car of empty list".to_string(),
                        list.span, // Span of the empty list argument
                    ))
                } else {
                    // Return a clone of the first element Node
                    Ok(elements[0].clone())
                }
            }
            // R5RS/R6RS error on car of non-pair (which includes Nil for proper lists)
            Sexpr::Nil => Err(EvalError::InvalidArguments(
                "car: Expected a pair/non-empty list, got ()".to_string(),
                list.span,
            )),
            _ => Err(EvalError::InvalidArguments(
                format!(
                    "car: Expected a list or pair, got {}",
                    list.kind.type_name()
                ),
                list.span, // Span of the incorrect argument
            )),
        }
    } else {
        arity_error("car", 2, args.len(), span)
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

pub fn prim_is_null(args: Vec<Node>, span: Span) -> EvalResult {
    // (null? obj) -> boolean
    check_arity!(args, 1, span, "null?");
    let is_null = matches!(args[0].kind, Sexpr::Nil);
    Ok(Node {
        kind: Sexpr::Boolean(is_null),
        span,
    })
}

pub fn prim_is_pair(args: Vec<Node>, span: Span) -> EvalResult {
    // (pair? obj) -> boolean
    // In our current Vec representation, any non-empty Sexpr::List is technically a "pair"
    // conceptually (it has a first element and the rest). Sexpr::Nil is not a pair.
    check_arity!(args, 1, span, "pair?");
    let is_pair = matches!(&args[0].kind, Sexpr::List(elements) if !elements.is_empty());
    Ok(Node {
        kind: Sexpr::Boolean(is_pair),
        span,
    })
}

// Add number?, boolean?, symbol?, string?, procedure? similarly

macro_rules! is_type {
    ($args:expr, $type:pat, $name:expr, $span:expr) => {
        if let [arg] = &$args[..] {
            Ok(Node {
                kind: Sexpr::Boolean(matches!(arg.kind, $type)),
                span: $span,
            })
        } else {
            Err(EvalError::InvalidArguments(
                format!(
                    "Primitive '{}' expects exactly 1 argument, got {}",
                    $name,
                    $args.len()
                ),
                $span,
            ))
        }
    };
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
