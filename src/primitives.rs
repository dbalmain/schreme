use crate::{EvalError, EvalResult, Node, Sexpr, Span, types::EvaluatedNodeIterator};

fn arity_error<T>(name: &str, expected: usize, actual: usize, span: Span) -> EvalResult<T> {
    Err(EvalError::InvalidArguments(
        format!(
            "Primitive '{}' expects exactly {} arguments, got {}",
            name, expected, actual
        ),
        span,
    ))
}

fn number_from_node(node: &Node, index: usize, operator: &str) -> EvalResult<f64> {
    match *node.kind.borrow() {
        Sexpr::Number(num) => Ok(num),
        _ => Err(EvalError::InvalidArguments(
            format!(
                "Primitive '{}' expects a list of numbers, got {} as argument {}",
                operator,
                node.kind.borrow().type_name(),
                index
            ),
            node.span,
        )),
    }
}

fn fold_numbers<F: Fn(f64, f64) -> f64>(
    args: EvaluatedNodeIterator,
    span: Span,
    start: f64,
    func: F,
    operator: &str,
) -> EvalResult {
    let mut acc = start;
    for (i, arg_result) in args.enumerate() {
        let arg = arg_result?;
        let num = number_from_node(&arg, i + 1, operator)?;
        acc = func(acc, num);
    }
    Ok(Node::new(Sexpr::Number(acc), span))
}

fn compare_numbers<F: Fn(f64, f64) -> bool>(
    args: EvaluatedNodeIterator,
    span: Span,
    compare: F,
    operator: &str,
) -> EvalResult {
    let mut args = args.enumerate();
    let first_arg = match args.next() {
        Some((_, arg)) => arg?,
        None => return arity_error(operator, 2, 1, span),
    };
    let mut current_num = number_from_node(&first_arg, 1, operator)?;
    for (i, arg_result) in args {
        let arg = arg_result?;
        let num = number_from_node(&arg, i + 1, operator)?;
        if !compare(current_num, num) {
            return Ok(Node::new(Sexpr::Boolean(false), span));
        }
        current_num = num;
    }
    Ok(Node::new(Sexpr::Boolean(true), span))
}

//fn any_node<F: Fn(Node) -> bool>(
//    args: EvaluatedNodeIterator,
//    predicate: F,
//) -> EvalResult<Option<Span>> {
//    for arg_result in args {
//        let arg = arg_result?;
//        if predicate(arg.clone()) {
//            return Ok(Some(arg.span));
//        }
//    }
//    Ok(None)
//}
//
//fn all_nodes<F: Fn(Node) -> bool>(args: EvaluatedNodeIterator, predicate: F) -> EvalResult<bool> {
//    for arg_result in args {
//        let arg = arg_result?;
//        if !predicate(arg.clone()) {
//            return Ok(false);
//        }
//    }
//    Ok(true)
//}

fn all_nodes_at_least_one<F: Fn(Node) -> bool>(
    mut args: EvaluatedNodeIterator,
    name: &str,
    span: Span,
    predicate: F,
) -> EvalResult<bool> {
    // Verify the first argument exists
    match args.next() {
        Some(arg) => {
            if !predicate(arg?.clone()) {
                return Ok(false);
            }
        }
        None => {
            return arity_error(name, 1, 0, span);
        }
    }
    for arg_result in args {
        let arg = arg_result?;
        if !predicate(arg.clone()) {
            return Ok(false);
        }
    }
    Ok(true)
}

pub fn prim_add(args: EvaluatedNodeIterator, span: Span) -> EvalResult {
    // (+) -> 0
    // (+ 1 2 3) -> 6
    fold_numbers(args, span, 0.0, |acc, val| acc + val, "+")
}

pub fn prim_sub(args: EvaluatedNodeIterator, span: Span) -> EvalResult {
    // (- x) -> -x
    // (- x y z) -> x - y - z
    let mut args = args.enumerate();
    let first_arg = match args.next() {
        Some((_, arg)) => arg?,
        None => return arity_error("-", 2, 1, span),
    };
    let mut acc = number_from_node(&first_arg, 1, "-")?;
    let second_arg = match args.next() {
        Some((_, arg)) => arg?,
        None => return Ok(Node::new(Sexpr::Number(-acc), span)),
    };
    let num = number_from_node(&second_arg, 2, "-")?;
    acc -= num;
    for (i, arg_result) in args {
        let num = number_from_node(&arg_result?, i + 1, "-")?;
        acc -= num;
    }
    Ok(Node::new(Sexpr::Number(acc), span))
}

pub fn prim_mul(args: EvaluatedNodeIterator, span: Span) -> EvalResult {
    // (*) -> 1
    // (* 1 2 3) -> 6
    fold_numbers(args, span, 1.0, |acc, val| acc * val, "*")
}

pub fn prim_div(args: EvaluatedNodeIterator, span: Span) -> EvalResult {
    // (/ x) -> 1/x
    // (/ x y z) -> x / y / z

    let div_by_zero_error = |span| {
        Err(EvalError::InvalidArguments(
            "Division by zero: (/ 0)".to_string(),
            span,
        ))
    };

    let mut args = args.enumerate();
    let first_arg = match args.next() {
        Some((_, arg)) => arg?,
        None => return arity_error("/", 2, 1, span),
    };
    let mut acc = number_from_node(&first_arg, 1, "/")?;
    let second_arg = match args.next() {
        Some((_, arg)) => arg?,
        None => {
            return if acc == 0.0 {
                div_by_zero_error(first_arg.span)
            } else {
                Ok(Node::new(Sexpr::Number(1.0 / acc), span))
            };
        }
    };
    let num = number_from_node(&second_arg, 2, "/")?;
    acc /= num;
    for (i, arg_result) in args {
        let num = number_from_node(&arg_result?, i + 1, "/")?;
        acc /= num;
    }
    Ok(Node::new(Sexpr::Number(acc), span))
}

pub fn prim_equals(args: EvaluatedNodeIterator, span: Span) -> EvalResult {
    compare_numbers(args, span, |left, right| left == right, "=")
}

pub fn prim_less_than(args: EvaluatedNodeIterator, span: Span) -> EvalResult {
    compare_numbers(args, span, |left, right| left < right, "<")
}

pub fn prim_less_than_or_equals(args: EvaluatedNodeIterator, span: Span) -> EvalResult {
    compare_numbers(args, span, |left, right| left <= right, "<=")
}

pub fn prim_greater_than(args: EvaluatedNodeIterator, span: Span) -> EvalResult {
    compare_numbers(args, span, |left, right| left > right, ">")
}

pub fn prim_greater_than_or_equals(args: EvaluatedNodeIterator, span: Span) -> EvalResult {
    compare_numbers(args, span, |left, right| left >= right, ">=")
}

// --- List Primitives ---

fn expect_node(
    args: &mut EvaluatedNodeIterator,
    operator: &str,
    index: usize,
    arity: usize,
    span: Span,
) -> EvalResult {
    match args.next() {
        Some(Ok(node)) => Ok(node),
        Some(Err(err)) => Err(err),
        None => arity_error(operator, arity, index, span),
    }
}

fn expect_no_more_args(
    args: &mut EvaluatedNodeIterator,
    operator: &str,
    arity: usize,
    span: Span,
) -> EvalResult<()> {
    match args.next() {
        Some(_) => Err(EvalError::InvalidArguments(
            format!(
                "Primitive '{}' expects exactly {} arguments, got more than that",
                operator, arity
            ),
            span,
        )),
        None => Ok(()),
    }
}

fn expect_one_arg(mut args: EvaluatedNodeIterator, span: Span, name: &str) -> EvalResult {
    let first_arg_node = expect_node(&mut args, name, 1, 1, span)?;
    expect_no_more_args(&mut args, name, 1, span)?;
    Ok(first_arg_node)
}

fn expect_two_args(
    mut args: EvaluatedNodeIterator,
    span: Span,
    name: &str,
) -> EvalResult<(Node, Node)> {
    let first_arg_node = expect_node(&mut args, name, 1, 2, span)?;
    let second_arg_node = expect_node(&mut args, name, 2, 2, span)?;
    expect_no_more_args(&mut args, name, 2, span)?;
    Ok((first_arg_node, second_arg_node))
}

pub fn prim_cons(args: EvaluatedNodeIterator, span: Span) -> EvalResult {
    // (cons item '()) -> '(item)
    // (cons a '(b c)) -> '(a b c)
    // (cons a b) -> '(a . b)
    let (car, cdr) = expect_two_args(args, span, "cons")?;
    Ok(Node::new(Sexpr::Pair(car, cdr), span))
}

pub fn prim_car(args: EvaluatedNodeIterator, span: Span) -> EvalResult {
    // (car '()) -> Error
    // (car a) -> Error // a is not a list
    // (car '(a b c)) -> a
    let first_arg_node = expect_one_arg(args, span, "car")?;
    let first_arg = first_arg_node.kind.borrow();
    match &*first_arg {
        Sexpr::Pair(car, _) => Ok(car.clone()),
        _ => {
            return Err(EvalError::InvalidArguments(
                format!("car: Expected a pair, got {}", first_arg.type_name()),
                first_arg_node.span,
            ));
        }
    }
}

pub fn prim_cdr(args: EvaluatedNodeIterator, span: Span) -> EvalResult {
    // (cdr '()) -> Error
    // (cdr a) -> Error // a is not a list
    // (cdr '(a b c)) -> '(b c)
    let first_arg_node = expect_one_arg(args, span, "cdr")?;
    let first_arg = first_arg_node.kind.borrow();
    match &*first_arg {
        Sexpr::Pair(_, cdr) => Ok(cdr.clone()),
        _ => {
            return Err(EvalError::InvalidArguments(
                format!("car: Expected a pair, got {}", first_arg.type_name()),
                first_arg_node.span,
            ));
        }
    }
}

pub fn prim_cddr(args: EvaluatedNodeIterator, span: Span) -> EvalResult {
    // (cddr '()) -> Error
    // (cddr a) -> Error // a is not a list
    // (cddr '(a b c)) -> '(c)
    let first_arg_node = expect_one_arg(args, span, "cddr")?;
    let first_arg = first_arg_node.kind.borrow();
    match &*first_arg {
        Sexpr::Pair(_, cdr) => match &*cdr.kind.borrow() {
            Sexpr::Pair(_, cddr) => Ok(cddr.clone()),
            _ => Err(EvalError::InvalidArguments(
                "cddr: list too short".to_string(),
                first_arg_node.span,
            )),
        },
        _ => Err(EvalError::InvalidArguments(
            format!("cddr: Expected a pair, got {}", first_arg.type_name()),
            first_arg_node.span,
        )),
    }
}

pub fn prim_cdddr(args: EvaluatedNodeIterator, span: Span) -> EvalResult {
    let first_arg_node = expect_one_arg(args, span, "cdddr")?;
    let first_arg = first_arg_node.kind.borrow();
    match &*first_arg {
        Sexpr::Pair(_, cdr) => match &*cdr.kind.borrow() {
            Sexpr::Pair(_, cddr) => match &*cddr.kind.borrow() {
                Sexpr::Pair(_, cdddr) => Ok(cdddr.clone()),
                _ => Err(EvalError::InvalidArguments(
                    "cdddr: list too short".to_string(),
                    first_arg_node.span,
                )),
            },
            _ => Err(EvalError::InvalidArguments(
                "cdddr: list too short".to_string(),
                first_arg_node.span,
            )),
        },
        _ => Err(EvalError::InvalidArguments(
            format!("cdddr: Expected a pair, got {}", first_arg.type_name()),
            first_arg_node.span,
        )),
    }
}

pub fn prim_cddddr(args: EvaluatedNodeIterator, span: Span) -> EvalResult {
    let first_arg_node = expect_one_arg(args, span, "cddddr")?;
    let first_arg = first_arg_node.kind.borrow();
    match &*first_arg {
        Sexpr::Pair(_, cdr) => match &*cdr.kind.borrow() {
            Sexpr::Pair(_, cddr) => match &*cddr.kind.borrow() {
                Sexpr::Pair(_, cdddr) => match &*cdddr.kind.borrow() {
                    Sexpr::Pair(_, cddddr) => Ok(cddddr.clone()),
                    _ => Err(EvalError::InvalidArguments(
                        "cddddr: list too short".to_string(),
                        first_arg_node.span,
                    )),
                },
                _ => Err(EvalError::InvalidArguments(
                    "cddddr: list too short".to_string(),
                    first_arg_node.span,
                )),
            },
            _ => Err(EvalError::InvalidArguments(
                "cddddr: list too short".to_string(),
                first_arg_node.span,
            )),
        },
        _ => Err(EvalError::InvalidArguments(
            format!("cddr: Expected a pair, got {}", first_arg.type_name()),
            first_arg_node.span,
        )),
    }
}

pub fn prim_cadr(args: EvaluatedNodeIterator, span: Span) -> EvalResult {
    // (cadr '()) -> Error
    // (cadr a) -> Error // a is not a list
    // (cadr '(a b c)) -> b
    let first_arg_node = expect_one_arg(args, span, "cadr")?;
    let first_arg = first_arg_node.kind.borrow();
    match &*first_arg {
        Sexpr::Pair(_, cdr) => match &*cdr.kind.borrow() {
            Sexpr::Pair(cadr, _) => Ok(cadr.clone()),
            _ => Err(EvalError::InvalidArguments(
                "cadr: list too short".to_string(),
                first_arg_node.span,
            )),
        },
        _ => Err(EvalError::InvalidArguments(
            format!("cadr: Expected a pair, got {}", first_arg.type_name()),
            first_arg_node.span,
        )),
    }
}

pub fn prim_caddr(args: EvaluatedNodeIterator, span: Span) -> EvalResult {
    let first_arg_node = expect_one_arg(args, span, "caddr")?;
    let first_arg = first_arg_node.kind.borrow();
    match &*first_arg {
        Sexpr::Pair(_, cdr) => match &*cdr.kind.borrow() {
            Sexpr::Pair(_, cddr) => match &*cddr.kind.borrow() {
                Sexpr::Pair(caddr, _) => Ok(caddr.clone()),
                _ => Err(EvalError::InvalidArguments(
                    "caddr: list too short".to_string(),
                    first_arg_node.span,
                )),
            },
            _ => Err(EvalError::InvalidArguments(
                "caddr: list too short".to_string(),
                first_arg_node.span,
            )),
        },
        _ => Err(EvalError::InvalidArguments(
            format!("caddr: Expected a pair, got {}", first_arg.type_name()),
            first_arg_node.span,
        )),
    }
}

pub fn prim_cadddr(args: EvaluatedNodeIterator, span: Span) -> EvalResult {
    let first_arg_node = expect_one_arg(args, span, "cadddr")?;
    let first_arg = first_arg_node.kind.borrow();
    match &*first_arg {
        Sexpr::Pair(_, cdr) => match &*cdr.kind.borrow() {
            Sexpr::Pair(_, cddr) => match &*cddr.kind.borrow() {
                Sexpr::Pair(_, cdddr) => match &*cdddr.kind.borrow() {
                    Sexpr::Pair(cadddr, _) => Ok(cadddr.clone()),
                    _ => Err(EvalError::InvalidArguments(
                        "cadddr: list too short".to_string(),
                        first_arg_node.span,
                    )),
                },
                _ => Err(EvalError::InvalidArguments(
                    "cadddr: list too short".to_string(),
                    first_arg_node.span,
                )),
            },
            _ => Err(EvalError::InvalidArguments(
                "cadddr: list too short".to_string(),
                first_arg_node.span,
            )),
        },
        _ => Err(EvalError::InvalidArguments(
            format!("cadddr: Expected a pair, got {}", first_arg.type_name()),
            first_arg_node.span,
        )),
    }
}

pub fn prim_set_car(args: EvaluatedNodeIterator, span: Span) -> EvalResult {
    // (define x (cons 1 2)) -> x is (1 . 2)
    // (set-car! x 'new) -> x is ('new . 2)
    let (list, car) = expect_two_args(args, span, "set-car!")?;
    let mut mut_list = list.kind.borrow_mut();
    match *mut_list {
        Sexpr::Pair(ref mut car_field, _) => {
            *car_field = car;
            Ok(Node::new_nil(span))
        }
        _ => {
            drop(mut_list);
            Err(EvalError::TypeMismatch {
                expected: "pair".to_string(),
                found: list.kind.borrow().clone(),
                span,
            })
        }
    }
}

pub fn prim_set_cdr(args: EvaluatedNodeIterator, span: Span) -> EvalResult {
    // (define x (cons 1 2)) -> x is (1 . 2)
    // (set-cdr! x 'new) -> x is (1 . 'new)
    let (list, cdr) = expect_two_args(args, span, "set-cdr!")?;
    let mut mut_list = list.kind.borrow_mut();
    match *mut_list {
        Sexpr::Pair(_, ref mut cdr_field) => {
            *cdr_field = cdr;
            Ok(Node::new_nil(span))
        }
        _ => {
            drop(mut_list);
            Err(EvalError::TypeMismatch {
                expected: "pair".to_string(),
                found: list.kind.borrow().clone(),
                span,
            })
        }
    }
}

pub fn prim_list(args: EvaluatedNodeIterator, span: Span) -> EvalResult {
    // (list item1 item2 ...) -> '(item1 item2 ...)
    let items: Vec<Node> = EvalResult::<Vec<Node>>::from_iter(args)?;

    if items.is_empty() {
        // nil represents the whole node in the case
        return Ok(Node::new_nil(span));
    }

    // We have sime items in the list so put nil at the very end of the list function
    let mut curr = Node::new_nil(Span::new(span.end - 1, span.end));

    // Iterate in reverse to build the list: (cons item (cons item ... nil))
    for item_node in items.into_iter().rev() {
        let span = item_node.span.merge(&curr.span);
        curr = Node::new_pair(item_node, curr, span);
    }

    Ok(curr)
}

// --- Type Predicates ---
macro_rules! type_predicate {
    ($args:expr, $type:pat, $name:expr, $span:expr) => {
        Ok(Node::new_bool(
            all_nodes_at_least_one($args, $name, $span, |arg| {
                matches!(*arg.kind.borrow(), $type)
            })?,
            $span,
        ))
    };
}

pub fn prim_is_null(args: EvaluatedNodeIterator, span: Span) -> EvalResult {
    // (null?) -> error
    // (null? obj) -> true if obj is nil
    // (null? a b c) -> true if a, b and c are all nil
    type_predicate!(args, Sexpr::Nil, "null?", span)
}

pub fn prim_is_pair(args: EvaluatedNodeIterator, span: Span) -> EvalResult {
    // (pair?) -> error
    // (pair? obj) -> true if obj is a pair
    // (pair? a b c) -> true if a, b and c are pairs
    type_predicate!(args, Sexpr::Pair(_, _), "null?", span)
}

pub fn prim_is_number(args: EvaluatedNodeIterator, span: Span) -> EvalResult {
    type_predicate!(args, Sexpr::Number(_), "number?", span)
}

pub fn prim_is_boolean(args: EvaluatedNodeIterator, span: Span) -> EvalResult {
    type_predicate!(args, Sexpr::Boolean(_), "boolean?", span)
}

pub fn prim_is_symbol(args: EvaluatedNodeIterator, span: Span) -> EvalResult {
    type_predicate!(args, Sexpr::Symbol(_), "symbol?", span)
}

pub fn prim_is_string(args: EvaluatedNodeIterator, span: Span) -> EvalResult {
    type_predicate!(args, Sexpr::String(_), "string?", span)
}

pub fn prim_is_procedure(args: EvaluatedNodeIterator, span: Span) -> EvalResult {
    type_predicate!(args, Sexpr::Procedure(_), "procedure?", span)
}

pub fn prim_kind(mut args: EvaluatedNodeIterator, span: Span) -> EvalResult {
    let first_arg_node = expect_node(&mut args, "kind", 1, 1, span)?;
    expect_no_more_args(&mut args, "kind", 1, span)?;
    Ok(Node::new_string(
        first_arg_node.kind.borrow().type_name(),
        span,
    ))
}

pub fn prim_eq(args: EvaluatedNodeIterator, span: Span) -> EvalResult {
    let (left_node, right_node) = expect_two_args(args, span, "eq")?;
    Ok(Node::new_bool(left_node.scheme_eq(&right_node), span))
}

pub fn prim_eqv(args: EvaluatedNodeIterator, span: Span) -> EvalResult {
    let (left_node, right_node) = expect_two_args(args, span, "eq")?;
    Ok(Node::new_bool(left_node.scheme_eqv(&right_node), span))
}

pub fn prim_equal(args: EvaluatedNodeIterator, span: Span) -> EvalResult {
    let (left_node, right_node) = expect_two_args(args, span, "eq")?;
    Ok(Node::new_bool(left_node.scheme_equal(&right_node), span))
}

pub fn prim_and(args: EvaluatedNodeIterator, span: Span) -> EvalResult {
    let mut arg_node = Node::new_bool(true, span);
    for arg in args {
        arg_node = arg?;
        if !arg_node.is_truthy() {
            return Ok(Node::new_bool(false, arg_node.span));
        }
    }
    Ok(arg_node)
}

pub fn prim_or(args: EvaluatedNodeIterator, span: Span) -> EvalResult {
    for arg in args {
        let arg_node = arg?;
        if arg_node.is_truthy() {
            return Ok(arg_node);
        }
    }
    Ok(Node::new_bool(false, span))
}
