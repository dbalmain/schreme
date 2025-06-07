use crate::environment::{EnvError, Environment};
use crate::source::Span;
use crate::types::{EvaluatedNodeIterator, Lambda, Node, Procedure, Sexpr};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::rc::Rc;

// --- Evaluation Error ---
#[derive(Debug, Clone, PartialEq)]
pub enum EvalError {
    EnvError(EnvError),                   // Errors from environment lookup
    NotAProcedure(Sexpr, Span),           // Tried to call something that isn't a procedure
    InvalidArguments(String, Span),       // Mismatched arity or wrong type of args
    NotASymbol(Sexpr, Span),              // Expected a symbol (e.g., for define/set!)
    InvalidSpecialForm(String, Span),     // Malformed special form (e.g., (if cond))
    UnexpectedError(Sexpr, Span, String), // Expected a list for procedure call or special form
    TypeMismatch {
        expected: String,
        found: Sexpr,
        span: Span,
    }, // Type of argument was incorrect
}

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // TODO: Improve error display using spans and source code later
        match self {
            EvalError::EnvError(env_err) => write!(f, "{}", env_err), // Delegate to EnvError display
            EvalError::NotAProcedure(sexpr, _span) => write!(
                f,
                "Evaluation Error: Expected a procedure, but got: {}",
                sexpr
            ),
            EvalError::InvalidArguments(msg, _span) => {
                write!(f, "Evaluation Error: Invalid arguments - {}", msg)
            }
            EvalError::UnexpectedError(sexpr, _span, description) => {
                write!(f, "Unexpected Error: {}: {}", description, sexpr)
            }
            EvalError::NotASymbol(sexpr, _span) => {
                write!(f, "Evaluation Error: Expected a symbol, but got: {}", sexpr)
            }
            EvalError::InvalidSpecialForm(msg, _span) => {
                write!(f, "Evaluation Error: Invalid special form - {}", msg)
            }
            EvalError::TypeMismatch {
                expected,
                found,
                span: _span,
            } => {
                write!(
                    f,
                    "Evaluation Error: TypeMispatch, expected {} but found {}",
                    expected, found
                )
            }
        }
    }
}

impl std::error::Error for EvalError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            EvalError::EnvError(env_err) => Some(env_err),
            _ => None,
        }
    }
}

// Allow easy conversion from EnvError
impl From<EnvError> for EvalError {
    fn from(err: EnvError) -> Self {
        EvalError::EnvError(err)
    }
}

// Result type alias for convenience
pub type EvalResult<T = Node> = Result<T, EvalError>;

// --- Evaluate Function ---

/// Evaluates a given AST Node within the specified environment.
pub fn evaluate(node: Node, env: Rc<RefCell<Environment>>) -> EvalResult {
    // Use a loop for Tail Call Optimization (TCO) later if needed (trampoline)
    // For now, simple recursive calls are fine.
    // let current_node = node;
    // let current_env = env;
    // loop { ... }

    match &*node.kind.borrow() {
        // 1. Self-evaluating atoms: Numbers, Strings, Booleans, Nil
        Sexpr::Number(_)
        | Sexpr::String(_)
        | Sexpr::Boolean(_)
        | Sexpr::Nil
        | Sexpr::Procedure(_) => {
            Ok(node.clone()) // Return the node itself (or a clone if ownership is an issue)
        }

        // 2. Symbols: Look up in the environment
        Sexpr::Symbol(name) => {
            // Use the symbol's span for error reporting if lookup fails
            Ok(env.borrow().get(name, node.span)?) // Propagate EnvError via From trait
        }

        // 3. Lists: Could be special forms or procedure calls
        Sexpr::Pair(first, rest) => {
            // Check the first element to see if it's a special form or procedure call
            match &*first.kind.borrow() {
                Sexpr::Symbol(sym_name) => match sym_name.as_str() {
                    "quote" => evaluate_quote(rest, node.span),
                    "quasiquote" => evaluate_quasiquote(rest, env, node.span),
                    "if" => evaluate_if(rest, env, node.span),
                    "define" => evaluate_define(rest, env, node.span),
                    "set!" => evaluate_set_bang(rest, env, node.span),
                    "lambda" => evaluate_lambda(rest, env, node.span),
                    "begin" => evaluate_body(rest.clone().into_iter(), env),
                    "let" => evaluate_let(rest, env, node.span),
                    "letrec" => evaluate_letrec(rest, env, node.span),
                    "let*" => evaluate_letstar(rest, env, node.span),
                    "unquote" | "unquote-splicing" => {
                        return invalid_special_form(
                            &format!(
                                "{}: unquote and unquote-splicing must be used inside quasiquote",
                                sym_name
                            ),
                            node.span,
                        );
                    }
                    "apply" => evaluate_apply(rest, env, node.span),
                    _ => evaluate_procedure(first, rest, env, node.span),
                },

                // 3f. Procedure Call (Implement later)
                _ => evaluate_procedure(first, rest, env, node.span),
            }
        } // Handle other Sexpr kinds if they exist (e.g., Pair, Vector) later
    }
}

pub fn special_form_identifiers() -> HashSet<String> {
    [
        "apply",
        "begin",
        "define",
        "if",
        "lambda",
        "let",
        "let*",
        "letrec",
        "quasiquote",
        "quote",
        "set!",
        "unquote",
        "unquote-splicing",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

fn invalid_special_form(error: &str, span: Span) -> EvalResult {
    Err(EvalError::InvalidSpecialForm(error.to_string(), span))
}

/// Evaluates the 'if' special form.
/// `operands` is the Node whose kind is Pair(condition, Pair(consequent, Pair(alternate, Nil)))
/// or Pair(condition, Pair(consequent, Nil)).
/// `env` is the current environment.
/// `original_if_span` is the span of the entire (if ...) expression.
fn evaluate_if(
    operands: &Node,
    env: Rc<RefCell<Environment>>,
    original_if_span: Span,
) -> EvalResult {
    let operands: Vec<Node> = operands.clone().into_iter().collect();

    let operand_count = operands.len();

    // 1. Check the arity of the operands
    if operand_count < 2 || operand_count > 3 {
        // Check if there are at least 2 operands (condition and consequent)
        return invalid_special_form(
            &format!(
                "if: Too {} arguments [{}], expected (if cond consequent [alternate])",
                if operand_count < 2 { "few" } else { "many" },
                operand_count
            ),
            original_if_span,
        );
    }
    let mut operands = operands.into_iter();

    let condition = operands.next().unwrap();
    let consequent = operands.next().unwrap();
    // 2. Evaluate the condition
    let condition_result = evaluate(condition, env.clone())?;

    // Determine truthiness
    let is_truthy = match &*condition_result.kind.borrow() {
        // Borrow to check the kind
        Sexpr::Boolean(false) => false,
        _ => true,
    };

    if is_truthy {
        // 3a. Condition is true: evaluate the consequent
        evaluate(consequent, env.clone())
    } else {
        // 3b. Condition is false: extract and evaluate the alternate, or return Nil
        match operands.next() {
            // Clone Sexpr
            Some(alternate) => evaluate(alternate, env.clone()),
            None => Ok(Node::new_nil(original_if_span)),
        }
    }
}

fn evaluate_quote(operands: &Node, original_quote_span: Span) -> EvalResult {
    let operands: Vec<Node> = operands.clone().into_iter().collect();

    match &operands[..] {
        [] => {
            return invalid_special_form(
                "quote: No arguments provided, expected (quote expr)",
                original_quote_span,
            );
        }
        [node] => Ok(node.clone()),
        [_, ..] => {
            return invalid_special_form(
                &format!(
                    "quote: Too many arguments [{}], expected (quote expr)",
                    operands.len()
                ),
                original_quote_span,
            );
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum QuasiQuoteResult {
    Splice(Node),
    Insert(Node),
}

fn evaluate_quasiquote_recursive(
    expr: Node,
    env: Rc<RefCell<Environment>>,
    depth: u32,                // Initial call from evaluate_quasiquote will use depth 1
    original_call_span: &Span, // Span of the top-level quasiquote form for errors
) -> EvalResult<QuasiQuoteResult> {
    if !expr.is_pair() {
        // If expr is not a pair, it is an atom (symbol, number, etc.)
        // Return it as is, since atoms are treated literally in quasiquote.
        return Ok(QuasiQuoteResult::Insert(expr.clone()));
    }

    // expr is a pair, so we need to process it recursively
    let mut results: Vec<Node> = Vec::new();
    let expr_span = expr.span.clone();

    if let Some(name) = expr.call_name()
        && (name == "unquote" || name == "unquote-splicing" || name == "quasiquote")
    {
        let mut items = expr.into_dotted_iter();
        items.next(); // skip the first item which we already checked
        let Some((arg, is_dotted)) = items.next() else {
            return Err(EvalError::InvalidSpecialForm(
                format!("{} expects exactly one argument but received none", name),
                original_call_span.clone(),
            ));
        };
        if is_dotted {
            return Err(EvalError::InvalidSpecialForm(
                format!("{} does not support dotted arguments", name),
                original_call_span.clone(),
            ));
        }
        if let Some(_) = items.next() {
            return Err(EvalError::InvalidSpecialForm(
                format!("{} expects exactly one argument but received more", name),
                original_call_span.clone(),
            ));
        }
        if name == "quasiquote" || depth > 1 {
            if let QuasiQuoteResult::Insert(processed_nested_expr) = evaluate_quasiquote_recursive(
                arg.clone(),
                env,
                if name == "quasiquote" {
                    depth + 1
                } else {
                    depth - 1
                }, // Increase depth for nested quasiquote
                original_call_span,
            )? {
                return Ok(QuasiQuoteResult::Insert(Node::new_quoted_expr(
                    processed_nested_expr,
                    &name,
                    expr_span,
                )));
            } else {
                return Err(EvalError::UnexpectedError(
                    arg.kind.borrow().clone(),
                    original_call_span.clone(),
                    format!("{} nested quasiquote should never return a splice", name),
                ));
            }
        } else if name == "unquote" {
            return evaluate(arg, env).map(QuasiQuoteResult::Insert);
        } else if name == "unquote-splicing" {
            return evaluate(arg, env).map(QuasiQuoteResult::Splice);
        } else {
            return Err(EvalError::UnexpectedError(
                arg.kind.borrow().clone(),
                original_call_span.clone(),
                format!("{} this block should be unreachable", name),
            ));
        }
    }

    let mut dotted_tail: Option<Node> = None;
    for (item, is_dotted) in expr.into_dotted_iter() {
        if dotted_tail.is_some() {
            return Err(EvalError::InvalidSpecialForm(
                "quasiquote: cannot have undotted after dotted tail".to_string(),
                expr_span.clone(),
            ));
        }
        if item.is_pair() {
            // If item is a pair, we need to process it recursively
            match evaluate_quasiquote_recursive(item, env.clone(), depth, original_call_span)? {
                QuasiQuoteResult::Insert(evaluated_item) => results.push(evaluated_item),
                QuasiQuoteResult::Splice(splice_items) => {
                    // If item is a splice, we need to extend results with the spliced items
                    if !splice_items.is_list() {
                        return Err(EvalError::TypeMismatch {
                            expected: "list".to_string(),
                            found: splice_items.kind.borrow().clone(),
                            span: splice_items.span.clone(),
                        });
                    }
                    for (splice_item, is_dotted) in splice_items.into_dotted_iter() {
                        if is_dotted {
                            dotted_tail = Some(splice_item);
                        } else {
                            results.push(splice_item);
                        }
                    }
                }
            }
        } else {
            // If item is an atom, we can just push it as is
            if is_dotted {
                dotted_tail = Some(item);
            } else {
                results.push(item);
            }
        }
    }

    // There is an edge case where the quote appears as the right element of a dotted pair.
    // for example, `(1 . ,x) is expanded as (quasiquote (1 . (quote x))) which is equivalent
    // to (quasiquote (1 quote x)). If this is not a dotted list and the second last element
    // is quote, quote-splicing or quasiquote, we need to handle it.
    if dotted_tail.is_none()
        && results.len() > 1
        && let Some(name_node) = results.get(results.len() - 2)
        && let Some(name) = name_node.symbol()
        && (name == "unquote" || name == "unquote-splicing" || name == "quasiquote")
    {
        let expr_node = results.pop().unwrap();
        let name_node = results.pop().unwrap();
        let span = name_node.span.merge(&expr_node.span);
        match evaluate_quasiquote_recursive(
            Node::new_tuple(name_node, expr_node, span),
            env.clone(),
            depth,
            original_call_span,
        )? {
            QuasiQuoteResult::Insert(evaluated_item) => dotted_tail = Some(evaluated_item),
            QuasiQuoteResult::Splice(_) => {
                return Err(EvalError::InvalidSpecialForm(
                    "unquote-splicing (,@) cannot appear in dotted tail".to_string(),
                    span,
                ));
            }
        }
    }

    Ok(QuasiQuoteResult::Insert(if dotted_tail.is_some() {
        Node::from_iter_with_dotted_tail(results.into_iter(), dotted_tail.unwrap())
    } else {
        results.into_iter().collect()
    }))
}

// This is the entry point called from your main evaluate function
pub fn evaluate_quasiquote(
    args_node: &Node, // This is the `datum` part from `(quasiquote datum)`
    env: Rc<RefCell<Environment>>,
    span: Span, // Span of the (quasiquote ...) form itself
) -> EvalResult {
    if let Some(expr) = args_node.singleton() {
        // If args_node is a singleton, we can directly process it
        match evaluate_quasiquote_recursive(expr.clone(), env, 1, &span)? {
            QuasiQuoteResult::Insert(processed_expr) => {
                // Return the processed expression wrapped in a Node
                Ok(processed_expr)
            }
            QuasiQuoteResult::Splice(_) => {
                // If we get a splice here, it means we tried to unquote-splice at the top level,
                // which is not allowed in quasiquote.
                Err(EvalError::InvalidSpecialForm(
                    "unquote-splicing (,@) must appear within a list context in quasiquote"
                        .to_string(),
                    span,
                ))
            }
        }
    } else {
        Err(EvalError::InvalidSpecialForm(
            "quasiquote expects exactly one argument".to_string(),
            span,
        ))
    }
}

fn evaluate_define(
    operands: &Node,
    env: Rc<RefCell<Environment>>,
    original_span: Span,
) -> EvalResult {
    match &*operands.kind.borrow() {
        Sexpr::Symbol(name) => {
            let value = Node::new_nil(original_span);
            env.borrow_mut().define(name.clone(), value.clone());
            Ok(value)
        }
        Sexpr::Pair(name_node, definition) => match &*name_node.kind.borrow() {
            Sexpr::Symbol(name) => {
                let value = match &*definition.kind.borrow() {
                    Sexpr::Pair(value_node, should_be_nil) => {
                        if Sexpr::Nil == *should_be_nil.kind.borrow() {
                            evaluate(value_node.clone(), env.clone())?
                        } else {
                            return invalid_special_form(
                                "define: Too many arguments provided, expected (define name [value])",
                                original_span,
                            );
                        }
                    }
                    _ => evaluate(definition.clone(), env.clone())?,
                };
                env.borrow_mut().define(name.clone(), value.clone());
                Ok(value)
            }
            Sexpr::Pair(name_node, args) => {
                if let Sexpr::Symbol(name) = &*name_node.kind.borrow() {
                    let span = args.span.merge(&definition.span);
                    let lambda = evaluate_lambda(
                        &Node::new_pair(args.clone(), definition.clone(), span),
                        env.clone(),
                        original_span,
                    )?;
                    env.borrow_mut().define(name.clone(), lambda.clone());
                    Ok(name_node.clone())
                } else {
                    Err(EvalError::NotASymbol(
                        name_node.kind.borrow().clone(),
                        name_node.span,
                    ))
                }
            }
            _ => Err(EvalError::NotASymbol(
                name_node.kind.borrow().clone(),
                name_node.span,
            )),
        },
        Sexpr::Nil => invalid_special_form(
            "define: No arguments provided, expected (define name [value]) or (define (name ...args) expr ...expr)",
            original_span,
        ),
        _ => Err(EvalError::NotASymbol(
            operands.kind.borrow().clone(),
            operands.span,
        )),
    }
}

fn evaluate_apply(
    operands: &Node,
    env: Rc<RefCell<Environment>>,
    original_span: Span,
) -> EvalResult {
    let operands = &*operands.kind.borrow();
    match operands {
        Sexpr::Pair(operator_node, args) => {
            if !args.is_list() {
                return Err(EvalError::InvalidArguments(
                    "Primitive 'apply' expects at least 2 arguments, got dotted pair".to_string(),
                    original_span,
                ));
            }
            let mut arg_collector: Vec<Node> = vec![];
            for (arg, is_dotted) in args.clone().into_dotted_iter() {
                if is_dotted {
                    return Err(EvalError::InvalidArguments(
                        "Primitive 'apply' does not accept dotted list".to_string(),
                        arg.span,
                    ));
                }
                arg_collector.push(arg);
            }
            if let Some(args) = arg_collector.pop() {
                let mut args = evaluate(args, env.clone())?;
                if !args.is_list() {
                    return Err(EvalError::InvalidArguments(
                        "Primitive 'apply' expects the last argument to be a list".to_string(),
                        args.span,
                    ));
                }
                for arg in arg_collector.into_iter().rev() {
                    let span = args.span.merge(&arg.span);
                    args = Node::new_pair(arg, args, span);
                }
                evaluate(
                    Node::new_pair(operator_node.clone(), args, original_span),
                    env,
                )
            } else {
                return Err(EvalError::InvalidArguments(
                    "Primitive 'apply' expects at least 2 arguments, got 1".to_string(),
                    original_span,
                ));
            }
        }
        Sexpr::Nil => Err(EvalError::InvalidArguments(
            "Primitive 'apply' expects at least 2 arguments, got 0".to_string(),
            original_span,
        )),
        _ => Err(EvalError::InvalidArguments(
            "Primitive 'apply' expects at least 2 arguments, got dotted pair".to_string(),
            original_span,
        )),
    }
}

fn evaluate_set_bang(
    operands: &Node,
    env: Rc<RefCell<Environment>>,
    original_span: Span,
) -> EvalResult {
    match &*operands.kind.borrow() {
        Sexpr::Pair(name_node, definition) => match &*name_node.kind.borrow() {
            Sexpr::Symbol(name) => {
                let value = match &*definition.kind.borrow() {
                    Sexpr::Pair(value_node, should_be_nil) => {
                        if Sexpr::Nil == *should_be_nil.kind.borrow() {
                            evaluate(value_node.clone(), env.clone())?
                        } else {
                            return invalid_special_form(
                                "set: Too many arguments provided, expected (set! name value)",
                                original_span,
                            );
                        }
                    }
                    _ => {
                        return invalid_special_form(
                            "set!: expected a value (set! name value)",
                            original_span,
                        );
                    }
                };
                env.borrow_mut().set(name, value.clone(), name_node.span)?;
                Ok(value)
            }
            _ => Err(EvalError::NotASymbol(
                name_node.kind.borrow().clone(),
                name_node.span,
            )),
        },
        _ => invalid_special_form("set!: expected (set! name value)", original_span),
    }
}

fn evaluate_lambda(
    operands: &Node,
    env: Rc<RefCell<Environment>>,
    original_span: Span,
) -> EvalResult {
    let mut params: Vec<String> = vec![];
    let mut variadic_param: Option<String> = None;
    match &*operands.kind.borrow() {
        Sexpr::Pair(params_node, body_node_list) => {
            // params_node: Node for (param1 param2 ...) or just param for variadic
            // body_node_list: Node for ((expr1) (expr2) ...) or (expr1)

            // Validate params_node:
            // - Must be a list of symbols, or a single symbol (for variadic),
            //   or a dotted list of symbols.
            // - For now, start by requiring a proper list of unique symbols.
            //   e.g., iterate params_node, ensure each element is a Symbol.
            //   Return SyntaxError if not.
            // Example (simplified validation - make this more robust):
            match &*params_node.kind.borrow() {
                Sexpr::Nil => {} // (lambda () ...) is valid
                Sexpr::Symbol(param) => {
                    variadic_param = Some(param.to_string());
                } // (lambda x ...) variadic is valid
                Sexpr::Pair(_, _) => {
                    // Collect parameter names
                    let mut temp_params = params_node.clone();
                    loop {
                        let temp_params_borrow = temp_params.kind.borrow();
                        match &*temp_params_borrow {
                            Sexpr::Pair(param, rest) => {
                                if !matches!(*param.kind.borrow(), Sexpr::Symbol(_)) {
                                    return Err(EvalError::NotASymbol(
                                        param.kind.borrow().clone(),
                                        param.span.clone(),
                                    ));
                                }
                                params.push(param.to_string());
                                let rest = rest.clone(); // clone here so we can drop temp_params_borrow
                                drop(temp_params_borrow);
                                temp_params = rest;
                            }
                            Sexpr::Nil => break,
                            Sexpr::Symbol(param) => {
                                // Dotted list for variadic
                                variadic_param = Some(param.to_string());
                                break;
                            }
                            sexpr => {
                                return Err(EvalError::NotASymbol(
                                    sexpr.clone(),
                                    params_node.span.clone(),
                                ));
                            }
                        }
                    }
                }
                _ => {
                    return Err(EvalError::InvalidArguments(
                        "lambda parameters must be a list or a symbol".to_string(),
                        params_node.span.clone(),
                    ));
                }
            }

            // Validate body_node_list:
            // - Must contain at least one expression.
            let body: Vec<Node> = body_node_list.clone().into_iter().collect();
            if body.len() == 0 {
                return Err(EvalError::InvalidSpecialForm(
                    "lambda body cannot be empty".to_string(),
                    body_node_list.span.clone(),
                ));
            }

            let lambda_object = Lambda {
                params,
                variadic_param,
                body,             // Store the list of body expressions
                env: env.clone(), // CAPTURE THE CURRENT ENVIRONMENT! Crucial for lexical scope.
            };

            Ok(Node {
                kind: Rc::new(RefCell::new(Sexpr::Procedure(Procedure::Lambda(Rc::new(
                    lambda_object,
                ))))),
                span: original_span, // Span of the (lambda ...) form
            })
        }
        _ => Err(EvalError::InvalidSpecialForm(
            // e.g. (lambda) or (lambda not-a-list)
            "lambda: expected (lambda <formals> <body>...)".to_string(),
            operands.span,
        )),
    }
}

fn evaluate_procedure(
    operator_node: &Node,      // The Node from the AST that represents the operator
    operands_list_node: &Node, // The Node from the AST representing the list of operands, e.g. kind is Pair(arg1, Pair(arg2, Nil))
    env: Rc<RefCell<Environment>>,
    original_call_span: Span, // Span of the entire procedure call expression, e.g., ((+ 1) 2)
) -> EvalResult {
    // 1. Evaluate the operator expression to get a procedure value
    let evaluated_operator_node = evaluate(operator_node.clone(), env.clone())?; // env.clone() is cheap (Rc)

    // 2. Check if the result of evaluating the operator is actually a procedure
    //    We need to borrow the Sexpr from the Rc<RefCell<Sexpr>>
    let procedure_value = match &*evaluated_operator_node.kind.borrow() {
        Sexpr::Procedure(proc_enum_val) => proc_enum_val.clone(), // Clone the Procedure enum (Primitive or Lambda)
        _ => {
            return Err(EvalError::NotAProcedure(
                evaluated_operator_node.kind.borrow().clone(), // The Sexpr that was found
                operator_node.span, // Span of the original operator expression from AST
            ));
        }
    };

    // 3. Apply the procedure
    match procedure_value {
        Procedure::Primitive(func, _) => {
            // Call the Rust function with evaluated args and the original call's span
            func(
                operands_list_node.clone().into_eval_iter(env),
                original_call_span,
            )
        }
        Procedure::Lambda(lambda) => apply_lambda(
            lambda,
            operands_list_node.clone().into_eval_iter(env.clone()),
            original_call_span,
        ),
    }
}

fn evaluate_let(operands: &Node, env: Rc<RefCell<Environment>>, original_span: Span) -> EvalResult {
    match &*operands.kind.borrow() {
        Sexpr::Pair(definitions_node, body_node) => {
            if !definitions_node.is_list() {
                return Err(EvalError::InvalidSpecialForm(
                    "let: expected (let (...(y <val>)) ...<expr>)".to_string(),
                    definitions_node.span.clone(),
                ));
            }
            let mut variables: HashMap<String, Node> = HashMap::new();
            for definition in definitions_node.clone().into_iter() {
                if let Some((name_node, value_node)) = definition.tuple() {
                    if let Some(name) = name_node.symbol() {
                        if variables.contains_key(&name) {
                            return Err(EvalError::InvalidSpecialForm(
                                "let: duplicate identifier".to_string(),
                                definitions_node.span.clone(),
                            ));
                        }
                        variables.insert(name, evaluate(value_node.clone(), env.clone())?);
                    } else {
                        return Err(EvalError::NotASymbol(
                            name_node.kind.borrow().clone(),
                            definition.span.clone(),
                        ));
                    }
                } else {
                    return Err(EvalError::InvalidSpecialForm(
                        "let: binding is not a pair (let (...(y <val>)) ...<expr>)".to_string(),
                        definition.span.clone(),
                    ));
                }
            }
            let env = Environment::new_enclosed(env.clone());
            {
                let mut env = env.borrow_mut();
                for (key, value) in variables.into_iter() {
                    env.define(key, value);
                }
            }
            evaluate_body(body_node.clone().into_iter(), env)
        }
        _ => Err(EvalError::InvalidSpecialForm(
            // e.g. (lambda) or (lambda not-a-list)
            "let: expected (let (...(y <val>)) ...<expr>)".to_string(),
            original_span,
        )),
    }
}

fn evaluate_letrec(
    operands: &Node,
    env: Rc<RefCell<Environment>>,
    original_span: Span,
) -> EvalResult {
    match &*operands.kind.borrow() {
        Sexpr::Pair(definitions_node, body_node) => {
            if !definitions_node.is_list() {
                return Err(EvalError::InvalidSpecialForm(
                    "let: expected (let (...(y <val>)) ...<expr>)".to_string(),
                    definitions_node.span.clone(),
                ));
            }
            let env = Environment::new_enclosed(env.clone());
            let mut names: HashSet<String> = HashSet::new();
            for definition in definitions_node.clone().into_iter() {
                if let Some((name_node, value_node)) = definition.tuple() {
                    if let Some(name) = name_node.symbol() {
                        if names.contains(&name) {
                            return Err(EvalError::InvalidSpecialForm(
                                "let: duplicate identifier".to_string(),
                                definitions_node.span.clone(),
                            ));
                        }
                        names.insert(name.clone());
                        let value = evaluate(value_node.clone(), env.clone())?;
                        env.borrow_mut().define(name, value);
                    } else {
                        return Err(EvalError::NotASymbol(
                            name_node.kind.borrow().clone(),
                            definition.span.clone(),
                        ));
                    }
                } else {
                    return Err(EvalError::InvalidSpecialForm(
                        "let: binding is not a pair (let (...(y <val>)) ...<expr>)".to_string(),
                        definition.span.clone(),
                    ));
                }
            }
            evaluate_body(body_node.clone().into_iter(), env)
        }
        _ => Err(EvalError::InvalidSpecialForm(
            // e.g. (lambda) or (lambda not-a-list)
            "let: expected (let (...(y <val>)) ...<expr>)".to_string(),
            original_span,
        )),
    }
}

fn evaluate_letstar(
    operands: &Node,
    env: Rc<RefCell<Environment>>,
    original_span: Span,
) -> EvalResult {
    match &*operands.kind.borrow() {
        Sexpr::Pair(definitions_node, body_node) => {
            if !definitions_node.is_list() {
                return Err(EvalError::InvalidSpecialForm(
                    "let: expected (let (...(y <val>)) ...<expr>)".to_string(),
                    definitions_node.span.clone(),
                ));
            }
            let env = Environment::new_enclosed(env.clone());
            for definition in definitions_node.clone().into_iter() {
                if let Some((name_node, value_node)) = definition.tuple() {
                    if let Some(name) = name_node.symbol() {
                        let value = evaluate(value_node.clone(), env.clone())?;
                        env.borrow_mut().define(name, value);
                    } else {
                        return Err(EvalError::NotASymbol(
                            name_node.kind.borrow().clone(),
                            definition.span.clone(),
                        ));
                    }
                } else {
                    return Err(EvalError::InvalidSpecialForm(
                        "let: binding is not a pair (let (...(y <val>)) ...<expr>)".to_string(),
                        definition.span.clone(),
                    ));
                }
            }
            evaluate_body(body_node.clone().into_iter(), env)
        }
        _ => Err(EvalError::InvalidSpecialForm(
            // e.g. (lambda) or (lambda not-a-list)
            "let: expected (let (...(y <val>)) ...<expr>)".to_string(),
            original_span,
        )),
    }
}

fn apply_lambda(
    lambda: Rc<Lambda>,
    mut args: EvaluatedNodeIterator,
    original_call_span: Span,
) -> EvalResult {
    let env = Environment::new_enclosed(lambda.env.clone());
    for (i, param) in lambda.params.iter().enumerate() {
        let Some(node) = args.next() else {
            return Err(EvalError::InvalidArguments(
                format!(
                    "lambda expects {} {} arguments, got {}",
                    if lambda.variadic_param.is_some() {
                        "at least"
                    } else {
                        "exactly"
                    },
                    lambda.params.len(),
                    i
                ),
                original_call_span,
            ));
        };
        env.borrow_mut().define(param.to_string(), node?);
    }
    if let Some(param) = &lambda.variadic_param {
        env.borrow_mut()
            .define(param.to_string(), args.collect::<Result<Node, _>>()?);
    } else {
        if args.next() != None {
            return Err(EvalError::InvalidArguments(
                format!(
                    "lambda expects exactly {} arguments, got more",
                    lambda.params.len(),
                ),
                original_call_span,
            ));
        }
    }
    evaluate_body(lambda.body.clone().into_iter(), env)
}

fn evaluate_body<T: Iterator<Item = Node>>(exprs: T, env: Rc<RefCell<Environment>>) -> EvalResult {
    let mut result: EvalResult = Ok(Node::new_nil(Span::default()));
    for expr in exprs {
        if let Ok(node) = result {
            env.borrow_mut().define("_".to_string(), node);
        } else {
            return result;
        }
        result = evaluate(expr.clone(), env.clone());
    }
    result
}

// Add more primitives: cons, car, cdr, list, null?, pair?, etc.
// --- Unit Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    use crate::ParseError;
    use crate::parser::parse_str; // Use parser to create AST nodes easily
    use crate::source::Span;

    fn new_test_env() -> Rc<RefCell<Environment>> {
        Environment::new_global_populated() // Or however you create your top-level env
    }

    // Helper to evaluate input string and check result kind (ignores span)
    fn assert_eval_kind(input: &str, expected_kind: &Sexpr, env: Option<Rc<RefCell<Environment>>>) {
        let env = env.unwrap_or_else(Environment::new_global_populated);
        match parse_str(input) {
            Ok(node) => match evaluate(node, env) {
                Ok(result_node) => {
                    assert_eq!(
                        *result_node.kind.borrow(),
                        *expected_kind,
                        "Input: '{}'",
                        input
                    )
                }
                Err(e) => panic!("Evaluation failed for input '{}': {}", input, e),
            },
            Err(e) => panic!("Parsing failed for input '{}': {}", input, e),
        }
    }

    fn assert_eval_number(input: &str, number: f64, env: Option<Rc<RefCell<Environment>>>) {
        assert_eval_kind(input, &Sexpr::Number(number), env);
    }

    fn assert_eval_sexpr(input: &str, expected_sexpr: &str, env: Option<Rc<RefCell<Environment>>>) {
        let env = env.unwrap_or_else(Environment::new_global_populated);
        match parse_str(input) {
            Ok(node) => match evaluate(node, env) {
                Ok(result_node) => {
                    assert_eq!(
                        result_node.to_string(),
                        *expected_sexpr,
                        "Input: '{}'",
                        input
                    )
                }
                Err(e) => panic!("Evaluation failed for input '{}': {}", input, e),
            },
            Err(e) => panic!("Parsing failed for input '{}': {}", input, e),
        }
    }

    fn assert_eval_node(input: &str, expected_node: Node, env: Option<Rc<RefCell<Environment>>>) {
        let env = env.unwrap_or_else(Environment::new_global_populated); // Use provided env or create new global one
        match parse_str(input) {
            Ok(node) => match evaluate(node, env) {
                Ok(result_node) => {
                    assert_eq!(result_node, expected_node, "Input: '{}'", input)
                }
                Err(e) => panic!("Evaluation failed for input '{}': {}", input, e),
            },
            Err(e) => panic!("Parsing failed for input '{}': {}", input, e),
        }
    }

    // Helper to assert evaluation errors
    fn assert_eval_error(
        input: &str,
        expected_error_variant: &EvalError,
        env: Option<Rc<RefCell<Environment>>>,
    ) {
        let env = env.unwrap_or_else(Environment::new_global_populated); // Use provided env or create new global one
        match parse_str(input) {
            Ok(node) => match evaluate(node, env) {
                Ok(result) => panic!(
                    "Expected evaluation to fail for input '{}', but got: {:?}",
                    input, result
                ),
                Err(e) => {
                    assert_eq!(
                        std::mem::discriminant(&e),
                        std::mem::discriminant(expected_error_variant),
                        "Input: '{}', Expected error variant like {:?}, got: {:?}",
                        input,
                        expected_error_variant,
                        e
                    );
                }
            },
            Err(e) => panic!("Parsing failed for input '{}': {}", input, e),
        }
    }

    fn eval_str(input: &str, env: Rc<RefCell<Environment>>) -> EvalResult {
        match parse_str(input) {
            Ok(node) => evaluate(node, env),
            Err(e) => panic!("Parsing failed for input '{}': {}", input, e),
        }
    }

    // Helper to check if a symbol in the environment has a specific f64 number value
    fn assert_env_has(env: &Rc<RefCell<Environment>>, name: &str, expected_val: &Sexpr) {
        let val_node = env
            .borrow()
            .get(name, Span::default())
            .unwrap_or_else(|_| panic!("'{}' not found in environment", name));
        let other = val_node.kind.borrow();
        if *expected_val == *other {
            return;
        } else {
            panic!(
                "Expected '{}' to be {:?}, got {:?}",
                name, expected_val, other
            );
        }
    }

    fn assert_eval_define(
        input: &str,
        env: &Rc<RefCell<Environment>>,
        name: &str,
        expected_val: &Sexpr,
    ) {
        assert_eval_kind(input, expected_val, Some(env.clone()));
        assert_env_has(env, name, expected_val);
    }

    fn assert_env_var_is_number(
        env: &Rc<RefCell<Environment>>,
        var_name: &str,
        expected_val: f64,
        test_name_prefix: &str,
    ) {
        let val_node = env
            .borrow()
            .get(var_name, Span::default())
            .unwrap_or_else(|err| {
                panic!(
                    "{}: Variable '{}' not found in environment: {}",
                    test_name_prefix, var_name, err
                )
            });
        match &*val_node.kind.borrow() {
            Sexpr::Number(n) => assert_eq!(
                *n, expected_val,
                "{}: Env var '{}' expected {}, got {}",
                test_name_prefix, var_name, expected_val, n
            ),
            other => panic!(
                "{}: Env var '{}' expected Number, got {:?}",
                test_name_prefix, var_name, other
            ),
        }
    }

    fn assert_env_var_is_not_defined(
        env: &Rc<RefCell<Environment>>,
        var_name: &str,
        test_name_prefix: &str,
    ) {
        assert!(
            env.borrow().get(var_name, Span::default()).is_err(),
            "{}: Variable '{}' should NOT be defined in environment",
            test_name_prefix,
            var_name
        );
    }

    #[test]
    fn test_eval_self_evaluating() {
        assert_eval_kind("123", &Sexpr::Number(123.0), None);
        assert_eval_kind("-4.5", &Sexpr::Number(-4.5), None);
        assert_eval_kind("#t", &Sexpr::Boolean(true), None);
        assert_eval_kind("#f", &Sexpr::Boolean(false), None);
        assert_eval_kind(r#""hello""#, &Sexpr::String("hello".to_string()), None);
        assert_eval_kind("()", &Sexpr::Nil, None); // Evaluating '()' -> Nil
    }

    #[test]
    fn test_eval_symbol_lookup_ok() {
        let env = Environment::new();
        env.borrow_mut().define(
            "x".to_string(),
            Node::new(Sexpr::Number(100.0), Span::default()),
        );
        assert_eval_kind("x", &Sexpr::Number(100.0), Some(env));
    }

    #[test]
    fn test_eval_symbol_lookup_unbound() {
        let env = Environment::new(); // Empty env
        // Create a dummy error variant just for discriminant comparison
        let unbound_error =
            EvalError::EnvError(EnvError::UnboundVariable("".into(), Span::default()));
        assert_eval_error("y", &unbound_error, Some(env));
    }

    #[test]
    fn test_eval_quote() {
        assert_eval_kind("'1", &Sexpr::Number(1.0), None); // '(quote 1) -> 1
        assert_eval_kind("'a", &Sexpr::Symbol("a".to_string()), None); // '(quote a) -> a
        assert_eval_kind("'#t", &Sexpr::Boolean(true), None); // '(quote #t) -> #t
        assert_eval_kind("'()", &Sexpr::Nil, None); // '(quote ()) -> ()
        assert_eval_kind("(quote ())", &Sexpr::Nil, None); // '(quote ()) -> ()

        // '(quote (1 2)) -> (1 2)
        // Need a way to represent the expected list structure (Nodes inside Sexpr::List)
        // Let's check the structure manually for now:
        let env = Environment::new();
        match parse_str("'(1 2)") {
            Ok(node) => match evaluate(node, env) {
                Ok(result_node) => {
                    if let Sexpr::Pair(first, rest) = &*result_node.kind.borrow() {
                        assert!(matches!(*first.kind.borrow(), Sexpr::Number(n) if n == 1.0));
                        if let Sexpr::Pair(expect_1, expect_nil) = &*rest.kind.borrow() {
                            assert!(
                                matches!(*expect_1.kind.borrow(), Sexpr::Number(n) if n == 2.0)
                            );
                            assert!(matches!(*expect_nil.kind.borrow(), Sexpr::Nil));
                        } else {
                        }
                    } else {
                        panic!("Expected list");
                    }
                }
                Err(e) => panic!("Eval failed: {}", e),
            },
            Err(e) => panic!("Parse failed: {}", e),
        }

        // Test quote error: wrong number of args
        let wrong_args_error = EvalError::InvalidSpecialForm("".into(), Span::default()); // Dummy
        assert_eval_error("(quote a b)", &wrong_args_error, None);
        assert_eval_error("(quote)", &wrong_args_error, None);
    }

    #[test]
    fn test_eval_if_true() {
        assert_eval_kind("(if #t 1 2)", &Sexpr::Number(1.0), None);
        assert_eval_kind("(if #t 1)", &Sexpr::Number(1.0), None); // No alternate
        assert_eval_kind("(if 0 1 2)", &Sexpr::Number(1.0), None); // 0 is true
        assert_eval_kind("(if '() 1 2)", &Sexpr::Number(1.0), None); // '() is true
        assert_eval_kind("(if \"hello\" 1 2)", &Sexpr::Number(1.0), None); // String is true
        assert_eval_kind("(if (quote x) 1 2)", &Sexpr::Number(1.0), None); // Symbol is true
    }

    #[test]
    fn test_eval_if_false() {
        assert_eval_kind("(if #f 1 2)", &Sexpr::Number(2.0), None);
        assert_eval_kind("(if #f 1)", &Sexpr::Nil, None); // No alternate, returns Nil (our choice)
    }

    #[test]
    fn test_eval_if_nested() {
        let env = Environment::new();
        env.borrow_mut().define(
            "x".to_string(),
            Node::new(Sexpr::Boolean(true), Span::default()),
        );
        env.borrow_mut().define(
            "y".to_string(),
            Node::new(Sexpr::Boolean(false), Span::default()),
        );

        assert_eval_kind(
            "(if x 1 (if y 2 3))",
            &Sexpr::Number(1.0),
            Some(env.clone()),
        );
        assert_eval_kind(
            "(if y 1 (if x 2 3))",
            &Sexpr::Number(2.0),
            Some(env.clone()),
        );
    }

    #[test]
    fn test_eval_if_evaluates_condition() {
        // Ensure the condition is actually evaluated
        let env = Environment::new();
        env.borrow_mut().define(
            "cond".to_string(),
            Node::new(Sexpr::Boolean(false), Span::default()),
        );
        assert_eval_kind("(if cond 1 2)", &Sexpr::Number(2.0), Some(env));
    }

    #[test]
    fn test_eval_if_does_not_evaluate_unused_branch() {
        // This is harder to test directly without side effects (like define/set! or print)
        // We can test it indirectly by putting an unbound variable in the unused branch.
        let env = Environment::new();
        // (if #t 'good unbound-variable) should evaluate to 'good without error
        assert_eval_kind(
            "(if #t 'good unbound-variable)",
            &Sexpr::Symbol("good".to_string()),
            Some(env.clone()),
        );
        // (if #f unbound-variable 'good) should evaluate to 'good without error
        assert_eval_kind(
            "(if #f unbound-variable 'good)",
            &Sexpr::Symbol("good".to_string()),
            Some(env),
        );
    }

    #[test]
    fn test_eval_if_error_arity() {
        let arity_error = &EvalError::InvalidSpecialForm("".into(), Span::default());
        assert_eval_error("(if)", arity_error, None);
        assert_eval_error("(if #t)", arity_error, None);
        assert_eval_error("(if #t 1 2 3)", arity_error, None);
    }

    #[test]
    fn test_eval_if_error_in_condition() {
        let unbound_error =
            &EvalError::EnvError(EnvError::UnboundVariable("".into(), Span::default())); // Dummy
        assert_eval_error("(if unbound 1 2)", unbound_error, None);
    }

    #[test]
    fn test_eval_list_placeholder_error() {
        // Evaluating a list where the first element is not a known special form
        // or procedure (which we haven't implemented yet) should error.
        let env = Environment::new();
        // (1 2 3) -> Error because 1 is not a procedure/special form
        let not_proc_error = EvalError::NotAProcedure(Sexpr::Number(0.0), Span::default()); // Dummy
        assert_eval_error("(1 2 3)", &not_proc_error, Some(env.clone()));

        // ("hello" 1) -> Error
        let not_proc_error_str =
            EvalError::NotAProcedure(Sexpr::String("".into()), Span::default());
        assert_eval_error("(\"hello\" 1)", &not_proc_error_str, Some(env.clone()));
    }

    #[test]
    fn test_eval_primitives_arithmetic() {
        assert_eval_kind("(+ 1 2)", &Sexpr::Number(3.0), None);
        assert_eval_kind("(+ 10 20 30 40)", &Sexpr::Number(100.0), None);
        assert_eval_kind("(+)", &Sexpr::Number(0.0), None); // Add identity
        assert_eval_kind("(- 10 3)", &Sexpr::Number(7.0), None);
        assert_eval_kind("(- 5)", &Sexpr::Number(-5.0), None);
        assert_eval_kind("(- 10 3 2)", &Sexpr::Number(5.0), None);
        assert_eval_kind("(* 2 3)", &Sexpr::Number(6.0), None);
        assert_eval_kind("(* 2 3 4)", &Sexpr::Number(24.0), None);
        assert_eval_kind("(*)", &Sexpr::Number(1.0), None); // Multiply identity
        assert_eval_kind("(/ 10 2)", &Sexpr::Number(5.0), None);
        assert_eval_kind("(/ 10 4)", &Sexpr::Number(2.5), None);
        assert_eval_kind("(/ 20 2 5)", &Sexpr::Number(2.0), None);
        assert_eval_kind("(/ 5)", &Sexpr::Number(0.2), None); // 1/5
    }

    #[test]
    fn test_eval_primitives_comparison() {
        assert_eval_kind("(= 5 5)", &Sexpr::Boolean(true), None);
        assert_eval_kind("(= 5 5 5 5)", &Sexpr::Boolean(true), None);
        assert_eval_kind("(= 5 6)", &Sexpr::Boolean(false), None);
        assert_eval_kind("(= 5 5 6)", &Sexpr::Boolean(false), None);
        assert_eval_kind("(< 4 5 6)", &Sexpr::Boolean(true), None);
        assert_eval_kind("(< 5 5 6)", &Sexpr::Boolean(false), None);
        assert_eval_kind("(<= 5 5 6)", &Sexpr::Boolean(true), None);
        assert_eval_kind("(>= 5 5 5)", &Sexpr::Boolean(true), None);
        assert_eval_kind("(>= 6 5 5)", &Sexpr::Boolean(true), None);
        assert_eval_kind("(> 6 5 5)", &Sexpr::Boolean(false), None);
        assert_eval_kind("(> 6 5 4)", &Sexpr::Boolean(true), None);
        assert_eval_kind("(< 1 2 3 4 5 6)", &Sexpr::Boolean(true), None);
        assert_eval_kind("(>= 5 5 4 4 4 3)", &Sexpr::Boolean(true), None);
    }

    #[test]
    fn test_eval_primitives_nested_calls() {
        assert_eval_kind("(+ 1 (* 2 3))", &Sexpr::Number(7.0), None);
        assert_eval_kind("(- (+ 5 5) (* 2 3))", &Sexpr::Number(4.0), None);
    }

    #[test]
    fn test_eval_primitives_arity_errors() {
        let arity_error = &EvalError::InvalidArguments("".into(), Span::default()); // Dummy
        // assert_eval_error("(-)", arity_error); // Handled by check_arity!
        assert_eval_error("(/)", arity_error, None);
        assert_eval_error("(=)", arity_error, None);
    }

    #[test]
    fn test_eval_primitives_type_errors() {
        let type_error = &EvalError::InvalidArguments("".into(), Span::default()); // Dummy
        assert_eval_error("(+ 1 #t)", type_error, None);
        assert_eval_error("(/ 1 \"hello\")", type_error, None);
        assert_eval_error("(= 1 #f)", type_error, None); // Mismatched types in comparison
    }

    fn node_nil(start: usize, end: usize) -> Node {
        Node::new_nil(Span::new(start, end))
    }

    fn node_pair(car: Node, cdr: Node, start: usize, end: usize) -> Node {
        Node::new_pair(car, cdr, Span::new(start, end))
    }

    fn node_number(n: f64, start: usize, end: usize) -> Node {
        Node::new_number(n, Span::new(start, end))
    }

    fn _node_symbol(symbol: &str, start: usize, end: usize) -> Node {
        Node::new_symbol(symbol.to_string(), Span::new(start, end))
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

    #[test]
    fn test_eval_not_procedure_error() {
        let not_proc_error = &EvalError::NotAProcedure(Sexpr::Number(0.0), Span::default()); // Dummy
        assert_eval_error("(1 2 3)", not_proc_error, None); // Updated: was placeholder, now should work
        let not_proc_error_list = &EvalError::NotAProcedure(Sexpr::Nil, Span::default()); // Dummy
        assert_eval_error("((list 1 2) 3)", not_proc_error_list, None); // Need list primitive first for this
    }

    #[test]
    fn test_eval_list_primitives() {
        // list
        assert_eval_kind("(list)", &Sexpr::Nil, None);
        assert_eval_node(
            "(list 1 2 3)",
            node_list(
                &[
                    node_number(1.0, 6, 7),
                    node_number(2.0, 8, 9),
                    node_number(3.0, 10, 11),
                ],
                6,
                12,
            ),
            None,
        );
        assert_eval_node(
            "(list (+ 1 1) (- 5 2))",
            node_list(&[node_number(2.0, 6, 13), node_number(3.0, 14, 21)], 6, 22),
            None,
        );

        // cons
        assert_eval_node(
            "(cons 1 '())",
            node_pair(node_number(1.0, 6, 7), node_nil(9, 11), 0, 12),
            None,
        );
        assert_eval_node(
            "(cons 1 (list 2 3))",
            node_pair(
                node_number(1.0, 6, 7),
                node_list(
                    &[node_number(2.0, 14, 15), node_number(3.0, 16, 17)],
                    14,
                    18,
                ),
                0,
                19,
            ),
            None,
        );
        assert_eval_node(
            "(cons (list 1) (list 2))",
            node_pair(
                node_list(&[node_number(1.0, 12, 13)], 12, 14),
                node_list(&[node_number(2.0, 21, 22)], 21, 23),
                0,
                24,
            ),
            None,
        );
        assert_eval_node(
            "(cons 1 2)",
            node_pair(node_number(1.0, 6, 7), node_number(2.0, 8, 9), 0, 10),
            None,
        );

        // car
        assert_eval_kind("(car (list 1 2 3))", &Sexpr::Number(1.0), None);
        assert_eval_kind("(car (cons 'a '()))", &Sexpr::Symbol("a".to_string()), None);

        // cdr
        assert_eval_node(
            "(cdr (list 1 2 3))",
            node_list(
                &[node_number(2.0, 13, 14), node_number(3.0, 15, 16)],
                13,
                17,
            ),
            None,
        );
        assert_eval_kind("(cdr (list 1))", &Sexpr::Nil, None); // cdr of single-element list is Nil
        assert_eval_node(
            "(cdr (cons 1 (cons 2 '())))",
            node_pair(node_number(2.0, 19, 20), node_nil(22, 24), 13, 25),
            None,
        );
    }

    #[test]
    fn test_eval_list_primitive_errors() {
        let type_error = &EvalError::InvalidArguments("".into(), Span::default()); // Dummy
        let arity_error = &EvalError::InvalidArguments("".into(), Span::default()); // Dummy

        // cons arity/type
        assert_eval_error("(cons 1)", arity_error, None);
        assert_eval_error("(cons 1 2 3)", arity_error, None);

        // car arity/type
        assert_eval_error("(car)", arity_error, None);
        assert_eval_error("(car 1 2)", arity_error, None);
        assert_eval_error("(car '())", type_error, None); // Error on car of empty list
        assert_eval_error("(car 5)", type_error, None); // Error on car of non-list

        // cdr arity/type
        assert_eval_error("(cdr)", arity_error, None);
        assert_eval_error("(cdr 1 2)", arity_error, None);
        assert_eval_error("(cdr '())", type_error, None); // Error on cdr of empty list
        assert_eval_error("(cdr 5)", type_error, None); // Error on cdr of non-list
    }

    #[test]
    fn test_eval_type_predicates() {
        assert_eval_kind("(null? '())", &Sexpr::Boolean(true), None);
        assert_eval_kind("(null? (list))", &Sexpr::Boolean(true), None);
        assert_eval_kind("(null? (list 1))", &Sexpr::Boolean(false), None);
        assert_eval_kind("(null? 1)", &Sexpr::Boolean(false), None);

        assert_eval_kind("(pair? (cons 1 (list 2)))", &Sexpr::Boolean(true), None); // Assuming cons returns list for now
        assert_eval_kind("(pair? (list 1))", &Sexpr::Boolean(true), None);
        //assert_eval_kind("(pair? '(1 . 2))", Sexpr::Boolean(true), None); // Need dotted pair support first
        assert_eval_kind("(pair? '())", &Sexpr::Boolean(false), None);
        assert_eval_kind("(pair? 1)", &Sexpr::Boolean(false), None);

        assert_eval_kind("(number? 1)", &Sexpr::Boolean(true), None);
        assert_eval_kind("(number? #f)", &Sexpr::Boolean(false), None);
        assert_eval_kind("(boolean? #t)", &Sexpr::Boolean(true), None);
        assert_eval_kind("(boolean? 0)", &Sexpr::Boolean(false), None);
        assert_eval_kind("(symbol? 'a)", &Sexpr::Boolean(true), None);
        assert_eval_kind("(symbol? \"a\")", &Sexpr::Boolean(false), None);
        assert_eval_kind("(string? \"a\")", &Sexpr::Boolean(true), None);
        assert_eval_kind("(string? 'a)", &Sexpr::Boolean(false), None);
        assert_eval_kind("(procedure? +)", &Sexpr::Boolean(true), None);
        assert_eval_kind("(procedure? 1)", &Sexpr::Boolean(false), None);
        // Add tests for lambda later: (procedure? (lambda (x) x)) -> true
    }

    #[test]
    fn test_define_number() {
        let env = new_test_env();
        assert_eval_define("(define x 10)", &env, "x", &Sexpr::Number(10.0));
    }

    #[test]
    fn test_define_boolean() {
        let env = new_test_env();
        assert_eval_define("(define x #t)", &env, "x", &Sexpr::Boolean(true));
        assert_eval_define("(define y #f)", &env, "y", &Sexpr::Boolean(false));
    }

    #[test]
    fn test_define_quoted_symbol() {
        let env = new_test_env();
        assert_eval_define(
            "(define s 'some-symbol)",
            &env,
            "s",
            &Sexpr::Symbol("some-symbol".to_string()),
        );
    }

    #[test]
    fn test_define_expression() {
        let env = new_test_env();
        assert_eval_define(
            "(define result (+ 5 (/ 15 3)))",
            &env,
            "result",
            &Sexpr::Number(10.0),
        );
    }

    #[test]
    fn test_define_uses_previously_defined_variable() {
        let env = new_test_env();
        assert_eval_define("(define a 5)", &env, "a", &Sexpr::Number(5.0));
        assert_eval_define("(define b (+ a 3))", &env, "b", &Sexpr::Number(8.0));
    }

    #[test]
    fn test_redefine_variable() {
        let env = new_test_env();
        assert_eval_define("(define val 100)", &env, "val", &Sexpr::Number(100.0));
        assert_eval_define("(define val 200)", &env, "val", &Sexpr::Number(200.0));
    }

    #[test]
    fn test_define_nil_by_default() {
        let env = new_test_env();
        assert_eval_define("(define x)", &env, "x", &Sexpr::Nil);
    }

    // --- Error Cases ---

    #[test]
    fn test_define_arity_error_too_few_args_zero() {
        assert_eval_error(
            "(define)",
            &EvalError::InvalidSpecialForm(
                "define: No arguments provided, expected (define name [value])".to_string(),
                Span::default(),
            ),
            None,
        );
    }

    #[test]
    fn test_define_arity_error_too_many_args() {
        assert_eval_error(
            "(define x 1 2)",
            &EvalError::InvalidSpecialForm(
                "define: Too many arguments [3], expected (define name [value])".to_string(),
                Span::default(),
            ),
            None,
        );
    }

    #[test]
    fn test_define_variable_not_a_symbol_number() {
        assert_eval_error(
            "(define 123 456)",
            &EvalError::NotASymbol(Sexpr::Number(123.0), Span::default()),
            None,
        );
    }

    #[test]
    fn test_define_expression_eval_error() {
        let env = new_test_env();
        let result = eval_str("(define x unbound-var)", env);
        assert_eq!(
            result,
            Err(EvalError::EnvError(EnvError::UnboundVariable(
                "unbound-var".to_string(),
                Span::new(10, 21)
            )))
        );
    }

    fn assert_is_lambda(node: Node, message: &str) {
        assert!(
            matches!(*node.kind.borrow(), Sexpr::Procedure(Procedure::Lambda(_))),
            "{}",
            message
        );
    }

    #[test]
    fn test_lambda_creates_procedure() {
        let env = new_test_env();
        for (str, message) in [
            ("(lambda (x) x)", "Lambda did not create a procedure"),
            ("(lambda () 42)", "Lambda with no params"),
            ("(lambda (a b c) (+ a b c))", "Lambda with multiple params"),
            (
                "(lambda (a b c . d) (+ a b c (apply + d)))",
                "Lambda with multiple params and variadic param",
            ),
            ("(lambda x (apply + x))", "Lambda with only variadic param"),
            (
                "(lambda x (apply + x) (apply * x) (apply / x))",
                "Lambda with with multiple bodies",
            ),
            (
                "(lambda () undefined-var1 undefined-var2)",
                "Lambda does not evaluate body elements",
            ),
        ] {
            let result_node = eval_str(str, env.clone()).unwrap();
            assert_is_lambda(result_node, message);
        }
    }

    #[test]
    fn test_lambda_captures_env() {
        let env = new_test_env();
        eval_str("(define y 10)", env.clone()).unwrap();
        let lambda_node = eval_str("(lambda (x) (+ x y))", env.clone()).unwrap(); // y is captured from outer env

        match &*lambda_node.kind.borrow() {
            Sexpr::Procedure(Procedure::Lambda(lambda_rc)) => {
                // Check if the captured environment has 'y'
                // This requires `lambda_rc.env` to be accessible and for Environment to have a `get`
                // or some way to inspect its bindings for testing.
                // If Environment::get returns Option<Node>:
                let result = lambda_rc.env.borrow().get("y", Span::default());

                assert!(result.is_ok(), "Lambda did not capture 'y'");
                if let Ok(y_val_node) = result {
                    assert!(
                        matches!(*y_val_node.kind.borrow(), Sexpr::Number(10.0)),
                        "Captured 'y' has wrong value"
                    );
                }
            }
            _ => panic!("Expected lambda procedure"),
        }
    }

    // --- Syntax Error Tests for `lambda` form ---
    #[test]
    fn test_lambda_syntax_error_no_params_or_body() {
        let env = new_test_env();
        let result = eval_str("(lambda)", env);
        assert!(matches!(result, Err(EvalError::InvalidSpecialForm(_, _))));
    }

    #[test]
    fn test_lambda_syntax_error_no_body() {
        let env = new_test_env();
        let result = eval_str("(lambda (x))", env);
        assert!(matches!(result, Err(EvalError::InvalidSpecialForm(_, _))));
    }

    #[test]
    fn test_lambda_syntax_error_param_not_symbol_list_number() {
        let env = new_test_env();
        let result = eval_str("(lambda 123 x)", env); // params is '123'
        assert!(matches!(result, Err(EvalError::InvalidArguments(_, _))));
    }

    #[test]
    fn test_lambda_syntax_error_param_in_list_not_symbol() {
        let env = new_test_env();
        let result = eval_str("(lambda (x 123 y) x)", env); // '123' in param list
        assert!(matches!(result, Err(EvalError::NotASymbol(_, _))));
    }

    // --- Tests for Lambda Application (Calling the procedure) ---

    #[test]
    fn test_lambda_application_simple() {
        assert_eval_kind("((lambda (x) (+ x 5)) 10)", &Sexpr::Number(15.0), None);
    }

    #[test]
    fn test_lambda_application_no_params() {
        assert_eval_kind("((lambda () 42))", &Sexpr::Number(42.0), None);
    }

    #[test]
    fn test_lambda_application_multiple_params() {
        assert_eval_kind("((lambda (a b) (* a b)) 3 7)", &Sexpr::Number(21.0), None);
    }

    #[test]
    fn test_lambda_application_closure_simple_capture() {
        let env = new_test_env();
        eval_str("(define y 100)", env.clone()).unwrap();
        assert_eval_kind("((lambda (x) (+ x y)) 5)", &Sexpr::Number(105.0), Some(env));
    }

    #[test]
    fn test_lambda_application_make_adder_closure() {
        let env = new_test_env();
        eval_str(
            "(define make-adder (lambda (n) (lambda (x) (+ x n))))",
            env.clone(),
        )
        .unwrap();
        eval_str("(define add5 (make-adder 5))", env.clone()).unwrap();
        assert_eval_kind("(add5 10)", &Sexpr::Number(15.0), Some(env.clone()));

        eval_str("(define add10 (make-adder 10))", env.clone()).unwrap();
        assert_eval_kind("(add10 10)", &Sexpr::Number(20.0), Some(env.clone()));

        assert_eval_kind("(add5 1)", &Sexpr::Number(6.0), Some(env.clone()));
        assert_eval_kind(
            "(add10 (add5 (add10 ((make-adder 2) 2))))",
            &Sexpr::Number(29.0),
            Some(env.clone()),
        );
    }

    #[test]
    fn test_lambda_shadowing_outer_variable() {
        let env = new_test_env();
        eval_str("(define x 10)", env.clone()).unwrap();
        // This lambda's 'x' parameter shadows the outer 'x'
        assert_eval_kind(
            "((lambda (x) (+ x 1)) 5)",
            &Sexpr::Number(6.0),
            Some(env.clone()),
        );
        // Ensure outer 'x' is unchanged
        assert_eval_kind("x", &Sexpr::Number(10.0), Some(env.clone()));
    }

    #[test]
    fn test_lambda_application_sequential_body_eval() {
        let env = new_test_env();
        // The result should be from the last expression in the body
        assert_eval_kind(
            "((lambda () (define temp 5) (+ temp 10)))",
            &Sexpr::Number(15.0),
            Some(env.clone()),
        );
    }

    #[test]
    fn test_lambda_application_sequential_underscore_eval() {
        let env = new_test_env();
        // The result should be from the last expression in the body
        assert_eval_kind(
            "((lambda (x y z) (+ x y z) (/ _ 3) (* _ 2)) 1 2 3)",
            &Sexpr::Number(4.0),
            Some(env.clone()),
        );
    }

    // --- Arity Error Tests for Lambda Application ---
    #[test]
    fn test_lambda_application_arity_error_too_few_args() {
        let env = new_test_env();
        let result = eval_str("((lambda (x y) (+ x y)) 5)", env);
        assert!(matches!(result, Err(EvalError::InvalidArguments(_, _))));
    }

    #[test]
    fn test_lambda_application_arity_error_too_many_args() {
        let env = new_test_env();
        let result = eval_str("((lambda (x) x) 5 6)", env);
        assert!(matches!(result, Err(EvalError::InvalidArguments(_, _))));
    }

    #[test]
    fn test_lambda_application_arity_error_no_params_given_one() {
        let env = new_test_env();
        let result = eval_str("((lambda () 42) 1)", env);
        assert!(matches!(result, Err(EvalError::InvalidArguments(_, _))));
    }

    // --- Tests for more complex scenarios (recursion, higher-order) ---
    #[test]
    fn test_lambda_recursion_factorial() {
        let env = new_test_env();
        eval_str(
            "(define factorial (lambda (n) (if (= n 0) 1 (* n (factorial (- n 1))))))",
            env.clone(),
        )
        .unwrap();
        assert_eval_kind("(factorial 5)", &Sexpr::Number(120.0), Some(env.clone()));
        assert_eval_kind(
            "(factorial 10)",
            &Sexpr::Number(3628800.0),
            Some(env.clone()),
        );
    }

    #[test]
    fn test_lambda_higher_order_map_simple() {
        let env = new_test_env();
        // Define a simple list for map to operate on (assuming `cons` and `list` or `quote` work)
        eval_str("(define my-list '(1 2 3 4))", env.clone()).unwrap();
        eval_str("(define square (lambda (x) (* x x)))", env.clone()).unwrap();

        eval_str(
            "(define map
               (lambda (proc lst)
                 (if (null? lst) ; Assuming null? primitive
                     '()
                     (cons (proc (car lst)) (map proc (cdr lst))))))", // Assuming car/cdr primitives
            env.clone(),
        )
        .unwrap();
        eval_str(
            "(define foldl
               (lambda (proc acc lst)
                 (if (null? lst) ; Assuming null? primitive
                     acc
                     (foldl proc (proc acc (car lst)) (cdr lst)))))", // Assuming car/cdr primitives
            env.clone(),
        )
        .unwrap();

        eval_str("(define apply-twice (lambda (f x) (f (f x))))", env.clone()).unwrap();
        assert_eval_kind(
            "(apply-twice square 2)",
            &Sexpr::Number(16.0),
            Some(env.clone()),
        );

        assert_eval_kind(
            "(foldl + 0 my-list)",
            &Sexpr::Number(10.0),
            Some(env.clone()),
        );

        assert_eval_kind(
            "(foldl + 0 (map square my-list))",
            &Sexpr::Number(30.0),
            Some(env.clone()),
        );
        assert_eval_kind(
            "(foldl + 0 (map (lambda (x) (apply-twice square x)) my-list))",
            &Sexpr::Number(354.0),
            Some(env.clone()),
        );
    }

    #[test]
    fn test_lambda_variadic_no_args() {
        assert_eval_kind("((lambda x x))", &Sexpr::Nil, None);
    }

    #[test]
    fn test_lambda_variadic_one_arg() {
        assert_eval_sexpr("((lambda x x) 10)", "(10)", None);
    }

    #[test]
    fn test_lambda_variadic_multiple_args() {
        assert_eval_sexpr("((lambda x x) 1 2 3.0)", "(1 2 3)", None);
    }

    #[test]
    fn test_lambda_variadic_args_are_evaluated() {
        assert_eval_sexpr("((lambda x x) (+ 1 2) (- 5 1))", "(3 4)", None);
    }

    #[test]
    fn test_lambda_variadic_accessing_args() {
        let env = new_test_env();
        eval_str(
            "(define my-var-func (lambda params (car params)))",
            env.clone(),
        )
        .unwrap();
        assert_eval_sexpr("(my-var-func 100 200 300)", "100", Some(env));
    }

    #[test]
    fn test_lambda_variadic_define_and_call() {
        let env = new_test_env();
        eval_str("(define collect (lambda x x))", env.clone()).unwrap();
        assert_eval_sexpr("(collect 'a 'b 'c)", "(a b c)", Some(env));
    }

    // --- Tests for Dotted-List Parameters `(lambda (p1 p2 . rest) ...)` ---

    #[test]
    fn test_lambda_dotted_exact_fixed_args_no_rest() {
        assert_eval_sexpr(
            "((lambda (p1 p2 . r) (list p1 p2 r)) 10 20)",
            "(10 20 ())",
            None,
        );
    }

    #[test]
    fn test_lambda_dotted_one_fixed_one_rest_arg() {
        assert_eval_sexpr("((lambda (p1 . r) (list p1 r)) 10 20)", "(10 (20))", None);
    }

    #[test]
    fn test_lambda_dotted_one_fixed_multiple_rest_args() {
        assert_eval_sexpr(
            "((lambda (p1 . r) (list p1 r)) 10 20 30 40)",
            "(10 (20 30 40))",
            None,
        );
    }

    #[test]
    fn test_lambda_dotted_multiple_fixed_multiple_rest_args() {
        assert_eval_sexpr(
            "((lambda (p1 p2 p3 . r) (list p1 p2 p3 r)) 1 2 3 4 5 6)",
            "(1 2 3 (4 5 6))",
            None,
        );
    }

    #[test]
    fn test_lambda_dotted_args_are_evaluated() {
        assert_eval_sexpr(
            "((lambda (a . r) (list a r)) (+ 1 2) (- 10 1) (* 2 3))",
            "(3 (9 6))",
            None,
        );
    }

    #[test]
    fn test_lambda_dotted_define_and_call() {
        let env = new_test_env();
        eval_str(
            "(define my-dot-func (lambda (first . rest) (list first (null? rest))))",
            env.clone(),
        )
        .unwrap();
        assert_eval_sexpr("(my-dot-func 1)", "(1 #t)", Some(env.clone()));
        assert_eval_sexpr("(my-dot-func 1 2 3)", "(1 #f)", Some(env.clone()));
    }

    // --- Arity Error Tests for Dotted-List Parameters ---
    #[test]
    fn test_lambda_dotted_arity_error_too_few_for_fixed() {
        let env = new_test_env();
        // (lambda (p1 p2 . r) ...) requires at least 2 arguments
        let result = eval_str("((lambda (p1 p2 . r) r) 5)", env);
        assert!(matches!(result, Err(EvalError::InvalidArguments(_, _))));
    }

    // --- Syntax Error Tests for Variadic/Dotted Parameter Definitions ---
    #[test]
    fn test_lambda_syntax_error_dotted_misplaced_dot() {
        // (lambda (. x) x) is invalid Scheme, dot must follow a symbol
        let result1 = parse_str("(lambda (. x) x)");
        assert!(matches!(result1, Err(ParseError::UnexpectedToken { .. })));

        let result2 = parse_str("(lambda (a . b . c) a)");
        assert!(matches!(result2, Err(ParseError::UnexpectedToken { .. })));
    }

    #[test]
    fn test_lambda_syntax_error_dotted_symbol_after_dot_not_symbol() {
        let env = new_test_env();
        // (lambda (a . 10) x) is invalid, item after dot must be a symbol
        let result = eval_str("(lambda (a . 10) x)", env);
        assert!(
            matches!(result, Err(EvalError::NotASymbol(_, _))),
            "Syntax error: non-symbol after dot"
        );
    }

    #[test]
    fn test_lambda_syntax_error_variadic_params_not_single_symbol() {
        let env = new_test_env();
        // (lambda (a b) body) -> params is a list
        // (lambda rest body) -> params is a symbol
        // (lambda (a . b) body) -> params is a dotted list
        // (lambda '(a b) body) -> params is `(quote (a b))`, not a list of symbols directly. This should be an error.
        let result = eval_str("(lambda '(a b) x)", env);
        assert!(
            matches!(result, Err(EvalError::NotASymbol(_, _))),
            "Syntax error: variadic parameter list specified as a quoted list"
        );
    }

    #[test]
    fn test_define_func_no_args() {
        let env = new_test_env();
        eval_str("(define (my-func) 3)", env.clone()).unwrap();
        assert_eval_kind("(my-func)", &Sexpr::Number(3.0), Some(env));
    }

    #[test]
    fn test_define_func_one_arg() {
        let env = new_test_env();
        eval_str("(define (add3 x) (+ 3 x))", env.clone()).unwrap();
        assert_eval_kind("(add3 3)", &Sexpr::Number(6.0), Some(env));
    }

    #[test]
    fn test_define_func_multiple_args() {
        let env = new_test_env();
        eval_str("(define (add-3 a b c) (+ a b c))", env.clone()).unwrap();
        assert_eval_kind("(add-3 1 2 3)", &Sexpr::Number(6.0), Some(env));
    }

    #[test]
    fn test_define_func_one_variadic_arg() {
        let env = new_test_env();
        eval_str("(define (collect . x) x)", env.clone()).unwrap();
        assert_eval_sexpr("(collect 'a 'b 'c)", "(a b c)", Some(env));
    }

    #[test]
    fn test_define_func_multiple_body_expressions() {
        let env = new_test_env();
        // The define inside the function body creates a local binding for that call
        eval_str(
            "(define (complex-body x) (define y (+ x 5)) (* y 2))",
            env.clone(),
        )
        .unwrap();
        assert_eval_kind("(complex-body 10)", &Sexpr::Number(30.0), Some(env));
    }

    #[test]
    fn test_define_func_closure_capture() {
        let env = new_test_env();
        eval_str("(define n 100)", env.clone()).unwrap();
        eval_str("(define (adder-val x) (+ x n))", env.clone()).unwrap(); // n is captured
        assert_eval_kind("(adder-val 5)", &Sexpr::Number(105.0), Some(env));
    }

    #[test]
    fn test_define_func_recursion() {
        let env = new_test_env();
        eval_str(
            "(define (factorial n) (if (= n 0) 1 (* n (factorial (- n 1)))))",
            env.clone(),
        )
        .unwrap();
        assert_eval_kind("(factorial 5)", &Sexpr::Number(120.0), Some(env));
    }

    // --- Tests for Variadic and Dotted Parameters with Function Define Sugar ---

    #[test]
    fn test_define_func_variadic_params() {
        let env = new_test_env();
        eval_str(
            "(define (fold proc acc lst)
                 (if (null? lst) ; Assuming null? primitive
                     acc
                     (fold proc (proc acc (car lst)) (cdr lst))))", // Assuming car/cdr primitives
            env.clone(),
        )
        .unwrap();

        eval_str("(define (sum-all . nums) (fold + 0 nums))", env.clone()).unwrap();
        assert_eval_kind("(sum-all)", &Sexpr::Number(0.0), Some(env.clone()));
        assert_eval_kind("(sum-all 10)", &Sexpr::Number(10.0), Some(env.clone()));
        assert_eval_kind("(sum-all 1 2 3 4)", &Sexpr::Number(10.0), Some(env));
    }

    #[test]
    fn test_define_func_dotted_params() {
        let env = new_test_env();
        eval_str(
            "(define (process first . rest) (list first (null? rest)))",
            env.clone(),
        )
        .unwrap();

        assert_eval_sexpr("(process 'a)", "(a #t)", Some(env.clone()));
        assert_eval_sexpr("(process 'x 'y 'z)", "(x #f)", Some(env));
    }

    // --- Error Cases for `(define (func ...) ...)` ---

    #[test]
    fn test_define_func_syntax_error_param_not_symbol() {
        let env = new_test_env();
        let result = eval_str("(define (my-func x 123 y) (+ x y))", env.clone());
        assert!(
            matches!(result, Err(EvalError::NotASymbol(_, _))),
            "Parameters in function definition must be symbols"
        );
    }

    #[test]
    fn test_define_func_syntax_error_no_body() {
        let env = new_test_env();
        let result = eval_str("(define (my-func x))", env.clone());
        assert!(
            matches!(result, Err(EvalError::InvalidSpecialForm(_, _))), // "lambda body cannot be empty" or similar
            "Function definition requires a body"
        );
    }

    #[test]
    fn test_define_func_syntax_error_empty_target_list() {
        let env = new_test_env();
        // (define () 1) -> "()" is not a valid (<name> <params...>)
        let result = eval_str("(define () 1)", env.clone());
        assert!(
            matches!(result, Err(EvalError::NotASymbol(_, _))),
            "Function definition target () is invalid"
        );
    }

    #[test]
    fn test_define_func_syntax_error_dotted_param_misplaced_dot() {
        let result = parse_str("(define (my-func a . b . c) x)");
        assert!(matches!(result, Err(ParseError::UnexpectedToken { .. })));
    }

    #[test]
    fn test_define_func_return_value() {
        // Same as variable define: R7RS says "unspecified".
        // Many interpreters return the function name symbol, or an 'ok symbol.
        let env = new_test_env();
        let result_node = eval_str("(define (id x) x)", env.clone()).unwrap();

        // Example: If it returns the function name symbol:
        match &*result_node.kind.borrow() {
            Sexpr::Symbol(s) => assert_eq!(s, "id"),
            _ => panic!(
                "Expected define func to return the func name symbol, got {:?}",
                result_node
            ),
        }
        // Or adapt this assertion based on your chosen return value for define.
    }

    #[test]
    fn test_simple_begin_block() {
        assert_eval_sexpr(
            "(begin
                (define (add x y) (+ x y))
                (define a (add 5 6))
                (define b (* 5 6))
                (list a b 'c))",
            "(11 30 c)",
            None,
        );
    }

    #[test]
    fn test_begin_empty() {
        let env = new_test_env();
        // R7RS: "If there are no expressions in the sequence, begin returns an unspecified value."
        // What does your implementation return? Nil? A special #<unspecified> object?
        let result = eval_str("(begin)", env).unwrap();
        // Assuming you return Nil for an empty begin, like many implementations.
        // Adjust if you have a specific "unspecified" value/node.
        match &*result.kind.borrow() {
            Sexpr::Nil => { /* Ok */ }
            // Or Sexpr::Unspecified => { /* Ok */ },
            other => panic!("(begin) should return Nil or Unspecified, got {:?}", other),
        }
    }

    #[test]
    fn test_begin_single_expression() {
        assert_eval_number("(begin (+ 10 5))", 15.0, None);
    }

    #[test]
    fn test_begin_multiple_expressions_side_effects() {
        let env = new_test_env();
        eval_str("(define x 0)", env.clone()).unwrap(); // Initialize x

        // Expected sequence:
        // 1. x becomes 10
        // 2. y becomes (10 + 5) = 15
        // 3. result is (15 * 2) = 30
        assert_eval_number(
            "(begin (define x 10) (define y (+ x 5)) (* y 2))",
            30.0,
            Some(env.clone()),
        );

        // Check side effects in the environment
        assert_eval_number("x", 10.0, Some(env.clone()));

        assert_eval_number("y", 15.0, Some(env));
    }

    #[test]
    fn test_begin_nested() {
        let env = new_test_env();
        eval_str("(define outer-x 100)", env.clone()).unwrap();
        assert_eval_number(
            "(begin
               (define outer-x 1)
               (begin
                 (define outer-x 2)
                 (+ outer-x 5))
               (+ outer-x 10))",
            12.0,
            Some(env.clone()),
        );
        assert_eval_number("outer-x", 2.0, Some(env));
    }

    #[test]
    fn test_begin_error_in_middle_expression() {
        let env = new_test_env();
        eval_str("(define z 0)", env.clone()).unwrap();
        let result = eval_str(
            "(begin (define z 5) (+ z non-existent-var) (define z 10))",
            env.clone(),
        );

        assert!(
            matches!(
                result,
                Err(EvalError::EnvError(EnvError::UnboundVariable(name, _))) if name == "non-existent-var"
            ),
            "Error in middle of begin should propagate"
        );

        assert_eval_number("z", 5.0, Some(env));
    }

    #[test]
    fn test_let_simple_binding() {
        assert_eval_number("(let ((x 10)) x)", 10.0, None);
    }

    #[test]
    fn test_let_multiple_bindings() {
        assert_eval_number("(let ((x 5) (y 20)) (+ x y))", 25.0, None);
    }

    #[test]
    fn test_let_binding_expressions_evaluated_in_outer_scope() {
        let env = new_test_env();
        eval_str("(define outer-val 100)", env.clone()).unwrap();
        // The 'outer-val' in (y (+ x outer-val)) should refer to the one from define.
        // The 'x' in (y (+ x outer-val)) refers to the 'x' from the *same* let binding group.
        // This is a key property of `let` vs `let*`: all value expressions are evaluated *before* any bindings are made.
        let result = eval_str("(let ((x 10) (y (+ x outer-val))) y)", env.clone());
        // If `let` desugars to ((lambda (x y) y) 10 (+ x outer-val)),
        // then `(+ x outer-val)` is evaluated in the *outer* scope where `x` is not yet bound.
        // This should cause an error for `x`. This is standard `let` behavior.
        assert!(
            matches!(result.clone(), Err(EvalError::EnvError(EnvError::UnboundVariable(name, _))) if name == "x"),
            "Value expressions in let are evaluated in outer scope; 'x' should not be visible to 'y's value expr. Error was: {:?}",
            result.err()
        );

        // To make the above work as one might intuitively expect, you'd need let* or nested lets:
        // (let ((x 10)) (let ((y (+ x outer-val))) y))
        assert_eval_number(
            "(let ((x 10)) (let ((y (+ x outer-val))) y))",
            110.0,
            Some(env),
        );
    }

    #[test]
    fn test_let_value_expressions_order_of_eval() {
        // R7RS: "It is an error for a <variable> to appear more than once in the list of variables
        // being bound. The <init>s are evaluated in the outer environment (in some unspecified order)"
        // For testing, we can see if side effects from one init are visible to another IF your impl
        // desugars to ((lambda (vars...) body) init-exprs...) where init-exprs are eval'd sequentially by
        // your argument evaluation logic.
        let env = new_test_env();
        eval_str("(define a 0)", env.clone()).unwrap();
        // If init exprs are evaluated left-to-right by your lambda arg eval:
        // 1. (begin (define a 10) a) -> x bound to 10, global 'a' becomes 10
        // 2. y bound to global 'a' (which is now 10)
        // Then (+ 10 10) -> 20
        // If evaluated in some other order, or truly "outer" before any define, y would be 0.
        // Most desugaring to lambda implies left-to-right eval of init expressions.
        assert_eval_number(
            "(let ((x (begin (define a 10) a)) (y a)) (+ x y))",
            20.0,
            Some(env),
        );
    }

    #[test]
    fn test_let_no_bindings() {
        // (let () <body>) is like ((lambda () <body>))
        assert_eval_number("(let () 123)", 123.0, None);
    }

    #[test]
    fn test_let_multiple_body_expressions() {
        // Inside let: x=1. Then (define y (+ 1 2)) -> y=3 (local to let's new scope).
        // Then (* 3 1) -> 3.
        assert_eval_number("(let ((x 1)) (define y (+ x 2)) (* y x))", 3.0, None);
    }

    #[test]
    fn test_let_shadowing_outer_variable() {
        let env = new_test_env();
        eval_str("(define x 100)", env.clone()).unwrap();
        assert_eval_number("(let ((x 10)) (+ x 5))", 15.0, Some(env.clone())); // Inner x is 10

        // Ensure outer x is unchanged
        assert_eval_number("x", 100.0, Some(env));
    }

    #[test]
    fn test_let_empty_body() {
        let env = new_test_env();
        // (let ((x 1)) ) -> result is unspecified (like an empty begin or lambda body)
        let result = eval_str("(let ((x 1)))", env).unwrap();
        match &*result.kind.borrow() {
            // Assuming Nil for unspecified
            Sexpr::Nil => { /* Ok */ }
            other => panic!(
                "let with empty body should return Nil/Unspecified, got {:?}",
                other
            ),
        }
    }

    // --- Error cases for `let` ---
    #[test]
    fn test_let_syntax_error_no_bindings_list() {
        let env = new_test_env();
        let result = eval_str("(let x x)", env); // Bindings part 'x' is not a list
        assert!(
            matches!(result, Err(EvalError::InvalidSpecialForm(_, _))),
            "Let bindings must be a list"
        );
    }

    #[test]
    fn test_let_syntax_error_binding_not_a_pair() {
        let env = new_test_env();
        let result = eval_str("(let (x) x)", env.clone()); // Binding 'x' is not (var expr)
        assert!(
            matches!(result, Err(EvalError::InvalidSpecialForm(_, _))),
            "Let binding 'x' is not a pair"
        );

        let result2 = eval_str("(let ((x 1) y (z 3)) y)", env); // Binding 'y' is not (var expr)
        assert!(
            matches!(result2, Err(EvalError::InvalidSpecialForm(_, _))),
            "Let binding 'y' is not a pair"
        );
    }

    #[test]
    fn test_let_syntax_error_binding_var_not_symbol() {
        let env = new_test_env();
        let result = eval_str("(let ((123 1)) 123)", env); // Var '123' is not a symbol
        assert!(
            matches!(result, Err(EvalError::NotASymbol(symbol, _)) if symbol == Sexpr::Number(123.0)),
            "Let binding variable must be a symbol"
        );
    }

    #[test]
    fn test_let_syntax_error_binding_pair_wrong_length() {
        let env = new_test_env();
        let result1 = eval_str("(let ((x)) x)", env.clone()); // (x) not (x expr)
        assert!(
            matches!(result1, Err(EvalError::InvalidSpecialForm(_, _))),
            "Let binding (x) wrong length"
        );

        let result2 = eval_str("(let ((x 1 2)) x)", env); // (x 1 2) not (x expr)
        assert!(
            matches!(result2, Err(EvalError::InvalidSpecialForm(_, _))),
            "Let binding (x 1 2) wrong length"
        );
    }

    #[test]
    fn test_let_error_in_value_expression() {
        let env = new_test_env();
        let result = eval_str("(let ((x (+ 1 non-existent))) x)", env);
        assert!(
            matches!(result, Err(EvalError::EnvError(EnvError::UnboundVariable(name, _))) if name == "non-existent"),
            "Error in let value expression should propagate"
        );
    }

    #[test]
    fn test_let_duplicate_variable_in_bindings() {
        let env = new_test_env();
        // R7RS: "It is an error for a <variable> to appear more than once in the list of variables being bound."
        let result = eval_str("(let ((x 1) (x 2)) x)", env);
        // This error might be caught during the parsing of the `let` form (before desugaring)
        // or by the `lambda` form if the desugared `(lambda (x x) ...)` has duplicate param check.
        assert!(
            matches!(result, Err(EvalError::InvalidSpecialForm(_, _))),
            "Duplicate variable in let bindings should be an error. Got: {:?}",
            result.err()
        );
    }

    // --- Tests for `set!` ---

    #[test]
    fn test_set_bang_simple() {
        let env = new_test_env();
        eval_str("(define x 10)", env.clone()).unwrap();
        assert_env_var_is_number(&env, "x", 10.0, "test_set_bang_simple initial");

        // R7RS: Return value of set! is unspecified.
        // We primarily care about the side effect.
        // Your `eval_str` will panic if there's an error from `set!`.
        eval_str("(set! x 20)", env.clone()).unwrap();
        assert_env_var_is_number(&env, "x", 20.0, "test_set_bang_simple after set!");
    }

    #[test]
    fn test_set_bang_value_is_expression() {
        let env = new_test_env();
        eval_str("(define y 5)", env.clone()).unwrap();
        eval_str("(set! y (+ y 15))", env.clone()).unwrap();
        assert_env_var_is_number(&env, "y", 20.0, "test_set_bang_value_is_expression");
    }

    #[test]
    fn test_set_bang_in_lambda_modifies_captured_var() {
        let env = new_test_env();
        eval_str("(define val 100)", env.clone()).unwrap();
        eval_str(
            "(define mutator (lambda (new-val) (set! val new-val)))",
            env.clone(),
        )
        .unwrap();

        assert_env_var_is_number(&env, "val", 100.0, "set_bang_in_lambda initial");
        eval_str("(mutator 200)", env.clone()).unwrap(); // Call the mutator
        assert_env_var_is_number(&env, "val", 200.0, "set_bang_in_lambda after mutator");
    }

    #[test]
    fn test_set_bang_in_let_modifies_let_var() {
        let env = new_test_env(); // Global env is parent
        // (let ((a 1)) (set! a 2) a) -> returns 2
        assert_eval_number("(let ((a 1)) (set! a 2) a)", 2.0, Some(env));
    }

    #[test]
    fn test_set_bang_return_value_is_unspecified_check_side_effect() {
        let env = new_test_env();
        eval_str("(define z 1)", env.clone()).unwrap();
        let result_node = eval_str("(set! z 2)", env.clone()).unwrap();
        // R7RS: "unspecified". Your implementation might return the new value, 'ok, or Nil/Unspecified.
        // We won't assert the exact return value here, only that it succeeded and had the side effect.
        // If you have a specific Sexpr::Unspecified, you could use assert_eval_kind for it.
        // For now, unwrap() above confirms it didn't error.
        let _ = result_node; // Use the result to avoid unused variable warning
        assert_env_var_is_number(&env, "z", 2.0, "test_set_bang_return_value_is_unspecified");
    }

    // --- Error Cases for `set!` ---
    #[test]
    fn test_set_bang_error_unbound_variable() {
        // Using your dummy error variant for comparison
        let dummy_error = EvalError::EnvError(EnvError::UnboundVariable(
            "unbound-var".to_string(),
            Default::default(),
        ));
        assert_eval_error("(set! unbound-var 123)", &dummy_error, None);
    }

    #[test]
    fn test_set_bang_arity_error_too_few_args_zero() {
        let dummy_error = EvalError::InvalidSpecialForm("set!".to_string(), Default::default()); // Or ArityMismatch
        assert_eval_error("(set!)", &dummy_error, None);
    }

    #[test]
    fn test_set_bang_arity_error_too_few_args_one() {
        let dummy_error = EvalError::InvalidSpecialForm("set!".to_string(), Default::default()); // Or ArityMismatch
        assert_eval_error("(set! x)", &dummy_error, None);
    }

    #[test]
    fn test_set_bang_arity_error_too_many_args() {
        let dummy_error = EvalError::InvalidSpecialForm("set!".to_string(), Default::default()); // Or ArityMismatch
        assert_eval_error("(set! x 1 2)", &dummy_error, None);
    }

    #[test]
    fn test_set_bang_variable_not_a_symbol() {
        // Assuming NotASymbol(Sexpr, Span)
        let not_symbol_val = Sexpr::Number(123.0); // The actual non-symbol Sexpr
        let dummy_error = EvalError::NotASymbol(not_symbol_val, Default::default());
        assert_eval_error("(set! 123 456)", &dummy_error, None);
    }

    #[test]
    fn test_set_bang_error_in_value_expression() {
        let env = new_test_env();
        eval_str("(define v 50)", env.clone()).unwrap();
        let dummy_error = EvalError::EnvError(EnvError::UnboundVariable(
            "non-existent".to_string(),
            Default::default(),
        ));
        assert_eval_error(
            "(set! v (+ v non-existent))",
            &dummy_error,
            Some(env.clone()),
        );
        assert_env_var_is_number(
            &env,
            "v",
            50.0,
            "set_bang_error_in_value_expression var unchanged",
        );
    }

    // --- Tests for `letrec` ---
    // `letrec` is for mutually recursive bindings, typically functions.

    #[test]
    fn test_letrec_simple_non_recursive_binding() {
        assert_eval_number("(letrec ((x 10)) x)", 10.0, None);
    }

    #[test]
    fn test_letrec_simple_function_definition_and_call() {
        assert_eval_number(
            "(letrec ((my-id (lambda (val) val))) (my-id 123))",
            123.0,
            None,
        );
    }

    #[test]
    fn test_letrec_mutual_recursion_even_odd() {
        let code = "(letrec ((is-even? (lambda (n) \
                               (if (= n 0) \
                                   #t \
                                   (is-odd? (- n 1))))) \
                  (is-odd? (lambda (n) \
                              (if (= n 0) \
                                  #f \
                                  (is-even? (- n 1)))))) \
           (list (is-even? 4) (is-odd? 3) (is-even? 5)))";
        assert_eval_sexpr(code, "(#t #t #f)", None);
    }

    #[test]
    fn test_letrec_recursion_factorial() {
        let code = "(letrec ((fact (lambda (n) \
                          (if (= n 0) \
                              1 \
                              (* n (fact (- n 1))))))) \
           (fact 5))";
        assert_eval_number(code, 120.0, None);
    }

    #[test]
    fn test_letrec_value_expressions_see_all_bindings_for_lambdas() {
        let code = "(letrec ((f (lambda () (g))) \
                  (g (lambda () 10))) \
           (f))";
        assert_eval_number(code, 10.0, None);
    }

    #[test]
    fn test_letrec_empty_bindings() {
        assert_eval_number("(letrec () 42)", 42.0, None);
    }

    #[test]
    fn test_letrec_multiple_body_exprs() {
        // define in letrec body creates a local binding within letrec's scope
        assert_eval_number("(letrec ((a 1)) (define b (+ a 2)) (* a b))", 3.0, None);
    }

    // --- Error cases for `letrec` ---
    #[test]
    fn test_letrec_error_direct_use_of_binding_in_non_lambda_value_expr() {
        // Standard Scheme: error to use a letrec-bound var in a value expr
        // if that var refers to a value that isn't a lambda, before that var has been assigned
        // from its (potentially recursive) value expression.
        // We allow this. Why not?
        //let dummy_unbound = EvalError::EnvError(EnvError::UnboundVariable(
        //    "x".to_string(),
        //    Default::default(),
        //));
        //// Or EvalError::AccessUninitializedVariable("x".to_string(), Default::default())
        //assert_eval_error("(letrec ((x 10) (y x)) y)", &dummy_unbound, None);

        let dummy_unbound_y = EvalError::EnvError(EnvError::UnboundVariable(
            "y".to_string(),
            Default::default(),
        ));
        assert_eval_error("(letrec ((x y) (y 10)) x)", &dummy_unbound_y, None);
    }

    #[test]
    fn test_letrec_error_in_value_expression_after_placeholder_binding() {
        let dummy_error = EvalError::EnvError(EnvError::UnboundVariable(
            "non-existent-fn".to_string(),
            Default::default(),
        ));
        assert_eval_error(
            "(letrec ((f (lambda () (non-existent-fn)))) (f))",
            &dummy_error,
            None,
        );
    }

    #[test]
    fn test_letrec_syntax_error_no_bindings_list() {
        let dummy_error = EvalError::InvalidSpecialForm("letrec".to_string(), Default::default());
        assert_eval_error("(letrec x x)", &dummy_error, None);
    }

    #[test]
    fn test_letrec_syntax_error_binding_not_a_pair() {
        let dummy_error = EvalError::InvalidSpecialForm("letrec".to_string(), Default::default());
        assert_eval_error("(letrec (x) x)", &dummy_error, None);
    }

    #[test]
    fn test_letrec_syntax_error_binding_var_not_symbol() {
        let not_symbol_val = Sexpr::Number(123.0);
        let dummy_error = EvalError::NotASymbol(not_symbol_val, Default::default());
        assert_eval_error("(letrec ((123 1)) 123)", &dummy_error, None);
    }

    #[test]
    fn test_letrec_duplicate_variable_in_bindings() {
        // R7RS: "It is an error for a <variable> to appear more than once."
        let dummy_error = EvalError::InvalidSpecialForm("letrec".to_string(), Default::default()); // Or DuplicateBinding
        assert_eval_error("(letrec ((x 1) (x (lambda () 2))) x)", &dummy_error, None);
    }

    // --- Tests for `let*` ---
    #[test]
    fn test_let_star_simple_binding() {
        assert_eval_number("(let* ((x 10)) x)", 10.0, None);
    }

    #[test]
    fn test_let_star_sequential_bindings() {
        // x is 5, y is (+ x 10) = 15. Result (+ x y) = (+ 5 15) = 20.
        let code = "(let* ((x 5) (y (+ x 10))) (+ x y))";
        assert_eval_number(code, 20.0, None);
    }

    #[test]
    fn test_let_star_value_expressions_can_refer_to_prior_bindings() {
        let env = new_test_env();
        eval_str("(define outer-val 100)", env.clone()).unwrap(); // Use eval_str to define
        // y's expression (+ x outer-val) uses x from the same let*
        // z's expression (+ y outer-val) uses y from the same let*
        let code = "(let* ((x 10) \
                           (y (+ x outer-val)) \
                           (z (+ y outer-val))) \
                      z)"; // x=10, y=110, z=210
        assert_eval_number(code, 210.0, Some(env));
    }

    #[test]
    fn test_let_star_no_bindings() {
        assert_eval_number("(let* () 123)", 123.0, None);
    }

    #[test]
    fn test_let_star_multiple_body_expressions() {
        // Inside let*: x=1. Then (define local-y (+ x 2)) -> local-y=3 (local to let*'s scope).
        // Then (* local-y x) -> 3.
        let code = "(let* ((x 1)) (define local-y (+ x 2)) (* local-y x))";
        assert_eval_number(code, 3.0, None);
    }

    #[test]
    fn test_let_star_shadowing_outer_variable() {
        let env = new_test_env();
        eval_str("(define x 100)", env.clone()).unwrap();
        assert_eval_number("(let* ((x 10)) (+ x 5))", 15.0, Some(env.clone()));
        // Check outer x is unchanged using an eval_str that returns Result and then assert_env_var_is_number
        let _ = eval_str("x", env.clone()).unwrap(); // just to make sure 'x' is evaluated for the next assert
        assert_env_var_is_number(&env, "x", 100.0, "test_let_star_shadowing_outer_variable");
    }

    #[test]
    fn test_let_star_shadowing_within_let_star() {
        // First x is 10. Second x (shadowing first) is (+ x 1) = 11. Result is this second x.
        let code = "(let* ((x 10) (x (+ x 1))) x)";
        assert_eval_number(code, 11.0, None);
    }

    #[test]
    fn test_let_star_empty_body() {
        // (let* ((x 1)) ) -> result is unspecified (like an empty begin or lambda body)
        // Assuming Nil for unspecified, as per your current implementation for empty begin/let.
        assert_eval_kind("(let* ((x 1)))", &Sexpr::Nil, None);
    }

    // --- Error cases for `let*` ---
    #[test]
    fn test_let_star_syntax_error_no_bindings_list() {
        let dummy_error = EvalError::InvalidSpecialForm("let*".to_string(), Default::default());
        assert_eval_error("(let* x x)", &dummy_error, None);
    }

    #[test]
    fn test_let_star_syntax_error_binding_not_a_pair() {
        let dummy_error = EvalError::InvalidSpecialForm("let*".to_string(), Default::default());
        assert_eval_error("(let* (x) x)", &dummy_error, None); // Binding 'x' is not (var expr)
        assert_eval_error("(let* ((a 1) b (c 2)) b)", &dummy_error, None); // Binding 'b' is not (var expr)
    }

    #[test]
    fn test_let_star_syntax_error_binding_var_not_symbol() {
        let not_symbol_val = Sexpr::Number(123.0);
        let dummy_error = EvalError::NotASymbol(not_symbol_val, Default::default());
        assert_eval_error("(let* ((123 1)) 123)", &dummy_error, None);
    }

    #[test]
    fn test_let_star_syntax_error_binding_pair_wrong_length() {
        let dummy_error = EvalError::InvalidSpecialForm("let*".to_string(), Default::default());
        assert_eval_error("(let* ((x)) x)", &dummy_error, None); // (x) not (x expr)
        assert_eval_error("(let* ((x 1 2)) x)", &dummy_error, None); // (x 1 2) not (x expr)
    }

    #[test]
    fn test_let_star_error_in_value_expression() {
        let dummy_error = EvalError::EnvError(EnvError::UnboundVariable(
            "non-existent".to_string(),
            Default::default(),
        ));
        assert_eval_error("(let* ((x (+ 1 non-existent))) x)", &dummy_error, None);
    }

    #[test]
    fn test_let_star_error_in_later_value_expression_first_binding_ok() {
        let env = new_test_env(); // Create env to check side effects
        let dummy_error = EvalError::EnvError(EnvError::UnboundVariable(
            "non-existent".to_string(),
            Default::default(),
        ));
        // x should be bound to 10 in a temporary scope for y's expression evaluation.
        // If (+ x non-existent) errors, x=10 should not pollute the outer env.
        assert_eval_error(
            "(let* ((x 10) (y (+ x non-existent))) y)",
            &dummy_error,
            Some(env.clone()),
        );
        assert_env_var_is_not_defined(&env, "x", "test_let_star_error_in_later_value_expression");
        assert_env_var_is_not_defined(&env, "y", "test_let_star_error_in_later_value_expression");
    }

    #[test]
    fn test_let_star_duplicate_variable_in_bindings_is_allowed_and_shadows() {
        // Unlike `let`, `let*` allows duplicate variables in its binding list because
        // it's like nested `let`s. (let* ((x 1) (x (+ x 10))) x) is valid and was tested in
        // test_let_star_shadowing_within_let_star. No specific error test needed here for duplicates.
        // This test just serves as a reminder of this behavior.
        let code = "(let* ((x 1) (x (+ x 10))) x)"; // x becomes 1, then new x becomes 1+10=11
        assert_eval_number(code, 11.0, None);
    }

    // --- Tests for Quasiquote (`), Unquote (,), Unquote-Splicing (,@) Evaluation ---
    #[test]
    fn test_eval_quasiquote_literal_atom() {
        assert_eval_sexpr("`foo", "foo", None);
        assert_eval_number("`123", 123.0, None);
        assert_eval_sexpr("`#t", "#t", None);
        assert_eval_sexpr("`\"hello\"", "\"hello\"", None);
        assert_eval_sexpr("`()", "()", None);
    }

    #[test]
    fn test_eval_quasiquote_literal_list() {
        assert_eval_sexpr("`(a b c)", "(a b c)", None);
        assert_eval_sexpr("`(a (b c) d)", "(a (b c) d)", None);
        assert_eval_sexpr("`(a . b)", "(a . b)", None);
    }

    #[test]
    fn test_eval_unquote_simple() {
        let env = new_test_env();
        eval_str("(define x 10)", env.clone()).unwrap();
        assert_eval_sexpr("`(a ,x c)", "(a 10 c)", Some(env.clone()));

        eval_str("(define y '(foo bar))", env.clone()).unwrap();
        assert_eval_sexpr("`(a ,y c)", "(a (foo bar) c)", Some(env));
    }

    #[test]
    fn test_eval_unquote_expression() {
        let env = new_test_env();
        eval_str("(define x 5)", env.clone()).unwrap();
        assert_eval_sexpr("`(a ,(+ x 2) c)", "(a 7 c)", Some(env));
    }

    #[test]
    fn test_eval_unquote_splicing_simple() {
        let env = new_test_env();
        eval_str("(define items '(1 2 3))", env.clone()).unwrap();
        assert_eval_sexpr("`(a ,@items c)", "(a 1 2 3 c)", Some(env.clone()));

        eval_str("(define head '(x y))", env.clone()).unwrap();
        eval_str("(define tail '(z))", env.clone()).unwrap();
        assert_eval_sexpr("`(,@head item ,@tail)", "(x y item z)", Some(env));
    }

    #[test]
    fn test_eval_unquote_splicing_expression_evaluates_to_list() {
        let env = new_test_env();
        eval_str("(define (get-list n) (list n (+ n 1)))", env.clone()).unwrap();
        assert_eval_sexpr(
            "`(start ,@(get-list 10) end)",
            "(start 10 11 end)",
            Some(env),
        );
    }

    #[test]
    fn test_eval_unquote_splicing_empty_list() {
        assert_eval_sexpr("`(a ,@'() c)", "(a c)", None);
        assert_eval_sexpr("`(,@'() c)", "(c)", None);
        assert_eval_sexpr("`(a ,@'())", "(a)", None);
        assert_eval_sexpr("`(,@'())", "()", None);
    }

    #[test]
    fn test_eval_quasiquote_mixed_unquote_and_splice() {
        let env = new_test_env();
        eval_str("(define item 'x)", env.clone()).unwrap();
        eval_str("(define prefix '(a b))", env.clone()).unwrap();
        eval_str("(define suffix '(y z))", env.clone()).unwrap();
        assert_eval_sexpr(
            "`(,@prefix ,item item2 ,@suffix final)",
            "(a b x item2 y z final)",
            Some(env),
        );
    }

    #[test]
    fn test_eval_nested_quasiquote_no_inner_unquote_active() {
        // `(a `(b ,x) c) -> (a (quasiquote (b (unquote x))) c)
        // The inner (unquote x) is not evaluated by the outer quasiquote's context.
        let env = new_test_env();
        eval_str("(define x 100)", env.clone()).unwrap(); // This x should not be used by inner ,x
        assert_eval_sexpr(
            "`(a `(b ,x) c)",
            "(a (quasiquote (b (unquote x))) c)", // x remains as symbol 'x'
            Some(env),
        );
    }

    #[test]
    fn test_eval_nested_quasiquote_with_inner_unquote_activated() {
        // `(a ,`(b ,x) c)
        // Outer unquote activates evaluation of `(b ,x)
        // This inner quasiquote then evaluates, and its unquote for x becomes active.
        let env = new_test_env();
        eval_str("(define x 10)", env.clone()).unwrap();
        assert_eval_sexpr("`(a ,`(b ,x) c)", "(a (b 10) c)", Some(env));
    }

    #[test]
    fn test_eval_nested_quasiquote_with_splicing() {
        let env = new_test_env();
        eval_str("(define inner-val 10)", env.clone()).unwrap();
        eval_str("(define items-to-splice '(foo bar))", env.clone()).unwrap();
        // `(level1 ,`(level2 ,inner-val ,@items-to-splice) level1-end)
        assert_eval_sexpr(
            "`(level1 ,`(level2 ,inner-val ,@items-to-splice) level1-end)",
            "(level1 (level2 10 foo bar) level1-end)",
            Some(env),
        );
    }

    #[test]
    fn test_eval_quasiquote_level_management_complex() {
        let env = new_test_env();
        eval_str("(define x 'outer-x)", env.clone()).unwrap();
        eval_str("(define y 'outer-y)", env.clone()).unwrap();
        eval_str("(define z 'outer-z)", env.clone()).unwrap();

        assert_eval_sexpr(
            "`(a ,x `(b ,y ,,z))",
            "(a outer-x (quasiquote (b (unquote y) (unquote outer-z))))",
            Some(env.clone()),
        );

        assert_eval_sexpr(
            "`(a ,x `(b ,y ,,z ,`(c ,x ,,y)))",
            "(a outer-x (quasiquote (b (unquote y) (unquote outer-z) (unquote (quasiquote (c (unquote x) (unquote outer-y)))))))",
            Some(env.clone()),
        );
    }

    #[test]
    fn test_eval_unquote_wrong_number_of_args() {
        let dummy_error = EvalError::InvalidSpecialForm(
            "unquote-splicing not in proper list context".to_string(),
            Default::default(),
        );
        assert_eval_error("`(a (unquote 1 2))", &dummy_error, None);
    }

    #[test]
    fn test_eval_quasiquote_dotted_list() {
        let env = new_test_env();
        eval_str("(define val 10)", env.clone()).unwrap();
        assert_eval_sexpr("`(a . ,val)", "(a . 10)", Some(env.clone()));
        assert_eval_sexpr("`(a b . ,val)", "(a b . 10)", Some(env));
    }

    #[test]
    fn test_eval_quasiquote_splicing_into_dotted_position_not_allowed_by_standard() {
        // ,@ must be in a list context. (a . ,@ L) is usually an error because '.' expects a single datum.
        // Your parser produced (quasiquote (a unquote-splicing L)) for `(a . ,@L) if L is '(1 2)
        // This evaluation depends on how your `evaluate_quasiquote_recursive` handles
        // (unquote-splicing L) when it's not directly an element of a list being built.
        // If it treats it like (unquote L) in this context if L is not a list for splicing.
        // Or it errors. Standard Scheme typically errors.
        let env = new_test_env();
        eval_str("(define my-list '(1 2))", env.clone()).unwrap();

        // If parser makes it `(quasiquote (a unquote-splicing my-list))`
        // And `evaluate_quasiquote_recursive` encounters `(unquote-splicing my-list)` as the cdr.
        // It should probably error if not in a list construction loop.
        // Let's assume an error similar to "unquote-splicing not in list context"
        // or a type error if it tries to make a pair with a non-spliced list.
        let dummy_error = EvalError::InvalidSpecialForm(
            "unquote-splicing not in proper list context".to_string(),
            Default::default(),
        );
        assert_eval_error("`(a . ,@my-list)", &dummy_error, Some(env));
    }

    #[test]
    fn test_eval_quasiquote_splicing_in_car_of_dotted_list() {
        // `(,@items . end)` where items is '(1 2)
        // parser -> (quasiquote ((unquote-splicing items) . end))
        // eval should produce (1 2 . end)
        let env = new_test_env();
        eval_str("(define items '(1 2))", env.clone()).unwrap();
        eval_str("(define end-val 'final)", env.clone()).unwrap();
        assert_eval_sexpr("`(,@items . ,end-val)", "(1 2 . final)", Some(env));
    }

    // --- Error Cases for Quasiquote Evaluation ---
    #[test]
    fn test_eval_error_unquote_outside_quasiquote() {
        // Parser might create (unquote x), evaluator should reject it at top level.
        let dummy_error = EvalError::InvalidSpecialForm(
            "unquote outside quasiquote".to_string(),
            Default::default(),
        );
        assert_eval_error("(unquote x)", &dummy_error, None);
    }

    #[test]
    fn test_eval_error_unquote_splicing_outside_quasiquote() {
        let dummy_error = EvalError::InvalidSpecialForm(
            "unquote-splicing outside quasiquote".to_string(),
            Default::default(),
        );
        assert_eval_error("(unquote-splicing x)", &dummy_error, None);
    }

    #[test]
    fn test_eval_error_unquote_splicing_non_list_value() {
        let env = new_test_env();
        eval_str("(define not-a-list 123)", env.clone()).unwrap();
        // Error should be TypeMismatch or similar when trying to splice a number
        let dummy_error = EvalError::TypeMismatch {
            expected: "list".to_string(),
            found: Sexpr::Symbol("not-a-list".to_string()),
            span: Default::default(),
        };
        assert_eval_error("`(a ,@not-a-list c)", &dummy_error, Some(env));
    }

    #[test]
    fn test_eval_error_unquote_missing_argument() {
        let result1 = parse_str("(,)");
        assert!(matches!(result1, Err(ParseError::UnexpectedToken { .. })));
    }

    #[test]
    fn test_eval_error_unquote_splicing_missing_argument() {
        let result1 = parse_str("(,@)");
        assert!(matches!(result1, Err(ParseError::UnexpectedToken { .. })));
    }

    #[test]
    fn test_apply_with_list() {
        assert_eval_kind("(apply + '(1 2 3))", &Sexpr::Number(6.0), None);
        assert_eval_sexpr("(apply (lambda (x y) (list x y)) '(1 2))", "(1 2)", None);
        assert_eval_kind("(apply + 1 2 '(3 4 5))", &Sexpr::Number(15.0), None);
        assert_eval_kind("(apply + 1 2 3 '())", &Sexpr::Number(6.0), None);
        assert_eval_kind(
            "(apply (lambda x (apply + x)) 1 2 3 '(4 5))",
            &Sexpr::Number(15.0),
            None,
        );
        assert_eval_kind(
            "(apply if '(#t 1 unbound-variable))",
            &Sexpr::Number(1.0),
            None,
        );
    }

    #[test]
    fn test_apply_invalid_procedure() {
        let env = new_test_env();
        assert_eval_error(
            "(apply 1 '(2 3))",
            &EvalError::NotAProcedure(Sexpr::Number(1.0), Span::default()),
            Some(env.clone()),
        );
    }

    #[test]
    fn test_apply_insufficient_arguments() {
        let env = new_test_env();
        assert_eval_error(
            "(apply)",
            &EvalError::InvalidArguments(
                "Primitive 'apply' expects at least 2 arguments, got 0".to_string(),
                Span::default(),
            ),
            Some(env.clone()),
        );
        assert_eval_error(
            "(apply +)",
            &EvalError::InvalidArguments(
                "Primitive 'apply' expects at least 2 arguments, got 1".to_string(),
                Span::default(),
            ),
            Some(env.clone()),
        );
    }

    // --- Tests for `eq?` ---
    #[test]
    fn test_eq_bang_symbols() {
        assert_eval_sexpr("(eq? 'foo 'foo)", "#t", None);
        assert_eval_sexpr("(eq? 'foo 'bar)", "#f", None);
        let env = new_test_env();
        eval_str("(define x 'a)", env.clone()).unwrap();
        eval_str("(define y 'a)", env.clone()).unwrap();
        eval_str("(define z x)", env.clone()).unwrap();
        assert_eval_sexpr("(eq? x y)", "#t", Some(env.clone())); // Symbols with same name are eq?
        assert_eval_sexpr("(eq? x z)", "#t", Some(env));
    }

    /*
    #[test]
    fn test_eq_bang_booleans() {
        assert_eval_sexpr("(eq? #t #t)", "#t", None);
        assert_eval_sexpr("(eq? #f #f)", "#t", None);
        assert_eval_sexpr("(eq? #t #f)", "#f", None);
        let env = new_test_env();
        eval_str("(define t1 #t)", env.clone()).unwrap();
        eval_str("(define t2 #t)", env.clone()).unwrap();
        assert_eval_sexpr("(eq? t1 t2)", "#t", Some(env));
    }

    #[test]
    fn test_eq_bang_empty_list() {
        assert_eval_sexpr("(eq? '() '())", "#t", None);
        assert_eval_sexpr("(eq? (list) '())", "#t", None); // If (list) produces the unique '()
    }

    #[test]
    fn test_eq_bang_numbers() {
        assert_eval_sexpr("(eq? 1 1)", "#t", None); // Small integers might be interned or eqv? implies eq?
        assert_eval_sexpr("(eq? 1 2)", "#f", None);
        assert_eval_sexpr("(eq? 1.0 1.0)", "#t", None); // Similar to integers
        assert_eval_sexpr("(eq? 1.0 1.1)", "#f", None);

        let env = new_test_env();
        eval_str("(define n1 1000.0)", env.clone()).unwrap();
        eval_str("(define n2 1000.0)", env.clone()).unwrap();
        // For numbers, (eq? x y) if (eqv? x y) is true and x is not a procedure or mutable.
        // Since your numbers are f64 (immutable values), this should hold.
        assert_eval_sexpr("(eq? n1 n2)", "#t", Some(env.clone()));
        eval_str("(define n3 (+ 500.0 500.0))", env.clone()).unwrap(); // n3 is also 1000.0
        assert_eval_sexpr("(eq? n1 n3)", "#t", Some(env));
    }

    #[test]
    fn test_eq_bang_strings() {
        assert_eval_sexpr("(eq? \"foo\" \"foo\")", "#t", None); // Literal strings might be interned
        assert_eval_sexpr("(eq? \"foo\" \"bar\")", "#f", None);

        let env = new_test_env();
        eval_str("(define s1 \"abc\")", env.clone()).unwrap();
        eval_str("(define s2 \"abc\")", env.clone()).unwrap();
        // R7RS: if (eqv? x y) and x is not mutable pair/string/vector/bytevector/procedure => (eq? x y) is true
        // If your strings are immutable values, this should be true.
        assert_eval_sexpr("(eq? s1 s2)", "#t", Some(env.clone()));
        // If strings are distinct mutable objects, this might be #f. Depends on your Sexpr::String impl.
        // Assuming immutable string values or effective interning for literals.
    }

    #[test]
    fn test_eq_bang_pairs_and_lists() {
        assert_eval_sexpr("(eq? (cons 1 2) (cons 1 2))", "#f", None); // Distinct pairs
        assert_eval_sexpr("(eq? (list 1 2) (list 1 2))", "#f", None); // Distinct lists

        let env = new_test_env();
        eval_str("(define p1 (cons 'a 'b))", env.clone()).unwrap();
        eval_str("(define p2 (cons 'a 'b))", env.clone()).unwrap();
        eval_str("(define p3 p1)", env.clone()).unwrap();
        assert_eval_sexpr("(eq? p1 p2)", "#f", Some(env.clone())); // Distinct objects
        assert_eval_sexpr("(eq? p1 p3)", "#t", Some(env.clone())); // Same object

        eval_str("(define lst1 '(x y))", env.clone()).unwrap();
        eval_str("(define lst2 '(x y))", env.clone()).unwrap(); // Should be a new list object from (quote (x y))
        assert_eval_sexpr("(eq? lst1 lst2)", "#f", Some(env.clone())); // Usually false, new list created by quote
        // unless quote interns deeply, which is uncommon.
    }

    #[test]
    fn test_eq_bang_procedures() {
        let env = new_test_env();
        eval_str("(define f (lambda (x) x))", env.clone()).unwrap();
        eval_str("(define g (lambda (x) x))", env.clone()).unwrap();
        eval_str("(define h f)", env.clone()).unwrap();

        assert_eval_sexpr("(eq? f g)", "#f", Some(env.clone())); // Distinct lambda objects
        assert_eval_sexpr("(eq? f h)", "#t", Some(env.clone())); // Same lambda object
        assert_eval_sexpr("(eq? + +)", "#t", Some(env)); // Primitives should be eq? to themselves
    }

    #[test]
    fn test_eq_bang_arity() {
        let arity_error = &EvalError::InvalidArguments("".into(), Span::default()); // Dummy
        assert_eval_error("(eq? 1)", &arity_error, None);
        assert_eval_error("(eq? 1 2 3)", &arity_error, None);
    }

    // --- Tests for `eqv?` ---
    #[test]
    fn test_eqv_bang_symbols() {
        assert_eval_sexpr("(eqv? 'foo 'foo)", "#t", None);
        assert_eval_sexpr("(eqv? 'foo 'bar)", "#f", None);
    }

    #[test]
    fn test_eqv_bang_booleans() {
        assert_eval_sexpr("(eqv? #t #t)", "#t", None);
        assert_eval_sexpr("(eqv? #t #f)", "#f", None);
    }

    #[test]
    fn test_eqv_bang_empty_list() {
        assert_eval_sexpr("(eqv? '() '())", "#t", None);
    }

    #[test]
    fn test_eqv_bang_numbers() {
        assert_eval_sexpr("(eqv? 1 1)", "#t", None);
        assert_eval_sexpr("(eqv? 1 2)", "#f", None);
        assert_eval_sexpr("(eqv? 1.0 1)", "#t", None); // Numerically equal, both inexact (f64)
        assert_eval_sexpr("(eqv? 1.0 1.0)", "#t", None);
        assert_eval_sexpr("(eqv? 1.0 1.0000000000000001)", "#t", None); // Standard f64 comparison
        assert_eval_sexpr("(eqv? 1.0 1.1)", "#f", None);
        // Add tests for +0.0 vs -0.0 if you distinguish them (R7RS says (eqv? 0.0 -0.0) is #t)
        assert_eval_sexpr("(eqv? 0.0 -0.0)", "#t", None);
        // Add tests for NaN if you support it (R7RS says (eqv? +nan.0 +nan.0) is implementation-dependent, often #f or errors)
        // For now, let's assume standard f64 NaN behavior where NaN != NaN
        // eval_str("(define nan (/ 0.0 0.0))", env.clone()).unwrap();
        // assert_eval_sexpr("(eqv? nan nan)", "#f", Some(env)); // Typical f64 behavior
    }

    #[test]
    fn test_eqv_bang_strings() {
        // eqv? on strings is like eq? according to R7RS (if they are eq? they are eqv?)
        // but if they are not eq? they are not eqv?
        assert_eval_sexpr("(eqv? \"foo\" \"foo\")", "#t", None); // Literals might be interned
        let env = new_test_env();
        eval_str("(define s1 \"abc\")", env.clone()).unwrap();
        eval_str("(define s2 \"abc\")", env.clone()).unwrap();
        assert_eval_sexpr("(eqv? s1 s2)", "#t", Some(env.clone())); // Assuming immutable strings, eq? -> eqv?
        // If strings are distinct mutable obj, then #f
    }

    #[test]
    fn test_eqv_bang_pairs_and_lists() {
        assert_eval_sexpr("(eqv? (cons 1 2) (cons 1 2))", "#f", None); // Distinct pairs
        assert_eval_sexpr("(eqv? (list 1 2) (list 1 2))", "#f", None); // Distinct lists
        let env = new_test_env();
        eval_str("(define p1 (cons 'a 'b))", env.clone()).unwrap();
        eval_str("(define p3 p1)", env.clone()).unwrap();
        assert_eval_sexpr("(eqv? p1 p3)", "#t", Some(env)); // Same object
    }

    #[test]
    fn test_eqv_bang_procedures() {
        // eqv? on procedures is like eq?
        let env = new_test_env();
        eval_str("(define f (lambda (x) x))", env.clone()).unwrap();
        eval_str("(define g (lambda (x) x))", env.clone()).unwrap();
        assert_eval_sexpr("(eqv? f g)", "#f", Some(env));
    }

    #[test]
    fn test_eqv_bang_arity() {
        let arity_error = &EvalError::InvalidArguments("".into(), Span::default()); // Dummy
        assert_eval_error("(eqv? 1)", &arity_error, None);
        assert_eval_error("(eqv? 1 2 3)", &arity_error, None);
    }

    // --- Tests for `equal?` ---
    #[test]
    fn test_equal_bang_symbols_numbers_booleans() {
        // Behaves like eqv? for these types
        assert_eval_sexpr("(equal? 'foo 'foo)", "#t", None);
        assert_eval_sexpr("(equal? 1 1.0)", "#t", None);
        assert_eval_sexpr("(equal? #t #t)", "#t", None);
        assert_eval_sexpr("(equal? '() '())", "#t", None);
    }

    #[test]
    fn test_equal_bang_strings() {
        assert_eval_sexpr("(equal? \"foo\" \"foo\")", "#t", None);
        assert_eval_sexpr("(equal? \"foo\" \"bar\")", "#f", None);
        let env = new_test_env();
        eval_str("(define s1 \"abc\")", env.clone()).unwrap();
        eval_str("(define s2 \"abc\")", env.clone()).unwrap(); // s1 and s2 may not be eq?
        assert_eval_sexpr("(equal? s1 s2)", "#t", Some(env)); // But they are equal?
    }

    #[test]
    fn test_equal_bang_pairs_and_lists() {
        assert_eval_sexpr("(equal? (cons 1 2) (cons 1 2))", "#t", None);
        assert_eval_sexpr("(equal? (list 1 2) (list 1 2))", "#t", None);
        assert_eval_sexpr("(equal? '(a (b c) d) '(a (b c) d))", "#t", None);
        assert_eval_sexpr("(equal? '(a b) '(a c))", "#f", None);
        assert_eval_sexpr("(equal? '(a b) '(a b c))", "#f", None);
        assert_eval_sexpr("(equal? '(a b . c) '(a b . c))", "#t", None);
        assert_eval_sexpr("(equal? '(a b . c) '(a b . d))", "#f", None);
    }

    #[test]
    fn test_equal_bang_nested_lists() {
        let env = new_test_env();
        eval_str("(define l1 '(1 (2 (3)) 4))", env.clone()).unwrap();
        eval_str("(define l2 (list 1 (list 2 (list 3)) 4))", env.clone()).unwrap();
        assert_eval_sexpr("(equal? l1 l2)", "#t", Some(env));
    }

    #[test]
    fn test_equal_bang_procedures() {
        // Behaves like eqv? for procedures (so, like eq?)
        let env = new_test_env();
        eval_str("(define f (lambda (x) x))", env.clone()).unwrap();
        eval_str("(define g (lambda (x) x))", env.clone()).unwrap();
        assert_eval_sexpr("(equal? f g)", "#f", Some(env));
    }

    #[test]
    fn test_equal_bang_arity() {
        let arity_error = &EvalError::InvalidArguments("".into(), Span::default()); // Dummy
        assert_eval_error("(equal? 1)", &arity_error, None);
        assert_eval_error("(equal? 1 2 3)", &arity_error, None);
    }
    */

    // Optional: Advanced tests for equal? with circular structures
    // These require your `equal?` implementation to handle cycles.
    // #[test]
    // fn test_equal_bang_circular_lists_identical() {
    //     let env = new_test_env();
    //     eval_str("(define c1 (list 'a))", env.clone()).unwrap();
    //     eval_str("(set-cdr! c1 c1)", env.clone()).unwrap(); // Requires set-cdr!
    //     eval_str("(define c2 (list 'a))", env.clone()).unwrap();
    //     eval_str("(set-cdr! c2 c2)", env.clone()).unwrap();
    //     // c1 is (a a a ...), c2 is (a a a ...)
    //     // (equal? c1 c2) should be #t if cycle detection is implemented.
    //     // This test will likely fail or loop infinitely without cycle detection.
    //     // For now, it's commented out.
    //     // assert_eval_sexpr("(equal? c1 c2)", "#t", Some(env));
    // }
}
