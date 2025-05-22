use crate::environment::{EnvError, Environment};
use crate::source::Span;
use crate::types::{EvaluatedNodeIterator, Lambda, Node, Procedure, Sexpr};
use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;

// --- Evaluation Error ---
#[derive(Debug, Clone, PartialEq)]
pub enum EvalError {
    EnvError(EnvError),               // Errors from environment lookup
    NotAProcedure(Sexpr, Span),       // Tried to call something that isn't a procedure
    InvalidArguments(String, Span),   // Mismatched arity or wrong type of args
    NotASymbol(Sexpr, Span),          // Expected a symbol (e.g., for define/set!)
    InvalidSpecialForm(String, Span), // Malformed special form (e.g., (if cond))
    UnexpectedError(Sexpr, Span, String), // Expected a list for procedure call or special form
                                      // Add more later: DivideByZero, WrongType, MacroError, etc.
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
                // 3a. Special Form: 'quote'
                Sexpr::Symbol(sym_name) if sym_name == "quote" => {
                    evaluate_quote(rest.clone(), node.span)
                }

                // 3b. Special Form: 'if' (Implement next)
                Sexpr::Symbol(sym_name) if sym_name == "if" => {
                    evaluate_if(rest.clone(), env, node.span)
                }

                // 3c. Special Form: 'define' (Implement later)
                Sexpr::Symbol(sym_name) if sym_name == "define" => {
                    evaluate_define(rest.clone(), env, node.span)
                }

                // 3d. Special Form: 'set!' (Implement later)
                // Sexpr::Symbol(ref sym_name) if sym_name == "set!" => { ... }

                // 3e. Special Form: 'lambda' (Implement later)
                Sexpr::Symbol(sym_name) if sym_name == "lambda" => {
                    evaluate_lambda(rest.clone(), env, node.span)
                }
                // Sexpr::Symbol(ref sym_name) if sym_name == "lambda" => { ... }

                // 3f. Procedure Call (Implement later)
                _ => evaluate_procedure(first.clone(), rest.clone(), env, node.span),
            }
        } // Handle other Sexpr kinds if they exist (e.g., Pair, Vector) later
    }
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
    operands: Node,
    env: Rc<RefCell<Environment>>,
    original_if_span: Span,
) -> EvalResult {
    let operands: Vec<Node> = operands.into_iter().collect();

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

fn evaluate_quote(operands: Node, original_quote_span: Span) -> EvalResult {
    let operands: Vec<Node> = operands.into_iter().collect();

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

fn evaluate_define(
    operands: Node,
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
                        Node::new_pair(args.clone(), definition.clone(), span),
                        env.clone(),
                        original_span,
                    )?;
                    env.borrow_mut().define(name.clone(), lambda.clone());
                    Ok(lambda)
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

fn evaluate_lambda(
    operands: Node,
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
                                    return Err(EvalError::InvalidArguments(
                                        "lambda parameters must be symbols".to_string(),
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
    operator_node: Node,      // The Node from the AST that represents the operator
    operands_list_node: Node, // The Node from the AST representing the list of operands, e.g. kind is Pair(arg1, Pair(arg2, Nil))
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
            func(operands_list_node.into_eval_iter(env), original_call_span)
        }
        Procedure::Lambda(lambda) => apply_lambda(
            lambda,
            operands_list_node.into_eval_iter(env.clone()),
            original_call_span,
        ),
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
    let mut result: EvalResult = Ok(Node::new_nil(Span::default()));
    for expr in lambda.body.iter() {
        if let Ok(node) = result {
            env.borrow_mut().define("_".to_string(), node);
        }
        result = evaluate(expr.clone(), env.clone());
    }
    result
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

// Add more primitives: cons, car, cdr, list, null?, pair?, etc.
// --- Unit Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    use crate::ParseError;
    use crate::parser::parse_str; // Use parser to create AST nodes easily
    use crate::source::Span;

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

    fn new_test_env() -> Rc<RefCell<Environment>> {
        Environment::new_global_populated() // Or however you create your top-level env
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

    fn assert_eval_define(input: &str, env: &Rc<RefCell<Environment>>, expected_result: &Sexpr) {
        assert_eval_kind(input, expected_result, Some(env.clone()));
    }

    #[test]
    fn test_define_number() {
        let env = new_test_env();
        let x = &Sexpr::Number(10.0);
        assert_eval_define("(define x 10)", &env, x);
        assert_env_has(&env, "x", x);
    }

    #[test]
    fn test_define_boolean() {
        let env = new_test_env();
        let x = &Sexpr::Boolean(true);
        let y = &Sexpr::Boolean(false);
        assert_eval_define("(define x #t)", &env, x);
        assert_eval_define("(define y #f)", &env, y);
        assert_env_has(&env, "x", x);
        assert_env_has(&env, "y", y);
    }

    #[test]
    fn test_define_quoted_symbol() {
        let env = new_test_env();
        let s = &Sexpr::Symbol("some-symbol".to_string());
        assert_eval_define("(define s 'some-symbol)", &env, s);
        assert_env_has(&env, "s", s);
    }

    #[test]
    fn test_define_expression() {
        let env = new_test_env();
        let result = &Sexpr::Number(10.0);
        assert_eval_define("(define result (+ 5 (/ 15 3)))", &env, result);
        assert_env_has(&env, "result", result);
    }

    #[test]
    fn test_define_uses_previously_defined_variable() {
        let env = new_test_env();
        let five = &Sexpr::Number(5.0);
        let eight = &Sexpr::Number(8.0);
        assert_eval_define("(define a 5)", &env, five);
        assert_eval_define("(define b (+ a 3))", &env, eight);
        assert_env_has(&env, "a", five);
        assert_env_has(&env, "b", eight);
    }

    #[test]
    fn test_redefine_variable() {
        let env = new_test_env();
        let c = &Sexpr::Number(100.0);
        let cc = &Sexpr::Number(200.0);
        assert_eval_define("(define val 100)", &env, c);
        assert_env_has(&env, "val", c);
        assert_eval_define("(define val 200)", &env, cc);
        assert_env_has(&env, "val", cc);
    }

    #[test]
    fn test_define_nil_by_default() {
        let env = new_test_env();
        let nil = &Sexpr::Nil;
        assert_eval_define("(define x)", &env, nil);
        assert_env_has(&env, "x", nil);
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
        assert!(matches!(result, Err(EvalError::InvalidArguments(_, _))));
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
            matches!(result, Err(EvalError::InvalidArguments(_, _))),
            "Syntax error: variadic parameter list specified as a quoted list"
        );
    }

    #[test]
    fn test_define_lambda_sugar_no_args() {
        let env = new_test_env();
        eval_str("(define (my-func) 3)", env.clone()).unwrap();
        assert_eval_kind("(my-func)", &Sexpr::Number(3.0), Some(env));
    }

    #[test]
    fn test_define_lambda_sugar_one_arg() {
        let env = new_test_env();
        eval_str("(define (add3 x) (+ 3 x))", env.clone()).unwrap();
        assert_eval_kind("(add3 3)", &Sexpr::Number(6.0), Some(env));
    }

    #[test]
    fn test_define_lambda_sugar_multiple_args() {
        let env = new_test_env();
        eval_str("(define (add-3 a b c) (+ a b c))", env.clone()).unwrap();
        assert_eval_kind("(add-3 1 2 3)", &Sexpr::Number(6.0), Some(env));
    }

    #[test]
    fn test_define_lambda_sugar_one_variadic_arg() {
        let env = new_test_env();
        eval_str("(define (collect . x) x)", env.clone()).unwrap();
        assert_eval_sexpr("(collect 'a 'b 'c)", "(a b c)", Some(env));
    }
}
