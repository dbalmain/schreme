use crate::environment::{EnvError, Environment};
use crate::source::Span;
use crate::types::{Node, Procedure, Sexpr};
use std::cell::{Ref, RefCell};
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
                Sexpr::Symbol(sym_name) if sym_name == "quote" => Ok(rest.clone()),

                // 3b. Special Form: 'if' (Implement next)
                Sexpr::Symbol(sym_name) if sym_name == "if" => {
                    evaluate_if(rest, env, node.span) // Pass env and whole form's span
                }

                // 3c. Special Form: 'define' (Implement later)
                // Sexpr::Symbol(ref sym_name) if sym_name == "define" => { ... }

                // 3d. Special Form: 'set!' (Implement later)
                // Sexpr::Symbol(ref sym_name) if sym_name == "set!" => { ... }

                // 3e. Special Form: 'lambda' (Implement later)
                // Sexpr::Symbol(ref sym_name) if sym_name == "lambda" => { ... }

                // 3f. Procedure Call (Implement later)
                _ => evaluate_procedure(first, rest, env, node.span),
            }
        } // Handle other Sexpr kinds if they exist (e.g., Pair, Vector) later
    }
}

fn invalid_special_form(error: &str, span: Span) -> EvalResult {
    Err(EvalError::InvalidSpecialForm(error.to_string(), span))
}

/// Evaluates the 'if' special form.
/// `if_operands_node` is the Node whose kind is Pair(condition, Pair(consequent, Pair(alternate, Nil)))
/// or Pair(condition, Pair(consequent, Nil)).
/// `env` is the current environment.
/// `original_if_span` is the span of the entire (if ...) expression.
fn evaluate_if(
    operands: &Node,
    env: Rc<RefCell<Environment>>,
    original_if_span: Span,
) -> EvalResult {
    // The operands are structured as a list (nested pairs)
    // (if condition consequent alternate?)
    // This means if_operands_node should be Pair(condition, rest_of_args_list)

    // 1. Extract the condition
    let (condition, consequent_and_maybe_alternate) = match operands.kind.borrow().clone() {
        // Clone the Sexpr to avoid holding borrow
        Sexpr::Pair(cond_node, rest_args_node) => (cond_node, rest_args_node),
        Sexpr::Nil => {
            return invalid_special_form("if: Missing condition and consequent", original_if_span);
        }
        _ => {
            // Should not happen if parser creates proper list structure for operands
            return invalid_special_form("if: Missing consequent", original_if_span);
        }
    };

    // 2. Evaluate the condition
    let condition_result = evaluate(condition.clone(), env.clone())?;

    // Determine truthiness
    let is_truthy = match &*condition_result.kind.borrow() {
        // Borrow to check the kind
        Sexpr::Boolean(false) => false,
        _ => true,
    };

    if is_truthy {
        // 3a. Condition is true: extract and evaluate the consequent
        match consequent_and_maybe_alternate.kind.borrow().clone() {
            // Clone Sexpr
            Sexpr::Pair(consequent_node, _) => {
                // We only care about the consequent here
                evaluate(consequent_node.clone(), env)
            }
            Sexpr::Nil => {
                // (if #t) - missing consequent
                Err(EvalError::InvalidSpecialForm(
                    "if: Missing consequent branch".to_string(),
                    original_if_span,
                ))
            }
            _ => Err(EvalError::InvalidSpecialForm(
                // Should be Pair or Nil
                "if: Invalid structure for consequent branch".to_string(),
                consequent_and_maybe_alternate.span,
            )),
        }
    } else {
        // 3b. Condition is false: extract and evaluate the alternate, or return Nil
        match consequent_and_maybe_alternate.kind.borrow().clone() {
            // Clone Sexpr
            Sexpr::Pair(_, alternate_branch_node) => {
                // Skip consequent, look at its cdr
                match alternate_branch_node.kind.borrow().clone() {
                    // Clone Sexpr
                    Sexpr::Pair(actual_alternate_node, term_node) => {
                        // Ensure the list of operands for if ends correctly: (if c t e) -> (c . (t . (e . ())))
                        // term_node should be Nil
                        if !matches!(*term_node.kind.borrow(), Sexpr::Nil) {
                            return Err(EvalError::InvalidSpecialForm(
                                "if: Too many arguments, alternate branch not properly terminated"
                                    .to_string(),
                                term_node.span,
                            ));
                        }
                        evaluate(actual_alternate_node.clone(), env)
                    }
                    Sexpr::Nil => {
                        // This is the case for (if #f consequent) - no alternate provided
                        Ok(Node {
                            kind: Rc::new(RefCell::new(Sexpr::Nil)),
                            span: original_if_span, // Span for the default return
                        })
                    }
                    _ => Err(EvalError::InvalidSpecialForm(
                        // Should be Pair or Nil
                        "if: Invalid structure for alternate branch".to_string(),
                        alternate_branch_node.span,
                    )),
                }
            }
            Sexpr::Nil => {
                // (if #f) - missing consequent and alternate
                Err(EvalError::InvalidSpecialForm(
                    "if: Missing consequent branch (condition was false)".to_string(),
                    original_if_span,
                ))
            }
            _ => Err(EvalError::InvalidSpecialForm(
                // Should be Pair or Nil
                "if: Invalid structure for branches".to_string(),
                consequent_and_maybe_alternate.span,
            )),
        }
    }
}

fn evaluate_procedure(
    operator_node_from_ast: Node, // The Node from the AST that represents the operator
    operands_list_node: Node, // The Node from the AST representing the list of operands, e.g. kind is Pair(arg1, Pair(arg2, Nil))
    env: Rc<RefCell<Environment>>,
    original_call_span: Span, // Span of the entire procedure call expression, e.g., ((+ 1) 2)
) -> EvalResult {
    // 1. Evaluate the operator expression to get a procedure value
    let evaluated_operator_node = evaluate(operator_node_from_ast.clone(), env.clone())?; // env.clone() is cheap (Rc)

    // 2. Check if the result of evaluating the operator is actually a procedure
    //    We need to borrow the Sexpr from the Rc<RefCell<Sexpr>>
    let procedure_value = match &*evaluated_operator_node.kind.borrow() {
        Sexpr::Procedure(proc_enum_val) => proc_enum_val.clone(), // Clone the Procedure enum (Primitive or Lambda)
        _ => {
            return Err(EvalError::NotAProcedure(
                evaluated_operator_node.kind.borrow().clone(), // The Sexpr that was found
                operator_node_from_ast.span, // Span of the original operator expression from AST
            ));
        }
    };

    // 3. Evaluate the operands (which are in a list structure)
    let mut evaluated_args: Vec<Node> = Vec::new();
    let mut current_operand_list_node: Node = operands_list_node; // Start with the AST node for the whole operand list

    loop {
        // Borrow the Sexpr from the current node in the operand list structure
        let kind_of_current_operand_list: Ref<Sexpr> = current_operand_list_node.kind.borrow();

        match &*kind_of_current_operand_list {
            Sexpr::Nil => {
                // End of the operand list
                break;
            }
            Sexpr::Pair(actual_operand_node, next_operand_list_node) => {
                // Evaluate the current actual_operand_node
                let evaluated_arg = evaluate(actual_operand_node.clone(), env.clone())?;
                evaluated_args.push(evaluated_arg);

                // Move to the next part of the operand list for the next iteration
                let temp_next_list_node = next_operand_list_node.clone();
                drop(kind_of_current_operand_list); // Release borrow before reassigning loop variable
                current_operand_list_node = temp_next_list_node;
            }
            _ => {
                // This case means the operand list wasn't a proper list structure
                // (e.g., ended with a non-nil atom in a dotted pair like ((foo 1 . 2)))
                // This is typically a syntax error caught by the parser, but good to handle defensively.
                // Or, if your language allows applying functions to improper lists, this might be valid
                // but usually function application expects a proper list of arguments.
                return Err(EvalError::InvalidSpecialForm(
                    // Or a more specific "Invalid argument list structure"
                    "Procedure call operands did not form a proper list.".to_string(),
                    current_operand_list_node.span, // Span of the problematic part of the arg list
                ));
            }
        }
    }

    // 4. Apply the procedure
    match procedure_value {
        Procedure::Primitive(func, _) => {
            // Call the Rust function with evaluated args and the original call's span
            func(evaluated_args, original_call_span)
        } // Procedure::Lambda(lambda_data) => {
          //     // - Create a new environment enclosing the lambda's captured env
          //     // - Bind lambda parameters to evaluated_args in the new env
          //     // - Evaluate the lambda body in the new env
          //     // - Handle TCO here eventually
          //     unimplemented!("Lambda evaluation not yet implemented");
          // }
    }
}
fn old_evaluate_procedure(
    operator: &Node,
    operands: &Node,
    env: Rc<RefCell<Environment>>,
    span: Span,
) -> EvalResult {
    let operator_result_node = evaluate(operator.clone(), env.clone()).map_err(|_err| {
        EvalError::NotAProcedure(
            operator.kind.borrow().clone(), // What was actually found
            operator.span,                  // Span of the operator expression
        )
    })?;

    let operator_sexpr = &*operator_result_node.kind.borrow();
    let procedure = match operator_sexpr {
        Sexpr::Procedure(proc) => proc, // Extract the Procedure enum
        _ => {
            return Err(EvalError::NotAProcedure(
                operator_sexpr.clone(), // What was actually found
                operator.span,          // Span of the operator expression
            ));
        }
    };

    // From here on, we need to get this working
    // 3. Evaluate the operands
    let mut evaluated_args: Vec<Node> = Vec::with_capacity(operands.len());
    for operand_node in operands {
        // Evaluate each operand in the current environment
        evaluated_args.push(evaluate(operand_node.clone(), env.clone())?);
    }

    // 4. Apply the procedure
    match procedure {
        Procedure::Primitive(func, _) => {
            // Call the Rust function with evaluated args and original call span
            func(evaluated_args, span)
        } // Procedure::Lambda(lambda_data) => {
          //     // - Create a new environment enclosing the lambda's captured env
          //     // - Bind lambda parameters to evaluated_args in the new env
          //     // - Evaluate the lambda body in the new env
          //     // - Handle TCO here eventually
          //     unimplemented!("Lambda evaluation not yet implemented");
          // }
    }
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
    use crate::parser::parse_str; // Use parser to create AST nodes easily
    use crate::source::Span;

    // Helper to evaluate input string and check result kind (ignores span)
    fn assert_eval_kind(input: &str, expected_kind: Sexpr, env: Option<Rc<RefCell<Environment>>>) {
        let env = env.unwrap_or_else(Environment::new_global_populated); // Use provided env or create new global one
        match parse_str(input) {
            Ok(node) => match evaluate(node, env) {
                Ok(result_node) => {
                    assert_eq!(result_node.kind, expected_kind, "Input: '{}'", input)
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
                    // Optional: Add checks here for the span within the error `e` if needed
                }
            },
            Err(e) => panic!("Parsing failed for input '{}': {}", input, e),
        }
    }

    #[test]
    fn test_eval_self_evaluating() {
        assert_eval_kind("123", Sexpr::Number(123.0), None);
        assert_eval_kind("-4.5", Sexpr::Number(-4.5), None);
        assert_eval_kind("#t", Sexpr::Boolean(true), None);
        assert_eval_kind("#f", Sexpr::Boolean(false), None);
        assert_eval_kind(r#""hello""#, Sexpr::String("hello".to_string()), None);
        assert_eval_kind("()", Sexpr::Nil, None); // Evaluating '()' -> Nil
    }

    #[test]
    fn test_eval_symbol_lookup_ok() {
        let env = Environment::new();
        env.borrow_mut().define(
            "x".to_string(),
            Node {
                kind: Sexpr::Number(100.0),
                span: Span::default(),
            },
        );
        assert_eval_kind("x", Sexpr::Number(100.0), Some(env));
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
        assert_eval_kind("'1", Sexpr::Number(1.0), None); // '(quote 1) -> 1
        assert_eval_kind("'a", Sexpr::Symbol("a".to_string()), None); // '(quote a) -> a
        assert_eval_kind("'#t", Sexpr::Boolean(true), None); // '(quote #t) -> #t
        assert_eval_kind("'()", Sexpr::Nil, None); // '(quote ()) -> ()
        assert_eval_kind("(quote ())", Sexpr::Nil, None); // '(quote ()) -> ()

        // '(quote (1 2)) -> (1 2)
        // Need a way to represent the expected list structure (Nodes inside Sexpr::List)
        // Let's check the structure manually for now:
        let env = Environment::new();
        match parse_str("'(1 2)") {
            Ok(node) => match evaluate(node, env) {
                Ok(result_node) => {
                    if let Sexpr::Pair(elements) = result_node.kind {
                        assert_eq!(elements.len(), 2);
                        assert!(matches!(elements[0].kind, Sexpr::Number(n) if n == 1.0));
                        assert!(matches!(elements[1].kind, Sexpr::Number(n) if n == 2.0));
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
        assert_eval_kind("(if #t 1 2)", Sexpr::Number(1.0), None);
        assert_eval_kind("(if #t 1)", Sexpr::Number(1.0), None); // No alternate
        assert_eval_kind("(if 0 1 2)", Sexpr::Number(1.0), None); // 0 is true
        assert_eval_kind("(if '() 1 2)", Sexpr::Number(1.0), None); // '() is true
        assert_eval_kind("(if \"hello\" 1 2)", Sexpr::Number(1.0), None); // String is true
        assert_eval_kind("(if (quote x) 1 2)", Sexpr::Number(1.0), None); // Symbol is true
    }

    #[test]
    fn test_eval_if_false() {
        assert_eval_kind("(if #f 1 2)", Sexpr::Number(2.0), None);
        assert_eval_kind("(if #f 1)", Sexpr::Nil, None); // No alternate, returns Nil (our choice)
    }

    #[test]
    fn test_eval_if_nested() {
        let env = Environment::new();
        env.borrow_mut().define(
            "x".to_string(),
            Node {
                kind: Sexpr::Boolean(true),
                span: Span::default(),
            },
        );
        env.borrow_mut().define(
            "y".to_string(),
            Node {
                kind: Sexpr::Boolean(false),
                span: Span::default(),
            },
        );

        assert_eval_kind("(if x 1 (if y 2 3))", Sexpr::Number(1.0), Some(env.clone()));
        assert_eval_kind("(if y 1 (if x 2 3))", Sexpr::Number(2.0), Some(env.clone()));
    }

    #[test]
    fn test_eval_if_evaluates_condition() {
        // Ensure the condition is actually evaluated
        let env = Environment::new();
        env.borrow_mut().define(
            "cond".to_string(),
            Node {
                kind: Sexpr::Boolean(false),
                span: Span::default(),
            },
        );
        assert_eval_kind("(if cond 1 2)", Sexpr::Number(2.0), Some(env));
    }

    #[test]
    fn test_eval_if_does_not_evaluate_unused_branch() {
        // This is harder to test directly without side effects (like define/set! or print)
        // We can test it indirectly by putting an unbound variable in the unused branch.
        let env = Environment::new();
        // (if #t 'good unbound-variable) should evaluate to 'good without error
        assert_eval_kind(
            "(if #t 'good unbound-variable)",
            Sexpr::Symbol("good".to_string()),
            Some(env.clone()),
        );
        // (if #f unbound-variable 'good) should evaluate to 'good without error
        assert_eval_kind(
            "(if #f unbound-variable 'good)",
            Sexpr::Symbol("good".to_string()),
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
        assert_eval_kind("(+ 1 2)", Sexpr::Number(3.0), None);
        assert_eval_kind("(+ 10 20 30 40)", Sexpr::Number(100.0), None);
        assert_eval_kind("(+)", Sexpr::Number(0.0), None); // Add identity
        assert_eval_kind("(- 10 3)", Sexpr::Number(7.0), None);
        assert_eval_kind("(- 5)", Sexpr::Number(-5.0), None);
        assert_eval_kind("(- 10 3 2)", Sexpr::Number(5.0), None);
        assert_eval_kind("(* 2 3)", Sexpr::Number(6.0), None);
        assert_eval_kind("(* 2 3 4)", Sexpr::Number(24.0), None);
        assert_eval_kind("(*)", Sexpr::Number(1.0), None); // Multiply identity
        assert_eval_kind("(/ 10 2)", Sexpr::Number(5.0), None);
        assert_eval_kind("(/ 10 4)", Sexpr::Number(2.5), None);
        assert_eval_kind("(/ 20 2 5)", Sexpr::Number(2.0), None);
        assert_eval_kind("(/ 5)", Sexpr::Number(0.2), None); // 1/5
    }

    #[test]
    fn test_eval_primitives_comparison() {
        assert_eval_kind("(= 5 5)", Sexpr::Boolean(true), None);
        assert_eval_kind("(= 5 5 5 5)", Sexpr::Boolean(true), None);
        assert_eval_kind("(= 5 6)", Sexpr::Boolean(false), None);
        assert_eval_kind("(= 5 5 6)", Sexpr::Boolean(false), None);
        assert_eval_kind("(< 4 5 6)", Sexpr::Boolean(true), None);
        assert_eval_kind("(< 5 5 6)", Sexpr::Boolean(false), None);
        assert_eval_kind("(<= 5 5 6)", Sexpr::Boolean(true), None);
        assert_eval_kind("(>= 5 5 5)", Sexpr::Boolean(true), None);
        assert_eval_kind("(>= 6 5 5)", Sexpr::Boolean(true), None);
        assert_eval_kind("(> 6 5 5)", Sexpr::Boolean(false), None);
        assert_eval_kind("(> 6 5 4)", Sexpr::Boolean(true), None);
        assert_eval_kind("(< 1 2 3 4 5 6)", Sexpr::Boolean(true), None);
        assert_eval_kind("(>= 5 5 4 4 4 3)", Sexpr::Boolean(true), None);
    }

    #[test]
    fn test_eval_primitives_nested_calls() {
        assert_eval_kind("(+ 1 (* 2 3))", Sexpr::Number(7.0), None);
        assert_eval_kind("(- (+ 5 5) (* 2 3))", Sexpr::Number(4.0), None);
    }

    #[test]
    fn test_eval_primitives_arity_errors() {
        let arity_error = &EvalError::InvalidArguments("".into(), Span::default()); // Dummy
        // assert_eval_error("(-)", arity_error); // Handled by check_arity!
        assert_eval_error("(/)", arity_error, None);
        assert_eval_error("(=)", arity_error, None);
        assert_eval_error("(= 1)", arity_error, None); // Assuming we enforce min 2 args for =
    }

    #[test]
    fn test_eval_primitives_type_errors() {
        let type_error = &EvalError::InvalidArguments("".into(), Span::default()); // Dummy
        assert_eval_error("(+ 1 #t)", type_error, None);
        assert_eval_error("(/ 1 \"hello\")", type_error, None);
        assert_eval_error("(= 1 #f)", type_error, None); // Mismatched types in comparison
    }

    fn list(nodes: &[Node], start: usize, end: usize) -> Node {
        match nodes {
            [] => Node::new(Sexpr::Nil, Span::new(start, end)),
            [first, rest @ ..] => Node::new(
                Sexpr::Pair(Rc::new(first.clone()), Rc::new(list(rest, start, end))),
                Span::new(start, end),
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

    fn node(kind: Sexpr, start: usize, end: usize) -> Node {
        Node::new(kind, Span::new(start, end))
    }

    #[test]
    fn test_eval_list_primitives() {
        // list
        assert_eval_kind("(list)", Sexpr::Nil, None);
        assert_eval_kind(
            "(list 1 2 3)",
            Sexpr::List(vec![
                node(Sexpr::Number(1.0), 6, 7),
                node(Sexpr::Number(2.0), 8, 9),
                node(Sexpr::Number(3.0), 10, 11),
            ]),
            None,
        );
        assert_eval_kind(
            "(list (+ 1 1) (- 5 2))",
            Sexpr::List(vec![
                node(Sexpr::Number(2.0), 6, 13),
                node(Sexpr::Number(3.0), 14, 21),
            ]),
            None,
        );

        // cons
        assert_eval_kind(
            "(cons 1 '())",
            Sexpr::List(vec![node(Sexpr::Number(1.0), 6, 7)]),
            None,
        );
        assert_eval_kind(
            "(cons 1 (list 2 3))",
            Sexpr::List(vec![
                node(Sexpr::Number(1.0), 6, 7),
                node(Sexpr::Number(2.0), 14, 15),
                node(Sexpr::Number(3.0), 16, 17),
            ]),
            None,
        );
        assert_eval_kind(
            "(cons (list 1) (list 2))",
            Sexpr::List(vec![
                node(Sexpr::List(vec![node(Sexpr::Number(1.0), 12, 13)]), 6, 14),
                node(Sexpr::Number(2.0), 21, 22),
            ]),
            None,
        );

        // car
        assert_eval_kind("(car (list 1 2 3))", Sexpr::Number(1.0), None);
        assert_eval_kind("(car (cons 'a '()))", Sexpr::Symbol("a".to_string()), None);

        // cdr
        assert_eval_kind(
            "(cdr (list 1 2 3))",
            Sexpr::List(vec![
                node(Sexpr::Number(2.0), 13, 14),
                node(Sexpr::Number(3.0), 15, 16),
            ]),
            None,
        );
        assert_eval_kind("(cdr (list 1))", Sexpr::Nil, None); // cdr of single-element list is Nil
        assert_eval_kind(
            "(cdr (cons 1 (cons 2 '())))",
            Sexpr::List(vec![node(Sexpr::Number(2.0), 19, 20)]),
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
        assert_eval_error("(cons 1 2)", type_error, None); // 2nd arg must be list/nil

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
        assert_eval_kind("(null? '())", Sexpr::Boolean(true), None);
        assert_eval_kind("(null? (list))", Sexpr::Boolean(true), None);
        assert_eval_kind("(null? (list 1))", Sexpr::Boolean(false), None);
        assert_eval_kind("(null? 1)", Sexpr::Boolean(false), None);

        assert_eval_kind("(pair? (cons 1 (list 2)))", Sexpr::Boolean(true), None); // Assuming cons returns list for now
        assert_eval_kind("(pair? (list 1))", Sexpr::Boolean(true), None);
        //assert_eval_kind("(pair? '(1 . 2))", Sexpr::Boolean(true), None); // Need dotted pair support first
        assert_eval_kind("(pair? '())", Sexpr::Boolean(false), None);
        assert_eval_kind("(pair? 1)", Sexpr::Boolean(false), None);

        assert_eval_kind("(number? 1)", Sexpr::Boolean(true), None);
        assert_eval_kind("(number? #f)", Sexpr::Boolean(false), None);
        assert_eval_kind("(boolean? #t)", Sexpr::Boolean(true), None);
        assert_eval_kind("(boolean? 0)", Sexpr::Boolean(false), None);
        assert_eval_kind("(symbol? 'a)", Sexpr::Boolean(true), None);
        assert_eval_kind("(symbol? \"a\")", Sexpr::Boolean(false), None);
        assert_eval_kind("(string? \"a\")", Sexpr::Boolean(true), None);
        assert_eval_kind("(string? 'a)", Sexpr::Boolean(false), None);
        assert_eval_kind("(procedure? +)", Sexpr::Boolean(true), None);
        assert_eval_kind("(procedure? 1)", Sexpr::Boolean(false), None);
        // Add tests for lambda later: (procedure? (lambda (x) x)) -> true
    }
}
