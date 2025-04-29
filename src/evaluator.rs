use crate::environment::{EnvError, Environment};
use crate::source::Span;
use crate::types::{Node, Sexpr};
use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;

// --- Evaluation Error ---
#[derive(Debug, Clone, PartialEq)]
pub enum EvalError {
    EnvError(EnvError),             // Errors from environment lookup
    NotAProcedure(Sexpr, Span),     // Tried to call something that isn't a procedure
    InvalidArguments(String, Span), // Mismatched arity or wrong type of args
    NotAList(Sexpr, Span),          // Expected a list for procedure call or special form
    NotASymbol(Sexpr, Span),        // Expected a symbol (e.g., for define/set!)
    InvalidSpecialForm(String, Span), // Malformed special form (e.g., (if cond))
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
            EvalError::NotAList(sexpr, _span) => write!(
                f,
                "Evaluation Error: Expected a list structure, but got: {}",
                sexpr
            ),
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
type EvalResult<T = Node> = Result<T, EvalError>;

// --- Evaluate Function ---

/// Evaluates a given AST Node within the specified environment.
pub fn evaluate(node: Node, env: Rc<RefCell<Environment>>) -> EvalResult {
    // Use a loop for Tail Call Optimization (TCO) later if needed (trampoline)
    // For now, simple recursive calls are fine.
    // let current_node = node;
    // let current_env = env;
    // loop { ... }

    match node.kind {
        // 1. Self-evaluating atoms: Numbers, Strings, Booleans, Nil
        Sexpr::Number(_) | Sexpr::String(_) | Sexpr::Boolean(_) | Sexpr::Nil => {
            Ok(node) // Return the node itself (or a clone if ownership is an issue)
        }

        // 2. Symbols: Look up in the environment
        Sexpr::Symbol(ref name) => {
            // Use the symbol's span for error reporting if lookup fails
            Ok(env.borrow().get(name, node.span)?) // Propagate EnvError via From trait
        }

        // 3. Lists: Could be special forms or procedure calls
        Sexpr::List(ref elements) => {
            if elements.is_empty() {
                // Evaluating an empty list '()' which is Sexpr::Nil
                // Nil evaluates to itself, so we return the original Node
                return Ok(node); // Or create a new Node { kind: Sexpr::Nil, span: node.span }
            }

            // Check the first element to see if it's a special form or procedure call
            let first = &elements[0];
            let rest = &elements[1..];

            match first.kind {
                // 3a. Special Form: 'quote'
                Sexpr::Symbol(ref sym_name) if sym_name == "quote" => {
                    evaluate_quote(rest, node.span) // Pass span of the whole (quote ...) form
                }

                // 3b. Special Form: 'if' (Implement next)
                // Sexpr::Symbol(ref sym_name) if sym_name == "if" => { ... }

                // 3c. Special Form: 'define' (Implement later)
                // Sexpr::Symbol(ref sym_name) if sym_name == "define" => { ... }

                // 3d. Special Form: 'set!' (Implement later)
                // Sexpr::Symbol(ref sym_name) if sym_name == "set!" => { ... }

                // 3e. Special Form: 'lambda' (Implement later)
                // Sexpr::Symbol(ref sym_name) if sym_name == "lambda" => { ... }

                // 3f. Procedure Call (Implement later)
                _ => {
                    // Evaluate the operator and operands, then apply
                    // evaluate_procedure_call(first, rest, env, node.span)
                    // For now, error - we haven't implemented calls yet
                    Err(EvalError::NotAProcedure(first.kind.clone(), first.span)) // Placeholder error
                }
            }
        } // Handle other Sexpr kinds if they exist (e.g., Pair, Vector) later
    }
}

// --- Helper for 'quote' ---
fn evaluate_quote(operands: &[Node], span: Span) -> EvalResult {
    // (quote operand)
    if operands.len() != 1 {
        return Err(EvalError::InvalidSpecialForm(
            "quote expects exactly one argument".to_string(),
            span, // Use the span of the whole (quote ...) form
        ));
    }
    // Quote returns the operand unevaluated.
    // The operand is already a Node, just return it (or a clone).
    Ok(operands[0].clone())
}

// --- Other helpers (if, define, set!, lambda, procedure calls) will go here ---

// --- Unit Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_str; // Use parser to create AST nodes easily
    use crate::source::Span;

    // Helper to evaluate input string and check result kind (ignores span)
    fn assert_eval_kind(input: &str, expected_kind: Sexpr, env: Option<Rc<RefCell<Environment>>>) {
        let env = env.unwrap_or_else(Environment::new); // Use provided env or create new global one
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
        let env = env.unwrap_or_else(Environment::new);
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
                    assert!(matches!(result_node.kind, Sexpr::List(_)));
                    if let Sexpr::List(elements) = result_node.kind {
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
    fn test_eval_list_placeholder_error() {
        // Evaluating a list where the first element is not a known special form
        // or procedure (which we haven't implemented yet) should error.
        let env = Environment::new();
        // (1 2 3) -> Error because 1 is not a procedure/special form
        let not_proc_error = EvalError::NotAProcedure(Sexpr::Number(0.0), Span::default()); // Dummy
        assert_eval_error("(1 2 3)", &not_proc_error, Some(env.clone()));

        // ("hello" 1) -> Error
        let not_proc_error_str =
            EvalError::NotAProcedure(Sexpr::String("".into()), Span::default()); // Dummy
        assert_eval_error("(\"hello\" 1)", &not_proc_error_str, Some(env.clone()));
    }
}
