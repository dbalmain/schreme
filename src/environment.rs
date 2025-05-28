use logos::Source;

use crate::source::Span;
use crate::types::{Node, PrimitiveFunc};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::rc::Rc;

// --- Environment Error ---
// This will likely become part of a larger EvalError later
#[derive(Debug, Clone, PartialEq)]
pub enum EnvError {
    UnboundVariable(String, Span), // Symbol name, span where lookup happened
}

impl fmt::Display for EnvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            // TODO: Improve error display using span and source code later
            EnvError::UnboundVariable(name, _span) => {
                write!(f, "Unbound variable: '{}'", name)
            }
        }
    }
}

impl std::error::Error for EnvError {}

// --- Environment Definition ---

#[derive(Debug, Clone, PartialEq)]
pub struct Environment {
    // Use Rc<RefCell<...>> to allow shared ownership and interior mutability.
    // Needed for closures capturing environments and for 'set!'.
    outer: Option<Rc<RefCell<Environment>>>,
    bindings: HashMap<String, Node>, // Maps variable names to Nodes
}

impl Environment {
    /// Creates a new, top-level (global) environment.
    pub fn new() -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Environment {
            outer: None,
            bindings: HashMap::new(),
        }))
    }
    pub fn new_global_populated() -> Rc<RefCell<Environment>> {
        let env_ptr = Environment::new(); // Create empty global env
        {
            // Borrow mutably only inside this scope
            let mut env = env_ptr.borrow_mut();
            // Add primitives
            env.add_primitive("+", crate::primitives::prim_add);
            env.add_primitive("-", crate::primitives::prim_sub);
            env.add_primitive("*", crate::primitives::prim_mul);
            env.add_primitive("/", crate::primitives::prim_div);
            env.add_primitive("=", crate::primitives::prim_equals);
            env.add_primitive("<", crate::primitives::prim_less_than);
            env.add_primitive("<=", crate::primitives::prim_less_than_or_equals);
            env.add_primitive(">", crate::primitives::prim_greater_than);
            env.add_primitive(">=", crate::primitives::prim_greater_than_or_equals);

            // --- ADD List Primitives ---
            env.add_primitive("cons", crate::primitives::prim_cons);
            env.add_primitive("car", crate::primitives::prim_car);
            env.add_primitive("cdr", crate::primitives::prim_cdr);
            env.add_primitive("list", crate::primitives::prim_list);

            // --- ADD Type Predicates ---
            env.add_primitive("null?", crate::primitives::prim_is_null);
            env.add_primitive("pair?", crate::primitives::prim_is_pair); // Or list? depending on exact semantics chosen
            env.add_primitive("number?", crate::primitives::prim_is_number);
            env.add_primitive("boolean?", crate::primitives::prim_is_boolean);
            env.add_primitive("symbol?", crate::primitives::prim_is_symbol);
            env.add_primitive("string?", crate::primitives::prim_is_string);
            env.add_primitive("procedure?", crate::primitives::prim_is_procedure);
        }
        env_ptr
    }

    /// Creates a new environment enclosed within an outer one.
    pub fn new_enclosed(outer_env: Rc<RefCell<Environment>>) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Environment {
            outer: Some(outer_env),
            bindings: HashMap::new(),
        }))
    }

    /// Defines a variable in the *current* environment frame.
    /// Replaces the value if the variable already exists in this frame.
    pub fn define(&mut self, name: String, value_node: Node) {
        self.bindings.insert(name, value_node);
    }

    /// Looks up a variable's value.
    /// Checks the current environment first, then walks up the outer environment chain.
    /// `lookup_span` is the location where the variable was referenced, used for error reporting.
    pub fn get(&self, name: &str, lookup_span: Span) -> Result<Node, EnvError> {
        // Try finding in the current environment's bindings
        if let Some(value_node) = self.bindings.get(name) {
            // Return a clone of the node found
            Ok(value_node.clone())
        } else {
            // If not found, try the outer environment recursively
            match &self.outer {
                Some(outer_env_ptr) => {
                    // Borrow the outer environment immutably for lookup
                    outer_env_ptr.borrow().get(name, lookup_span)
                }
                None => {
                    // Reached the top-level environment without finding it
                    Err(EnvError::UnboundVariable(name.to_string(), lookup_span))
                }
            }
        }
    }

    /// Sets the value of an *existing* variable in the environment chain.
    /// Searches outward from the current environment and updates the first frame
    /// where the variable is found. Errors if the variable is not defined.
    /// `set_span` is the location of the `set!` expression.
    pub fn set(&mut self, name: &str, value_node: Node, set_span: Span) -> Result<(), EnvError> {
        if let Some(value_mut) = self.bindings.get_mut(name) {
            // Variable exists in the current frame, update it here
            *value_mut = value_node;
            Ok(())
        } else {
            // Try setting in the outer environment
            match &self.outer {
                Some(outer_env_ptr) => {
                    // Borrow the outer environment mutably to potentially set
                    outer_env_ptr.borrow_mut().set(name, value_node, set_span)
                }
                None => {
                    // Reached the top level, variable not found anywhere
                    Err(EnvError::UnboundVariable(name.to_string(), set_span))
                }
            }
        }
    }

    /// Helper to add a primitive procedure to the environment.
    fn add_primitive(&mut self, name: &str, func: PrimitiveFunc) {
        let node = Node::new_primitive(func, name, Span::default());
        self.define(name.to_string(), node);
    }

    fn add_identifiers(&self, mut identifiers: HashSet<String>) -> HashSet<String> {
        // Try finding in the current environment's bindings
        for identifier in self.bindings.keys() {
            identifiers.insert(identifier.to_string());
        }
        identifiers
    }

    /// Gets a list of all identifiers in the current environment
    pub fn get_identifiers(&self) -> HashSet<String> {
        let identifiers = self.bindings.keys().map(|i| i.to_string()).collect();
        match self.outer {
            Some(ref outer_env_ptr) => outer_env_ptr.borrow().add_identifiers(identifiers),
            None => identifiers,
        }
    }
}

// --- Unit Tests ---
#[cfg(test)]
mod tests {
    use super::*;

    // Helper to create a dummy node with default span
    fn num_node(n: f64) -> Node {
        Node::new_number(n, Span::default())
    }

    fn sym_node(s: &str) -> Node {
        Node::new_symbol(s.to_string(), Span::default())
    }

    #[test]
    fn test_define_and_get_global() {
        let env = Environment::new();
        env.borrow_mut().define("x".to_string(), num_node(10.0));

        let result = env.borrow().get("x", Span::default());
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), num_node(10.0));
    }

    #[test]
    fn test_get_unbound_global() {
        let env = Environment::new();
        let result = env.borrow().get("y", Span::default());
        assert!(matches!(result, Err(EnvError::UnboundVariable(s, _)) if s == "y"));
    }

    #[test]
    fn test_define_and_get_enclosed() {
        let global_env = Environment::new();
        global_env
            .borrow_mut()
            .define("x".to_string(), num_node(10.0)); // Define x globally

        let local_env = Environment::new_enclosed(global_env);
        local_env
            .borrow_mut()
            .define("y".to_string(), num_node(20.0)); // Define y locally

        // Get local var y
        let result_y = local_env.borrow().get("y", Span::default());
        assert_eq!(result_y.unwrap(), num_node(20.0));

        // Get global var x from local scope
        let result_x = local_env.borrow().get("x", Span::default());
        assert_eq!(result_x.unwrap(), num_node(10.0));
    }

    #[test]
    fn test_get_unbound_enclosed() {
        let global_env = Environment::new();
        let local_env = Environment::new_enclosed(global_env);

        let span = Span::new(11, 12);
        let result = local_env.borrow().get("z", span);
        assert_eq!(
            result,
            Err(EnvError::UnboundVariable("z".to_string(), span))
        );
    }

    #[test]
    fn test_shadowing() {
        let global_env = Environment::new();
        global_env
            .borrow_mut()
            .define("x".to_string(), num_node(10.0));

        let local_env = Environment::new_enclosed(global_env.clone()); // Clone Rc for local
        local_env
            .borrow_mut()
            .define("x".to_string(), num_node(50.0)); // Shadow global x

        let inner_local_env = Environment::new_enclosed(local_env.clone()); // Clone Rc for inner local
        inner_local_env
            .borrow_mut()
            .define("y".to_string(), sym_node("y-value"));

        // Get x from inner local (should be 50.0 from local_env)
        assert_eq!(
            inner_local_env.borrow().get("x", Span::default()).unwrap(),
            num_node(50.0)
        );

        // Get y from inner local
        assert_eq!(
            inner_local_env.borrow().get("y", Span::default()).unwrap(),
            sym_node("y-value")
        );

        // Get x from local (should be 50.0)
        assert_eq!(
            local_env.borrow().get("x", Span::default()).unwrap(),
            num_node(50.0)
        );

        // Get x from global (should be 10.0)
        assert_eq!(
            global_env.borrow().get("x", Span::default()).unwrap(),
            num_node(10.0)
        );
    }

    // Tests for `set` would go here once implemented
    /*
    #[test]
    fn test_set_global() { ... }
    #[test]
    fn test_set_outer() { ... }
    #[test]
    fn test_set_unbound_error() { ... }
    */
}
