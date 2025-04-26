use std::fmt; // For custom display formatting

/// Represents a Scheme S-expression (Symbolic Expression).
/// This enum will be the core data structure for both code (AST) and data.
#[derive(Debug, Clone, PartialEq)] // Add traits for easy debugging, copying, and comparison
pub enum Sexpr {
    Symbol(String),   // e.g., +, variable-name, quote
    Number(f64),      // Using f64 for simplicity for now
    Boolean(bool),    // #t or #f
    String(String),   // For string literals "hello\n"
    List(Vec<Sexpr>), // e.g., (+ 1 2), (define x 10)
    Nil,              // Represents the empty list '()

                      // --- Future additions ---
                      // Pair(Box<Sexpr>, Box<Sexpr>), // For dotted pairs, alternative list rep
                      // Primitive(PrimitiveFunc),
                      // Lambda(LambdaExpr),
}

// Implement Display trait for pretty printing the Sexpr values
impl fmt::Display for Sexpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Sexpr::Symbol(s) => write!(f, "{}", s),
            Sexpr::Number(n) => write!(f, "{}", n),
            Sexpr::Boolean(b) => write!(f, "{}", if *b { "#t" } else { "#f" }),
            Sexpr::List(list) => {
                write!(f, "(")?;
                let mut first = true;
                for expr in list {
                    if !first {
                        write!(f, " ")?;
                    }
                    write!(f, "{}", expr)?;
                    first = false;
                }
                write!(f, ")")
            }
            Sexpr::Nil => write!(f, "()"),
            Sexpr::String(str) => {
                write!(
                    f,
                    "\"{}\"",
                    str.replace("\"", "\\\"")
                        .replace("\n", "\\n")
                        .replace("\r", "\\r")
                        .replace("\t", "\\t")
                )
            }
        }
    }
}

// --- Placeholders for future types (optional for now) ---
/*
#[derive(Debug, Clone, PartialEq)]
pub struct LambdaExpr {
    pub params: Vec<String>, // Parameter names
    pub body: Box<Sexpr>,    // Body expression
    // pub env: Environment, // Closure environment - needs careful design
}

// Type alias for built-in primitive functions
pub type PrimitiveFunc = fn(Vec<Sexpr>) -> Result<Sexpr, String>; // Simple signature for now
*/
