// Declare modules publicly so they are part of the library interface
pub mod environment;
pub mod evaluator;
pub mod lexer;
pub mod parser;
pub mod types;

// Optional: Re-export key types/functions for easier use by consumers
// of the library (like main.rs or benchmarks)
// pub use lexer::{tokenize, tokenize2, Token, LexerError};
// pub use types::Sexpr;
// ... etc.
