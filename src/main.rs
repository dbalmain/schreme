// Use the library crate (whose name is defined in Cargo.toml)
use schreme::lexer::tokenize; // Use the specific items you need
use schreme::types::Sexpr; // Example if you use Sexpr directly here
// Or use rusty_scheme::lexer; etc.

fn main() {
    println!("Welcome to Rusty Scheme!");

    let input = "(define x 10)";
    println!("Input:\n{}", input);

    // Now call the functions via the library crate
    match tokenize(input) {
        Ok(tokens) => {
            println!("Tokens:");
            for token in tokens {
                println!("  {:?}", token);
            }
        }
        Err(e) => {
            eprintln!("Lexer Error: {}", e);
        }
    }
    // ... rest of your main function
}
