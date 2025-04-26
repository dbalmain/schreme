use schreme::parser::parse_str; // Use parse_str

fn main() {
    println!("Welcome to Rusty Scheme!");

    // --- TEMPORARY Parser Test ---
    let inputs = vec![
        "(define x (+ 10 5.5))",
        "; comment\n   '(\"hello\" #t (nested list))",
        "()",
        "123",
        "( unbalanced",   // Error case
        "'",              // Error case
        "\"unterminated", // Error case
    ];

    for input in inputs {
        println!("--------------------");
        println!("Input:\n{}", input);
        match parse_str(input) {
            // Use the helper function
            Ok(sexpr) => {
                println!("Parsed Sexpr: {:?}", sexpr);
                // Also test Display impl
                println!("Formatted: {}", sexpr);
            }
            Err(e) => {
                eprintln!("Parser Error: {}", e);
            }
        }
    }
    println!("--------------------");
    // --- End TEMPORARY Parser Test ---

    // Later, the REPL will use parse_str (or similar)
}
