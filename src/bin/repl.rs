// src/bin/repl.rs
use schreme::{
    Environment, // Your existing parse_str or similar
    // Node, Sexpr, etc. might be needed for printing results
    // Assuming your library's root is `schreme`
    evaluator::{EvalError, evaluate}, // Adjust paths
    lexer::tokenize,
    parser::parse_str,
};
use std::io::{self, Write}; // For stdin/stdout

fn main() {
    println!("Schreme REPL v0.1.0"); // Or your version
    println!("Type 'exit' or press Ctrl-D to quit.");

    let global_env = Environment::new_global_populated(); // Your global env

    loop {
        print!("schreme> "); // Prompt
        io::stdout().flush().unwrap(); // Ensure prompt is shown before input

        let mut input_line = String::new();
        match io::stdin().read_line(&mut input_line) {
            Ok(0) => {
                // EOF (Ctrl-D)
                println!("\nExiting.");
                break;
            }
            Ok(_) => {
                let trimmed_input = input_line.trim();
                if trimmed_input.is_empty() {
                    continue; // Skip empty lines
                }
                if trimmed_input.eq_ignore_ascii_case("exit") {
                    println!("Exiting.");
                    break;
                }

                // TODO: Handle multi-line input later

                match parse_str(trimmed_input) {
                    Ok(node) => {
                        match evaluate(node, global_env.clone()) {
                            // Clone Rc for each eval
                            Ok(result_node) => {
                                // TODO: Pretty print result_node
                                println!("{}", result_node.to_string()); // Or result_node.to_string()
                            }
                            Err(e) => {
                                // TODO: Pretty print error e
                                eprintln!("Error: {}", e); // Basic error printing
                            }
                        }
                    }
                    Err(parse_err) => {
                        // TODO: Pretty print parse_err
                        eprintln!("Parse Error: {}", parse_err);
                    }
                }
            }
            Err(io_err) => {
                eprintln!("I/O Error: {}", io_err);
                break;
            }
        }
    }
}
