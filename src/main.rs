use schreme::{Environment, evaluate, parser::parse_str}; // Use parse_str

fn main() {
    println!("Welcome to Schreme!");

    // Create a global environment for evaluation
    let global_env = Environment::new();
    // Optional: Pre-populate global env with primitives later

    let inputs = vec![
        "123",
        "\"hello\"",
        "#t",
        "'symbol",   // quote symbol
        "'(1 2 #f)", // quote list
        // Add variable test once define works, or pre-define manually
        // "x", // Error: unbound
        "(quote two args)", // Error: quote args
        "(1 2 3)",          // Error: 1 not procedure
    ];

    println!("--- Evaluating Inputs ---");
    for input in inputs {
        println!("--------------------");
        println!("Input: {}", input);
        match parse_str(input) {
            Ok(node) => {
                println!("Parsed: {:?}", node.kind); // Show parsed structure
                // Evaluate the parsed node in the global environment
                match evaluate(node, global_env.clone()) {
                    // Clone the Rc to share env ownership
                    Ok(result_node) => {
                        println!("Result Kind: {:?}", result_node.kind);
                        println!("Result Span: {:?}", result_node.span);
                        println!("Formatted: {}", result_node); // Uses Display for Node/Sexpr
                    }
                    Err(e) => {
                        eprintln!("Evaluation Error: {}", e);
                        // TODO: Print better errors using span info + source context
                    }
                }
            }
            Err(e) => {
                eprintln!("Parser Error: {}", e);
            }
        }
    }
    println!("--------------------");
    println!("--- End Evaluation ---");
}
