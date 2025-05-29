use std::cell::RefCell;
use std::rc::Rc;

use rustyline::completion::Completer;
use rustyline::highlight::MatchingBracketHighlighter;
use rustyline::validate::MatchingBracketValidator;
use rustyline::{
    Cmd, Completer, Context, Editor, EventHandler, KeyCode, KeyEvent, Modifiers, Result,
};
use rustyline::{Helper, Highlighter, Hinter, Validator};
// src/bin/repl.rs
use rustyline::error::ReadlineError;
use schreme::TokenKind;
use schreme::{
    Environment, // Your existing parse_str or similar
    // Node, Sexpr, etc. might be needed for printing results
    // Assuming your library's root is `schreme`
    evaluator::{EvalError, evaluate}, // Adjust paths
    lexer::tokenize,
    parser::parse_str,
};

struct SchremeCompleter {
    env: Rc<RefCell<Environment>>,
}

impl SchremeCompleter {
    fn new(env: Rc<RefCell<Environment>>) -> Self {
        SchremeCompleter { env }
    }
}

impl rustyline::completion::Completer for SchremeCompleter {
    type Candidate = String;
    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &Context<'_>,
    ) -> rustyline::Result<(usize, Vec<String>)> {
        Ok((
            pos,
            match tokenize(&line[..pos]) {
                Ok(tokens) => {
                    if let Some(TokenKind::Symbol(prefix)) = tokens.last().map(|t| t.kind.clone()) {
                        self.env
                            .borrow()
                            .get_identifiers()
                            .union(&schreme::evaluator::special_form_identifiers())
                            .filter_map(|id| {
                                if id.starts_with(&prefix) {
                                    Some(id[prefix.len()..].to_string())
                                } else {
                                    None
                                }
                            })
                            .collect()
                    } else {
                        vec![]
                    }
                }
                Err(_) => vec![],
            },
        ))
    }
}

#[derive(Completer, Helper, Highlighter, Hinter, Validator)]
struct InputValidator {
    #[rustyline(Validator)]
    brackets: MatchingBracketValidator,
    #[rustyline(Highlighter)]
    highlighter: MatchingBracketHighlighter,
    #[rustyline(Completer)]
    completer: SchremeCompleter,
}

fn main() -> rustyline::Result<()> {
    println!("Schreme REPL v0.1.0"); // Or your version
    println!("Type 'exit' or press Ctrl-D to quit.");

    let global_env = Environment::new_global_populated();
    let h = InputValidator {
        brackets: MatchingBracketValidator::new(),
        highlighter: MatchingBracketHighlighter::new(),
        completer: SchremeCompleter::new(global_env.clone()),
    };
    let mut rl = Editor::new()?;
    rl.set_helper(Some(h));
    rl.bind_sequence(
        KeyEvent(KeyCode::Char('s'), Modifiers::CTRL),
        EventHandler::Simple(Cmd::Newline),
    );
    if rl.load_history("schreme_history.txt").is_err() {
        println!("No previous history.");
    }

    loop {
        let readline = rl.readline("schreme> ");
        match readline {
            Ok(line) => {
                rl.add_history_entry(line.as_str())?;
                let trimmed_input = line.trim();
                if trimmed_input.is_empty() {
                    continue;
                }
                if trimmed_input.eq_ignore_ascii_case("exit") {
                    break;
                }

                match parse_str(trimmed_input) {
                    Ok(node) => {
                        match evaluate(node, global_env.clone()) {
                            // Clone Rc for each eval
                            Ok(result_node) => {
                                // TODO: Pretty print result_node
                                println!("{}", result_node.to_string());
                            }
                            Err(e) => {
                                // TODO: Pretty print error e
                                eprintln!("Error: {}", e);
                            }
                        }
                    }
                    Err(parse_err) => {
                        // TODO: Pretty print parse_err
                        eprintln!("Parse Error: {}", parse_err);
                    }
                }
            }
            Err(ReadlineError::Interrupted) => {
                // Ctrl-C
                println!("Interrupted. Type 'exit' or Ctrl-D to quit.");
            }
            Err(ReadlineError::Eof) => {
                // Ctrl-D
                println!("\nExiting.");
                break;
            }
            Err(err) => {
                eprintln!("Readline Error: {:?}", err);
                break;
            }
        }
    }
    rl.save_history("schreme_history.txt")
}
