use std::cell::RefCell;
use std::rc::Rc;

use rustyline::error::ReadlineError;
use rustyline::highlight::{CmdKind, Highlighter};
use rustyline::validate::{ValidationContext, ValidationResult, Validator};
use rustyline::{Cmd, Completer, Context, Editor, EventHandler, KeyCode, KeyEvent, Modifiers};
use rustyline::{Helper, Highlighter, Hinter, Validator};
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
    validator: SchremeValidator,
    #[rustyline(Highlighter)]
    highlighter: SchremeHighlighter,
    #[rustyline(Completer)]
    completer: SchremeCompleter,
}

struct SchremeValidator;

impl Validator for SchremeValidator {
    fn validate(&self, ctx: &mut ValidationContext) -> rustyline::Result<ValidationResult> {
        let input = ctx.input();
        let mut stack = Vec::new();
        let mut in_string = false;
        let mut escape = false;

        for (i, c) in input.chars().enumerate() {
            if in_string {
                if escape {
                    escape = false;
                } else if c == '\\' {
                    escape = true;
                } else if c == '"' {
                    in_string = false;
                }
                continue;
            }

            match c {
                '"' => {
                    in_string = true;
                }
                '(' | '[' | '{' => {
                    stack.push((c, i));
                }
                ')' | ']' | '}' => {
                    if let Some((opening, _)) = stack.pop() {
                        if !((opening == '(' && c == ')')
                            || (opening == '[' && c == ']')
                            || (opening == '{' && c == '}'))
                        {
                            return Ok(ValidationResult::Invalid(Some(format!(
                                "  - Unmatched '{}' at position {}",
                                c, i
                            ))));
                        }
                    } else {
                        return Ok(ValidationResult::Invalid(Some(format!(
                            "  - Unmatched '{}' at position {}",
                            c, i
                        ))));
                    }
                }
                _ => {}
            }
        }

        if in_string {
            Ok(ValidationResult::Incomplete)
        } else if let Some((_, _)) = stack.pop() {
            Ok(ValidationResult::Incomplete)
        } else {
            Ok(ValidationResult::Valid(None))
        }
    }
}

struct SchremeHighlighter;

impl Highlighter for SchremeHighlighter {
    fn highlight<'l>(&self, line: &'l str, pos: usize) -> std::borrow::Cow<'l, str> {
        let mut stack: Vec<(char, usize)> = Vec::new();
        let mut highlighted = String::new();
        let mut in_string = false;
        let mut escape = false;

        for (i, c) in line.chars().enumerate() {
            if in_string {
                if escape {
                    escape = false;
                } else if c == '\\' {
                    escape = true;
                } else if c == '"' {
                    in_string = false;
                }
                highlighted.push_str(&format!("\x1b[32m{}\x1b[0m", c)); // Green for strings
                continue;
            }

            match c {
                '"' => {
                    in_string = true;
                    highlighted.push_str(&format!("\x1b[32m{}\x1b[0m", c)); // Green for strings
                }
                '(' | '[' | '{' => {
                    stack.push((c, highlighted.len()));
                    highlighted.push(c);
                }
                ')' | ']' | '}' => {
                    if let Some((opening, matching_pos)) = stack.pop() {
                        if (opening == '(' && c == ')')
                            || (opening == '[' && c == ']')
                            || (opening == '{' && c == '}')
                        {
                            if matching_pos == pos - 1 || i == pos - 1 {
                                highlighted.push_str(&format!("\x1b[34m{}\x1b[0m", c)); // Blue for matching brackets
                                highlighted.replace_range(
                                    matching_pos..=matching_pos,
                                    &format!("\x1b[1;34m{}\x1b[0m", opening as char),
                                );
                            } else {
                                highlighted.push(c);
                            }
                        } else {
                            highlighted.push_str(&format!("\x1b[31m{}\x1b[0m", c)); // Red for unmatched closing brackets
                            highlighted.replace_range(
                                matching_pos..=matching_pos,
                                &format!("\x1b[1;31m{}\x1b[0m", opening as char),
                            );
                        }
                    } else {
                        highlighted.push_str(&format!("\x1b[31m{}\x1b[0m", c)); // Red for unmatched closing brackets
                    }
                }
                _ => {
                    highlighted.push(c);
                }
            }
        }

        std::borrow::Cow::Owned(highlighted)
    }

    fn highlight_char(&self, _line: &str, _pos: usize, _kind: CmdKind) -> bool {
        return true;
    }
}

fn main() -> rustyline::Result<()> {
    println!("Schreme REPL v0.1.0"); // Or your version
    println!("Type 'exit' or press Ctrl-D to quit.");

    let global_env = Environment::new_global_populated();
    let h = InputValidator {
        highlighter: SchremeHighlighter,
        validator: SchremeValidator,
        completer: SchremeCompleter::new(global_env.clone()),
    };
    let config = rustyline::config::Config::builder()
        .edit_mode(rustyline::EditMode::Vi)
        .build();
    let mut rl = Editor::with_config(config)?;
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
