# Schreme - A Scheme-inspired language & interpreter in Rust

This project aims to build an interpreter, initially for a subset of Scheme, iteratively adding features and eventually evolving into a distinct language with type safety and a Cranelift-based backend.

The goal is to learn about building a language along with all of the supporting tooling like formatter, language server, etc.

## TODO Roadmap

This list outlines the planned development steps, subject to change as the project evolves.

### Phase 1: Core Scheme Subset & Foundation

* [x] **Project Setup:** Basic Rust binary project structure (`Cargo.toml`, `src/main.rs`).
    * [x] Module structure (`lexer.rs`, `types.rs`, `parser.rs`, `evaluator.rs`, `environment.rs`).
    * [x] Setup `lib.rs` for library structure.
* [ ] **Core Data Types (`types.rs`):** Define `Sexpr` enum (Symbol, Number, Boolean, List, Nil).
    * [x] Implement `Display` for pretty-printing `Sexpr`.
    * [x] Add `Pair`/`Cons` cell representation (alternative/complement to `Vec<Sexpr>` for lists).
    * [x] Add `String` type.
    * [ ] Add `Vector` type (`#( ... )`).
    * [ ] Add `Character` type (`#\a`).
    * [ ] Add `HashMap` type
    * [ ] Add `HashSet` type
* [x] **Lexer (`lexer.rs`):** Convert input string to `Token` stream.
    * [x] Handle basic tokens: `(`, `)`, `'`, symbols, numbers (float), booleans (`#t`, `#f`).
    * [x] Handle string literals (`"`).
    * [x] Handle whitespace and comments (`;`).
    * [x] Implement robust error handling (`LexerError`).
    * [x] Add comprehensive unit tests.
    * [x] Setup benchmarking with Criterion (`cargo bench`).
    * [x] Turn the lexer into a FSM
* [x] **Parser (`parser.rs`):** Convert `Token` stream to `Sexpr` (AST).
    * [x] Parse atoms (symbols, numbers, booleans, strings).
    * [x] Parse lists (`(...)`).
    * [x] Handle quote sugar (`'expr` -> `(quote expr)`).
    * [x] Handle quasiquote/unquote (`,`, `,@`) later.
    * [x] Implement robust error handling (`ParseError`).
    * [x] Add comprehensive unit tests.
    * [x] Add helpful explanations of errors, like Elm.
* [x] **Basic Environment (`environment.rs`):** Store variable bindings.
    * [x] Implement nested environments (for lexical scoping).
    * [x] Define/get variables.
    * [x] Handle `set!` for mutation.
* [x] **Evaluator (`evaluator.rs`):** Execute `Sexpr` AST.
    * [x] Evaluate self-evaluating atoms (numbers, booleans, strings).
    * [x] Evaluate symbols (variable lookup in environment).
    * [x] Implement `quote` special form.
    * [x] Implement `if` special form.
    * [x] Implement `begin` special form.
    * [x] Implement `define` special form (global/local).
    * [x] Implement `set!` special form.
    * [x] Implement `lambda` special form
    * [x] Implement `let` special form
    * [x] Implement `let*` special form
    * [x] Implement `letrec` special form
    * [x] Implement basic procedure calls (primitives first).
    * [x] Implement robust error handling (`EvalError`).
* [ ] **Primitive Procedures:** Implement core built-in functions.
    * [x] Basic arithmetic (`+`, `-`, `*`, `/`).
    * [x] Comparison (`=`, `<`, `>`, `<=`, `>=`). (Note: `=` is numeric equal in Scheme)
    * [x] Type predicates (`number?`, `symbol?`, `boolean?`, `list?`, `pair?`, `null?`, `procedure?`).
    * [x] List operations (`cons`, `car`, `cdr`, `list`).
    * [x] Lambda application (`apply`)
    * [ ] Equality (`eq?`, `eqv?`, `equal?`).
* [ ] **REPL (`main.rs` or separate module):** Basic Read-Eval-Print Loop.
    * [x] Read input line by line.
    * [x] Integrate Lexer, Parser, Evaluator.
    * [x] Print results or errors.
    * [x] Handle multiline input.
    * [x] Add history (using rustyline?).
    * [x] Add tab completion.
    * [x] Print clear errors showing where the error occured.
    * [ ] Syntax highlight bound and unbound variables.
* [x] **Error Handling:** Unified and user-friendly error reporting.
    * [x] Include source location/span information in errors.
    * [x] Consistent error types/display.

### Phase 2: Tooling & Ecosystem (Start Early)

* [ ] **Language Server Protocol (LSP) Implementation (`schreme-lsp/`?):** Provide IDE features.
    * [ ] **Setup:** Create a separate LSP binary crate.
    * [ ] **Communication:** Implement JSON-RPC communication layer (e.g., using `lsp-server`, `tower-lsp`).
    * [ ] **Basic Diagnostics:** Report lexer/parser errors (`textDocument/publishDiagnostics`). Requires parsing on change.
    * [ ] **Syntax Highlighting:** Implement basic semantic token highlighting (`textDocument/semanticTokens`).
    * [ ] **Completions:** Offer basic completions for keywords and primitives (`textDocument/completion`).
    * [ ] **Hover:** Show basic type info for primitives (`textDocument/hover`).
    * [ ] **Advanced Features (Later):**
        * [ ] Go To Definition (requires environment analysis).
        * [ ] Find References.
        * [ ] Renaming.
        * [ ] Diagnostics based on basic analysis (e.g., unbound variables).
* [ ] **Basic Formatter (`schreme-fmt/`?):** Auto-format code.
    * [ ] Define formatting rules.
    * [ ] Integrate with parser/AST.
    * [ ] Command-line tool.
* [ ] **Tree-sitter Parser (Exploration/Alternative):**
    * [ ] Evaluate existing `tree-sitter-scheme`.
    * [ ] Consider creating `tree-sitter-schreme` if syntax diverges significantly.
    * [ ] Explore using Tree-sitter *within* the LSP for faster, more resilient parsing for IDE features.
* [ ] **Debugger Implementation:** Provide step-through debugging capabilities.
    * [ ] **Foundation & Control:**
        * [ ] Define Breakpoint structure (file/line/column or span).
        * [ ] Implement setting/clearing breakpoints.
        * [ ] Modify Evaluator loop/structure to check for breakpoints and pause execution.
        * [ ] Implement execution control commands (Step In, Step Over, Step Out, Continue).
    * [ ] **State Inspection:**
        * [ ] Inspect current `Environment` bindings (variable names/values).
        * [ ] Implement call stack tracking during evaluation.
        * [ ] Display the current call stack.
        * [ ] Pretty-print `Sexpr`/`Node` values during inspection.
    * [ ] **Interface/Protocol:**
        * [ ] **Option A: Debug Adapter Protocol (DAP):**
            * [ ] Setup separate DAP server binary.
            * [ ] Implement DAP communication (JSON-RPC).
            * [ ] Handle DAP requests (setBreakpoints, configurationDone, threads, stackTrace, scopes, variables, continue, stepIn, etc.).
            * [ ] Send DAP events (stopped, terminated, output, etc.).
        * [ ] **Option B: REPL Integration:**
            * [ ] Add debug-specific commands to the REPL (`break`, `step`, `continue`, `print`, `stack`, etc.).
    * [ ] **Cranelift Backend Debugging (Later):**
        * [ ] Generate Debug Information (DWARF?) mapping compiled code to source Spans during Cranelift compilation.
        * [ ] Investigate integration with native debuggers (GDB/LLDB) via DWARF.
        * [ ] Explore JIT debugging hooks if using `cranelift-jit`.
* [ ] **Profiler Implementation:** Analyze performance characteristics.
    * [ ] **Data Collection Strategy:**
        * [ ] **Option A: Instrumentation:**
            * [ ] Instrument procedure calls (entry/exit time/count) in the evaluator.
            * [ ] Instrument primitive calls.
        * [ ] **Option B: Sampling:**
            * [ ] Periodically sample the interpreter's call stack.
        * [ ] Choose and implement one or both strategies.
    * [ ] **Metrics:**
        * [ ] Track time spent per function/primitive.
        * [ ] Track call counts per function/primitive.
        * [ ] (Optional) Track allocations per function.
    * [ ] **Interpreter Integration:**
        * [ ] Add hooks/wrappers around evaluation steps and function calls.
        * [ ] Manage profiler state (enabled/disabled, data storage).
    * [ ] **Data Reporting:**
        * [ ] Implement basic text-based report (e.g., table sorted by time).
        * [ ] Output data compatible with external tools (e.g., `perf` format, flamegraph collapsed stack format).
        * [ ] Implement flame graph generation (e.g., using `inferno`).
    * [ ] **Cranelift Backend Profiling (Later):**
        * [ ] Integrate with Cranelift/LLVM profiling features if applicable.
        * [ ] Correlate compiled code performance metrics back to Schreme source functions.
* [ ] **Testing Framework:**
    * [ ] Develop infrastructure for running Scheme code tests against the interpreter.
    * [ ] Integrate with `cargo test`.
* [ ] **Package/Module System (`schreme-pkg/`?):** Manage dependencies and code organization.
    * [ ] **Design:** Define module syntax (`define-library`, custom?).
    * [ ] **Manifest:** Define a package manifest file format (`Schreme.toml`?).
    * [ ] **Resolution:** Implement dependency resolution logic.
    * [ ] **Fetching:** Mechanism to fetch dependencies (Git, registry?).
    * [ ] **Build Integration:** Integrate package management into the execution flow.

### Phase 3: Advanced Scheme Features

* [x] **Lambda & Closures:** Implement user-defined procedures.
    * [x] `lambda` special form.
    * [x] Proper lexical scoping (capture environment).
    * [x] Procedure call evaluation for user functions.
* [ ] **Tail Call Optimization (TCO):** Essential for idiomatic Scheme recursion.
    * [ ] Implement TCO within the tree-walking evaluator (Trampoline, or direct jump).
* [ ] **Macros:** Implement hygienic macros (`syntax-rules`).
    * [ ] `define-syntax`, `syntax-rules`.
    * [ ] Expansion mechanism.
    * [ ] Consider `syntax-case` later for more power.
* [ ] **Continuations:** Implement `call/cc` (`call-with-current-continuation`). (Challenging but powerful).
* [ ] **More Data Types & Primitives:**
    * [ ] Vectors (`vector`, `vector-ref`, `vector-set!`, etc.).
    * [ ] Characters (`char?`, `char=?`, etc.).
    * [ ] Ports (Input/Output: `open-input-file`, `read`, `write`, `display`).
    * [ ] Hash Tables.
* [ ] **File Execution:** Allow running `.ss`/`.scm` files directly.
* [ ] **Garbage Collection (GC):** Crucial for managing memory, especially with closures and continuations.
    * [ ] Choose a GC strategy (Reference Counting, Mark-Sweep, Generational).
    * [ ] Integrate GC with Rust data structures (using libraries like `gc-arena`, `bacon-rajan-cc`, or custom).

### Phase 4: Cranelift Backend & Performance

* [ ] **Intermediate Representation (IR):** Design an IR suitable for compilation.
    * [ ] Possibly a typed IR distinct from `Sexpr`.
    * [ ] SSA (Static Single Assignment) form?
* [ ] **Lowering:** Translate `Sexpr` AST (or a high-level IR) to the chosen IR.
* [ ] **Cranelift Integration:**
    * [ ] Basic setup: Link Cranelift, create `Context`, `Module`.
    * [ ] Function compilation: Translate IR functions to Cranelift IR (`cranelift-frontend`).
    * [ ] JIT Execution: Compile and execute functions on the fly (`cranelift-jit`).
    * [ ] AOT Compilation: Option to compile to object files (`cranelift-object`).
* [ ] **Runtime System:** Support code generated by Cranelift.
    * [ ] Memory management integration (calling GC).
    * [ ] Primitive function implementation callable from compiled code.
    * [ ] Calling conventions between interpreted and compiled code.
    * [ ] Exception/Continuation handling integration.
* [ ] **Optimization:** Apply optimization passes within Cranelift.

### Phase 5: Schreme - Divergence & Custom Features

* [ ] **Static Typing:** Design and implement a type system.
    * [ ] Type syntax.
    * [ ] Type checking algorithm.
    * [ ] Type inference?
    * [ ] Integrate type checking into the evaluator or compilation pipeline.
    * [ ] Update LSP for type information and errors.
* [ ] **Syntax Changes:** Evolve the language syntax away from pure S-expressions if desired.
    * [ ] This might require significant parser changes (potentially favouring Tree-sitter).
* [ ] **Feature Modification/Removal:** Change or remove Scheme features that don't fit the vision for Schreme.
* [ ] **Foreign Function Interface (FFI):** Define how to call Rust/C code from Schreme and vice-versa.
* [ ] **Concurrency/Parallelism:** Design and implement features for concurrent execution (actors, futures?).

### Phase 6: Polish & Release

* [ ] **Documentation:** Comprehensive user guides, language reference, API docs.
* [ ] **Examples:** Provide clear examples of how to use Schreme.
* [ ] **Refinement:** Code cleanup, address TODOs/FIXMEs.
* [ ] **Further Benchmarking and Optimization.**
* [ ] **Release Strategy:** Versioning, packaging.

### Meta / Ongoing

* [ ] Maintain comprehensive unit and integration tests.
* [ ] Keep dependencies updated.
* [ ] Refactor code for clarity and maintainability.
* [ ] Write documentation alongside features.
* [ ] Continuously benchmark performance bottlenecks.
