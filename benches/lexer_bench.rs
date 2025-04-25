use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use schreme::lexer::tokenize; // Import your functions

// A reasonably complex input string for benchmarking
const BENCH_INPUT: &str = r#"
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
(define (fib n)
  ; Calculate the nth Fibonacci number
  (if (< n 2)
      n
      (+ (fib (- n 1))
         (fib (- n 2)))))

(define (factorial n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

; Some calls
(fib 10)
(factorial 5)
'("string with spaces" #t #f 123 45.67 -10 +)
; Another comment at the end
'("string with escapes \"\n\r\t\t\n\r\"" #t #f 123 45.67 -10 +)
"#;

fn bench_tokenizers(c: &mut Criterion) {
    // Create a benchmark group
    let mut group = c.benchmark_group("Tokenizer Comparison");

    // Benchmark the original tokenize function
    group.bench_with_input(
        BenchmarkId::new("tokenize", "complex_input"), // Label for the benchmark
        &BENCH_INPUT,                                  // Input parameter passed to the closure
        |b, input| {
            // `b` is the Bencher object
            // `input` is &BENCH_INPUT
            // `iter` runs the closure multiple times to collect data
            // `black_box` prevents the compiler from optimizing away the input/work
            b.iter(|| tokenize(black_box(input)))
        },
    );

    group.finish(); // Finish the group
}

// Register the benchmark group with Criterion
criterion_group!(benches, bench_tokenizers);
// Generate the main function necessary for the benchmark executable
criterion_main!(benches);
