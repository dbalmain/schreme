use std::fmt;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)] // Default for convenience
pub struct Span {
    pub start: usize, // Byte offset
    pub end: usize,   // Byte offset (exclusive)
}

impl Span {
    pub fn new(start: usize, end: usize) -> Self {
        Span { start, end }
    }

    // Helper to merge two spans (e.g., for lists)
    pub fn merge(&self, other: &Span) -> Span {
        Span {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }

    pub fn to_range(&self) -> std::ops::RangeInclusive<usize> {
        self.start..=(if self.end == 0 { 0 } else { self.end - 1 })
    }
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}..{}", self.start, self.end)
    }
}
