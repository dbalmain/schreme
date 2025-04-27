#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)] // Default for convenience
pub struct Span {
    pub start: usize, // Byte offset
    pub end: usize,   // Byte offset (exclusive)
}

impl Span {
    // Helper to merge two spans (e.g., for lists)
    pub fn merge(self, other: Span) -> Span {
        Span {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }
}
