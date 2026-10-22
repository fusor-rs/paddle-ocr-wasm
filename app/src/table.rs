//! Tables in OTSL, as PaddleOCR-VL writes them: each row ends with `<nl>`, and each cell starts
//! with a tag: `<fcel>` its text follows, `<ecel>` it is empty, `<lcel>` it merges into the
//! cell on its left, `<ucel>` into the one above, `<xcel>` into both.

const NEW_ROW: &str = "<nl>";
const TAGS: [(&str, Kind); 5] = [
    ("<fcel>", Kind::Text),
    ("<ecel>", Kind::Empty),
    ("<lcel>", Kind::Left),
    ("<ucel>", Kind::Up),
    ("<xcel>", Kind::Both),
];

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Text,
    Empty,
    Left,
    Up,
    Both,
}

/// A cell the page draws: its text, and the columns and rows it spans.
#[derive(Clone, PartialEq)]
pub(crate) struct Cell {
    pub(crate) column: usize,
    pub(crate) text: String,
    pub(crate) columns: usize,
    pub(crate) rows: usize,
}

#[derive(Clone, PartialEq)]
pub(crate) struct Row {
    pub(crate) index: usize,
    pub(crate) cells: Vec<Cell>,
}

/// The rows of an OTSL table, each with the cells that start in it; none for text without
/// cells.
pub(crate) fn rows(otsl: &str) -> Vec<Row> {
    let grid: Vec<Vec<(Kind, &str)>> = otsl
        .split(NEW_ROW)
        .map(cells)
        .filter(|row| !row.is_empty())
        .collect();
    let kinds_below = |index: usize, column: usize| {
        let below = grid[index + 1..].iter();
        below.map(move |row| row.get(column).map_or(Kind::Empty, |cell| cell.0))
    };
    let row = |(index, row): (usize, &Vec<(Kind, &str)>)| {
        let starts = row
            .iter()
            .enumerate()
            .filter(|(_, cell)| matches!(cell.0, Kind::Text | Kind::Empty));
        let cells = starts.map(|(column, (_, text))| Cell {
            column,
            text: text.trim().into(),
            columns: 1 + run(row[column + 1..].iter().map(|cell| cell.0), Kind::Left),
            rows: 1 + run(kinds_below(index, column), Kind::Up),
        });
        Row {
            index,
            cells: cells.collect(),
        }
    };
    grid.iter().enumerate().map(row).collect()
}

/// How many of `kinds` are `kind` before the first that is not.
fn run(kinds: impl Iterator<Item = Kind>, kind: Kind) -> usize {
    kinds.take_while(|&next| next == kind).count()
}

/// A row's cells, each with the text after its tag.
fn cells(row: &str) -> Vec<(Kind, &str)> {
    let mut cells: Vec<(Kind, &str)> = Vec::new();
    let mut rest = row;
    while let Some((at, tag, kind)) = next_tag(rest) {
        if let Some(last) = cells.last_mut() {
            last.1 = &rest[..at];
        }
        cells.push((kind, ""));
        rest = &rest[at + tag.len()..];
    }
    if let Some(last) = cells.last_mut() {
        last.1 = rest;
    }
    cells
}

fn next_tag(text: &str) -> Option<(usize, &'static str, Kind)> {
    let found = TAGS
        .iter()
        .filter_map(|&(tag, kind)| text.find(tag).map(|at| (at, tag, kind)));
    found.min_by_key(|(at, ..)| *at)
}
