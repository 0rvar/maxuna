//! Interactive chat REPL on a raw-mode crossterm line editor.
//!
//! Why raw mode instead of `read_line`:
//! - Multi-line pastes must not submit: bracketed paste delivers the whole
//!   paste as one event (iTerm2/kitty/Ghostty/WezTerm). Terminal.app has no
//!   bracketed paste, so a pasted newline is recognized there by the input
//!   flood (another event is already queued when the Enter arrives) and is
//!   inserted instead of submitting.
//! - Shift+Enter inserts a newline on terminals speaking the kitty keyboard
//!   protocol; Alt+Enter and Ctrl+J work everywhere.
//! - Ctrl-C becomes ours: it cancels a running generation (polled by
//!   `Generator::generate` between chunks/tokens), clears a non-empty prompt,
//!   and quits on an empty prompt. A dedicated input thread reads events, so
//!   cancellation is registered even while the main thread is blocked inside a
//!   Metal forward — and if a cancel is never honored (wedged GPU work), a
//!   repeat Ctrl-C ≥1s later restores the terminal and force-exits.

use std::io::{IsTerminal, Stdout, Write, stdout};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::time::{Duration, Instant};

use anyhow::Result;
use crossterm::event::{
    self, DisableBracketedPaste, EnableBracketedPaste, Event, KeyCode, KeyEvent, KeyEventKind,
    KeyModifiers, KeyboardEnhancementFlags, PopKeyboardEnhancementFlags,
    PushKeyboardEnhancementFlags,
};
use crossterm::terminal::{self, Clear, ClearType, disable_raw_mode, enable_raw_mode};
use crossterm::{cursor, execute, queue};
use unicode_width::UnicodeWidthChar;

use laguna::chat::{ChatOptions, Message, build_prompt};
use laguna::generate::{GenStats, Generator};

const DIM: &str = "\x1b[2m";
const RESET: &str = "\x1b[0m";
/// Both prefixes render 2 columns wide (see `PREFIX_W`).
const PROMPT_PREFIX: &str = "\x1b[36m\u{276f}\x1b[0m ";
const CONT_PREFIX: &str = "\x1b[2m\u{2502}\x1b[0m ";
const PREFIX_W: usize = 2;

pub fn run(generator: &mut Generator, max_tokens: usize, show_thinking: bool) -> Result<()> {
    if !std::io::stdin().is_terminal() || !stdout().is_terminal() {
        return pipe_repl(generator, max_tokens, show_thinking);
    }

    let mut show_thinking = show_thinking;
    let opts = ChatOptions { enable_thinking: true };
    let mut messages: Vec<Message> = Vec::new();
    let mut history: Vec<String> = Vec::new();
    let mut out = stdout();

    let guard = TermGuard::new()?;
    {
        // A panic unwinds past the guard only after the hook has run; restore
        // the terminal first so the panic message prints legibly.
        let enhanced = guard.enhanced;
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            restore_terminal(enhanced);
            prev(info);
        }));
    }
    let generating = Arc::new(AtomicBool::new(false));
    let cancel = Arc::new(AtomicBool::new(false));
    let events = spawn_input_thread(generating.clone(), cancel.clone(), guard.enhanced);

    banner(&mut out, generator.max_ctx(), guard.enhanced)?;

    loop {
        let input = match read_turn(&events, &mut out, &history)? {
            Turn::Quit => break,
            Turn::Submit(text) => text,
        };
        match input.trim() {
            "exit" | "quit" | "/exit" | "/quit" => break,
            "/clear" => {
                messages.clear();
                dim_line(&mut out, "conversation cleared")?;
                continue;
            }
            "/think" => {
                show_thinking = !show_thinking;
                dim_line(
                    &mut out,
                    if show_thinking { "showing reasoning" } else { "hiding reasoning" },
                )?;
                continue;
            }
            "/help" => {
                help(&mut out)?;
                continue;
            }
            _ => {}
        }
        history.push(input.clone());
        messages.push(Message::User(input));
        let prompt = build_prompt(&messages, &opts)?;

        // Clear cancel BEFORE raising `generating`: a Ctrl-C that raced the end
        // of the previous turn must not instantly cancel this one.
        cancel.store(false, Ordering::SeqCst);
        generating.store(true, Ordering::SeqCst);
        let outcome = stream_reply(generator, &prompt, max_tokens, show_thinking, &cancel, &mut out);
        generating.store(false, Ordering::SeqCst);

        match outcome {
            Err(err) => {
                // Keep the REPL alive (e.g. prompt outgrew max_ctx); drop the
                // user turn so the conversation state matches what the model saw.
                messages.pop();
                dim_line(&mut out, &format!("error: {err:#}"))?;
            }
            Ok((full, stats)) => {
                let (reasoning, content) = split_thinking(&full);
                // A cancel before `</think>` leaves only raw reasoning in
                // `full`, which split_thinking misfiles as content — nothing
                // usable either way. Forget the turn so it can be retried.
                if stats.cancelled && (content.is_empty() || !full.contains("</think>")) {
                    messages.pop();
                } else {
                    messages.push(Message::Assistant {
                        content: content.to_string(),
                        reasoning: if reasoning.is_empty() {
                            None
                        } else {
                            Some(reasoning.to_string())
                        },
                    });
                }
                let used = stats.prefill_tokens + stats.decode_tokens;
                write!(
                    out,
                    "{DIM}prefill {} tok \u{b7} {:.0} tok/s   decode {} tok \u{b7} {:.1} tok/s   \
                     ctx {used}/{}{RESET}\r\n\r\n",
                    stats.prefill_tokens,
                    stats.prefill_tps(),
                    stats.decode_tokens,
                    stats.decode_tps(),
                    generator.max_ctx(),
                )?;
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Terminal setup/teardown

struct TermGuard {
    /// Terminal speaks the kitty keyboard protocol (Shift+Enter distinguishable).
    enhanced: bool,
}

impl TermGuard {
    fn new() -> Result<Self> {
        enable_raw_mode()?;
        // The support query round-trips through stdin, so it must run before
        // the input thread takes over event reading.
        let enhanced = terminal::supports_keyboard_enhancement().unwrap_or(false);
        let mut out = stdout();
        execute!(out, EnableBracketedPaste)?;
        if enhanced {
            execute!(
                out,
                PushKeyboardEnhancementFlags(KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES)
            )?;
        }
        Ok(Self { enhanced })
    }
}

impl Drop for TermGuard {
    fn drop(&mut self) {
        restore_terminal(self.enhanced);
    }
}

fn restore_terminal(enhanced: bool) {
    let mut out = stdout();
    if enhanced {
        let _ = execute!(out, PopKeyboardEnhancementFlags);
    }
    let _ = execute!(out, DisableBracketedPaste, cursor::Show);
    let _ = disable_raw_mode();
}

// ---------------------------------------------------------------------------
// Input thread

enum Input {
    /// A key event, plus whether it arrived in an input flood — another event
    /// already queued behind it, or the previous event less than a keystroke
    /// apart. Tells a pasted Enter from a typed one on terminals without
    /// bracketed paste (Terminal.app).
    Key(KeyEvent, bool),
    Paste(String),
    Resize,
}

/// Events closer together than this are a paste flood, not typing. Fast
/// typists bottom out around 15-30ms between keystrokes; pasted characters
/// arrive back-to-back in well under a millisecond.
const FLOOD_GAP: Duration = Duration::from_millis(8);

/// Read terminal events on a dedicated thread so Ctrl-C is seen even while
/// the main thread is inside a forward pass. While `generating` is set,
/// Ctrl-C/Esc request cancellation instead of being forwarded; everything
/// else is queued for the editor (type-ahead).
fn spawn_input_thread(
    generating: Arc<AtomicBool>,
    cancel: Arc<AtomicBool>,
    enhanced: bool,
) -> Receiver<Input> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut last_cancel: Option<Instant> = None;
        let mut last_event: Option<Instant> = None;
        loop {
            let ev = match event::read() {
                Ok(ev) => ev,
                Err(_) => return,
            };
            let flood = last_event.is_some_and(|t| t.elapsed() < FLOOD_GAP)
                || event::poll(Duration::ZERO).unwrap_or(false);
            last_event = Some(Instant::now());
            let forward = match ev {
                Event::Key(key) if is_press(&key, KeyCode::Char('c'), KeyModifiers::CONTROL) => {
                    if generating.load(Ordering::SeqCst) {
                        // First press asks generate() to stop at its next poll.
                        // If a previous request is still pending ≥1s later, the
                        // model is wedged: restore the terminal and force-quit.
                        let pending = cancel.swap(true, Ordering::SeqCst);
                        if pending
                            && last_cancel.is_some_and(|t| t.elapsed() >= Duration::from_secs(1))
                        {
                            restore_terminal(enhanced);
                            eprintln!("\nlaguna: force quit");
                            std::process::exit(130);
                        }
                        last_cancel = Some(Instant::now());
                        None
                    } else {
                        Some(Input::Key(key, flood))
                    }
                }
                Event::Key(key) if is_press(&key, KeyCode::Esc, KeyModifiers::NONE) => {
                    if generating.load(Ordering::SeqCst) {
                        cancel.store(true, Ordering::SeqCst);
                        None
                    } else {
                        Some(Input::Key(key, flood))
                    }
                }
                Event::Key(key) => Some(Input::Key(key, flood)),
                Event::Paste(text) => Some(Input::Paste(text)),
                Event::Resize(..) => Some(Input::Resize),
                _ => None,
            };
            if let Some(input) = forward
                && tx.send(input).is_err()
            {
                return;
            }
        }
    });
    rx
}

fn is_press(key: &KeyEvent, code: KeyCode, modifiers: KeyModifiers) -> bool {
    key.kind == KeyEventKind::Press && key.code == code && key.modifiers == modifiers
}

// ---------------------------------------------------------------------------
// Line editor

struct Editor {
    buf: String,
    /// Byte offset of the cursor in `buf` (always on a char boundary).
    cursor: usize,
    width: usize,
    /// Visual rows of the last redraw, and which of them held the cursor —
    /// the anchor for finding the block's top on the next redraw.
    drawn_rows: usize,
    drawn_cursor_row: usize,
}

impl Editor {
    fn new(width: usize) -> Self {
        Self { buf: String::new(), cursor: 0, width, drawn_rows: 1, drawn_cursor_row: 0 }
    }

    fn insert_str(&mut self, text: &str) {
        self.buf.insert_str(self.cursor, text);
        self.cursor += text.len();
    }

    fn set(&mut self, text: String) {
        self.cursor = text.len();
        self.buf = text;
    }

    fn clear(&mut self) {
        self.buf.clear();
        self.cursor = 0;
    }

    fn left(&mut self) {
        if let Some(ch) = self.buf[..self.cursor].chars().next_back() {
            self.cursor -= ch.len_utf8();
        }
    }

    fn right(&mut self) {
        if let Some(ch) = self.buf[self.cursor..].chars().next() {
            self.cursor += ch.len_utf8();
        }
    }

    fn backspace(&mut self) {
        if let Some(ch) = self.buf[..self.cursor].chars().next_back() {
            let start = self.cursor - ch.len_utf8();
            self.buf.replace_range(start..self.cursor, "");
            self.cursor = start;
        }
    }

    fn delete(&mut self) {
        if let Some(ch) = self.buf[self.cursor..].chars().next() {
            self.buf.replace_range(self.cursor..self.cursor + ch.len_utf8(), "");
        }
    }

    /// Byte range of the logical line containing `at` (end excludes the '\n').
    fn line_bounds(&self, at: usize) -> (usize, usize) {
        let start = self.buf[..at].rfind('\n').map_or(0, |i| i + 1);
        let end = self.buf[at..].find('\n').map_or(self.buf.len(), |i| at + i);
        (start, end)
    }

    fn home(&mut self) {
        self.cursor = self.line_bounds(self.cursor).0;
    }

    fn end(&mut self) {
        self.cursor = self.line_bounds(self.cursor).1;
    }

    fn kill_to_line_end(&mut self) {
        let (_, end) = self.line_bounds(self.cursor);
        self.buf.replace_range(self.cursor..end, "");
    }

    fn kill_to_line_start(&mut self) {
        let (start, _) = self.line_bounds(self.cursor);
        self.buf.replace_range(start..self.cursor, "");
        self.cursor = start;
    }

    fn word_left_boundary(&self) -> usize {
        let mut i = self.cursor;
        while let Some(ch) = self.buf[..i].chars().next_back() {
            if !ch.is_whitespace() {
                break;
            }
            i -= ch.len_utf8();
        }
        while let Some(ch) = self.buf[..i].chars().next_back() {
            if ch.is_whitespace() {
                break;
            }
            i -= ch.len_utf8();
        }
        i
    }

    fn word_left(&mut self) {
        self.cursor = self.word_left_boundary();
    }

    fn word_right(&mut self) {
        let mut i = self.cursor;
        while let Some(ch) = self.buf[i..].chars().next() {
            if ch.is_whitespace() {
                break;
            }
            i += ch.len_utf8();
        }
        while let Some(ch) = self.buf[i..].chars().next() {
            if !ch.is_whitespace() {
                break;
            }
            i += ch.len_utf8();
        }
        self.cursor = i;
    }

    fn delete_word_back(&mut self) {
        let start = self.word_left_boundary();
        self.buf.replace_range(start..self.cursor, "");
        self.cursor = start;
    }

    /// Move the cursor to the previous/next logical line, keeping the char
    /// column. Returns false when already on the first/last line.
    fn move_line(&mut self, up: bool) -> bool {
        let (start, _) = self.line_bounds(self.cursor);
        let col = self.buf[start..self.cursor].chars().count();
        let (target_start, target_end) = if up {
            if start == 0 {
                return false;
            }
            self.line_bounds(start - 1)
        } else {
            let (_, end) = self.line_bounds(self.cursor);
            if end == self.buf.len() {
                return false;
            }
            self.line_bounds(end + 1)
        };
        self.cursor = self.buf[target_start..target_end]
            .char_indices()
            .nth(col)
            .map_or(target_end, |(i, _)| target_start + i);
        true
    }

    /// Lay the buffer out into terminal rows (prefixes included, soft-wrapped
    /// at `width` so the terminal never auto-wraps) plus the cursor's visual
    /// (row, col).
    fn layout(&self) -> (Vec<String>, usize, usize) {
        let width = self.width.max(PREFIX_W + 2);
        let mut rows: Vec<String> = Vec::new();
        let mut row = String::from(PROMPT_PREFIX);
        let mut col = PREFIX_W;
        let (mut cursor_row, mut cursor_col) = (0, PREFIX_W);
        for (i, ch) in self.buf.char_indices() {
            if i == self.cursor {
                (cursor_row, cursor_col) = (rows.len(), col);
            }
            if ch == '\n' {
                rows.push(std::mem::replace(&mut row, String::from(CONT_PREFIX)));
                col = PREFIX_W;
                continue;
            }
            let w = UnicodeWidthChar::width(ch).unwrap_or(0);
            if col + w > width {
                rows.push(std::mem::take(&mut row));
                col = 0;
            }
            row.push(ch);
            col += w;
        }
        if self.cursor >= self.buf.len() {
            (cursor_row, cursor_col) = (rows.len(), col);
        }
        rows.push(row);
        // A cursor at the exact wrap column lives at the start of the next row.
        if cursor_col >= width {
            cursor_row += 1;
            cursor_col = 0;
            if cursor_row >= rows.len() {
                rows.push(String::new());
            }
        }
        (rows, cursor_row, cursor_col)
    }

    /// Repaint the whole input block in place and park the terminal cursor at
    /// the editing position.
    fn redraw(&mut self, out: &mut Stdout) -> Result<()> {
        let (rows, cursor_row, cursor_col) = self.layout();
        queue!(out, cursor::Hide)?;
        if self.drawn_cursor_row > 0 {
            queue!(out, cursor::MoveUp(self.drawn_cursor_row as u16))?;
        }
        queue!(out, cursor::MoveToColumn(0), Clear(ClearType::FromCursorDown))?;
        for (i, row) in rows.iter().enumerate() {
            if i > 0 {
                out.write_all(b"\r\n")?;
            }
            out.write_all(row.as_bytes())?;
        }
        let below = rows.len() - 1 - cursor_row;
        if below > 0 {
            queue!(out, cursor::MoveUp(below as u16))?;
        }
        queue!(out, cursor::MoveToColumn(cursor_col as u16), cursor::Show)?;
        out.flush()?;
        self.drawn_rows = rows.len();
        self.drawn_cursor_row = cursor_row;
        Ok(())
    }

    /// Leave the input block: cursor to its last row, then a fresh line.
    fn finish(&mut self, out: &mut Stdout) -> Result<()> {
        let below = self.drawn_rows - 1 - self.drawn_cursor_row;
        if below > 0 {
            queue!(out, cursor::MoveDown(below as u16))?;
        }
        out.write_all(b"\r\n")?;
        out.flush()?;
        Ok(())
    }
}

/// Pastes keep newlines/tabs (tabs become spaces so column math stays honest);
/// other control chars are dropped.
fn normalize_paste(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.replace("\r\n", "\n").replace('\r', "\n").chars() {
        match ch {
            '\t' => out.push_str("    "),
            '\n' => out.push('\n'),
            c if c.is_control() => {}
            c => out.push(c),
        }
    }
    out
}

enum Turn {
    Submit(String),
    Quit,
}

/// Drive the editor until a submission or quit. History (Up/Down) applies
/// while the buffer is single-line; in a multi-line buffer Up/Down move
/// between lines instead.
fn read_turn(events: &Receiver<Input>, out: &mut Stdout, history: &[String]) -> Result<Turn> {
    let width = terminal::size().map_or(80, |(w, _)| w as usize);
    let mut ed = Editor::new(width);
    let mut hist_idx: Option<usize> = None;
    let mut stash = String::new();
    ed.redraw(out)?;

    loop {
        let Ok(input) = events.recv() else { return Ok(Turn::Quit) };
        match input {
            Input::Resize => {
                ed.width = terminal::size().map_or(ed.width, |(w, _)| w as usize);
            }
            Input::Paste(text) => ed.insert_str(&normalize_paste(&text)),
            Input::Key(key, flood) => {
                if !matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) {
                    continue;
                }
                let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                let alt = key.modifiers.contains(KeyModifiers::ALT);
                let shift = key.modifiers.contains(KeyModifiers::SHIFT);
                match key.code {
                    // A queued event behind an Enter means it arrived in a
                    // paste flood (no bracketed paste): insert, don't submit.
                    KeyCode::Enter if shift || alt || flood => ed.insert_str("\n"),
                    KeyCode::Enter => {
                        if ed.buf.trim().is_empty() {
                            ed.clear();
                        } else {
                            ed.finish(out)?;
                            return Ok(Turn::Submit(std::mem::take(&mut ed.buf)));
                        }
                    }
                    KeyCode::Char('j') if ctrl => ed.insert_str("\n"),
                    KeyCode::Char('c') if ctrl => {
                        if ed.buf.is_empty() {
                            ed.finish(out)?;
                            return Ok(Turn::Quit);
                        }
                        ed.clear();
                    }
                    KeyCode::Char('d') if ctrl => {
                        if ed.buf.is_empty() {
                            ed.finish(out)?;
                            return Ok(Turn::Quit);
                        }
                        ed.delete();
                    }
                    KeyCode::Char('a') if ctrl => ed.home(),
                    KeyCode::Char('e') if ctrl => ed.end(),
                    KeyCode::Char('k') if ctrl => ed.kill_to_line_end(),
                    KeyCode::Char('u') if ctrl => ed.kill_to_line_start(),
                    KeyCode::Char('w') if ctrl => ed.delete_word_back(),
                    KeyCode::Char('l') if ctrl => {
                        execute!(out, Clear(ClearType::All), cursor::MoveTo(0, 0))?;
                        ed.drawn_cursor_row = 0;
                    }
                    KeyCode::Char(c) if !ctrl => {
                        hist_idx = None;
                        ed.insert_str(c.encode_utf8(&mut [0u8; 4]));
                    }
                    KeyCode::Backspace if alt => ed.delete_word_back(),
                    KeyCode::Backspace => ed.backspace(),
                    KeyCode::Delete => ed.delete(),
                    KeyCode::Left if alt || ctrl => ed.word_left(),
                    KeyCode::Right if alt || ctrl => ed.word_right(),
                    KeyCode::Left => ed.left(),
                    KeyCode::Right => ed.right(),
                    KeyCode::Home => ed.home(),
                    KeyCode::End => ed.end(),
                    KeyCode::Tab => ed.insert_str("    "),
                    KeyCode::Up => {
                        let single_line = !ed.buf.contains('\n');
                        if !ed.move_line(true) && single_line {
                            let prev = match hist_idx {
                                None if !history.is_empty() => Some(history.len() - 1),
                                Some(i) if i > 0 => Some(i - 1),
                                _ => None,
                            };
                            if let Some(i) = prev {
                                if hist_idx.is_none() {
                                    stash = std::mem::take(&mut ed.buf);
                                }
                                hist_idx = Some(i);
                                ed.set(history[i].clone());
                            }
                        }
                    }
                    KeyCode::Down if !ed.move_line(false) => match hist_idx {
                        Some(i) if i + 1 < history.len() => {
                            hist_idx = Some(i + 1);
                            ed.set(history[i + 1].clone());
                        }
                        Some(_) => {
                            hist_idx = None;
                            ed.set(std::mem::take(&mut stash));
                        }
                        None => {}
                    },
                    _ => {}
                }
            }
        }
        ed.redraw(out)?;
    }
}

// ---------------------------------------------------------------------------
// Streaming a reply

/// Raw mode disables output post-processing, so '\n' alone doesn't return the
/// carriage — translate while streaming.
fn write_stream(out: &mut Stdout, text: &str) {
    let _ = out.write_all(text.replace('\n', "\r\n").as_bytes());
}

fn write_stream_dim(out: &mut Stdout, text: &str) {
    let _ = write!(out, "{DIM}");
    write_stream(out, text);
    let _ = write!(out, "{RESET}");
}

/// Stream one reply: a dim pending indicator during prefill (and, with
/// reasoning hidden, a live "thinking" counter), reasoning dimmed when shown,
/// and the accumulated raw text returned for the conversation state.
fn stream_reply(
    generator: &mut Generator,
    prompt: &str,
    max_tokens: usize,
    show_thinking: bool,
    cancel: &AtomicBool,
    out: &mut Stdout,
) -> Result<(String, GenStats)> {
    write!(out, "{DIM}\u{2026}{RESET}")?;
    out.flush()?;

    let mut full = String::new();
    let mut in_think = true;
    // Whether the pending/thinking indicator line has been cleared yet.
    let mut started = false;
    let mut think_chunks = 0usize;
    let stats = {
        let sink = &mut *out;
        generator.generate(
            prompt,
            max_tokens,
            &mut |chunk| {
                full.push_str(chunk);
                if !in_think {
                    write_stream(sink, chunk);
                    let _ = sink.flush();
                    return;
                }
                // `</think>` is a single token, so it never splits across chunks.
                match chunk.split_once("</think>") {
                    Some((think, rest)) => {
                        in_think = false;
                        if show_thinking {
                            if !started {
                                let _ = sink.write_all(b"\r\x1b[K");
                            }
                            write_stream_dim(sink, think);
                            write_stream(sink, rest);
                        } else {
                            let _ = sink.write_all(b"\r\x1b[K");
                            write_stream(sink, rest.trim_start());
                        }
                        started = true;
                    }
                    None => {
                        if show_thinking {
                            if !started {
                                let _ = sink.write_all(b"\r\x1b[K");
                                started = true;
                            }
                            write_stream_dim(sink, chunk);
                        } else {
                            think_chunks += 1;
                            let _ = write!(
                                sink,
                                "\r\x1b[K{DIM}thinking\u{2026} {think_chunks} tok{RESET}"
                            );
                        }
                    }
                }
                let _ = sink.flush();
            },
            &mut || cancel.load(Ordering::SeqCst),
        )?
    };

    if !started {
        write!(out, "\r\x1b[K")?;
    }
    if stats.cancelled {
        write!(out, "{DIM}[cancelled]{RESET}")?;
    }
    write!(out, "\r\n")?;
    out.flush()?;
    Ok((full, stats))
}

/// Split a reply into (reasoning, content) at the first `</think>`. With no
/// marker the whole reply is content.
fn split_thinking(full: &str) -> (&str, &str) {
    match full.split_once("</think>") {
        Some((reasoning, content)) => (reasoning.trim(), content.trim()),
        None => ("", full.trim()),
    }
}

// ---------------------------------------------------------------------------
// Chrome

fn banner(out: &mut Stdout, max_ctx: usize, enhanced: bool) -> Result<()> {
    let newline_key = if enhanced { "Shift+Enter" } else { "Alt+Enter or Ctrl+J" };
    write!(
        out,
        "Laguna chat \u{b7} ctx {max_ctx}\r\n\
         {DIM}Enter sends \u{b7} {newline_key} inserts a newline \u{b7} paste is multi-line safe\r\n\
         Ctrl+C clears / cancels generation / quits on an empty prompt \u{b7} /help for commands{RESET}\r\n\r\n",
    )?;
    out.flush()?;
    Ok(())
}

fn help(out: &mut Stdout) -> Result<()> {
    write!(
        out,
        "{DIM}/clear    reset the conversation\r\n\
         /think    toggle showing <think> reasoning\r\n\
         exit      quit (also Ctrl+D, or Ctrl+C on an empty prompt)\r\n\
         while generating: Ctrl+C or Esc cancels; Ctrl+C twice force-quits{RESET}\r\n\r\n",
    )?;
    out.flush()?;
    Ok(())
}

fn dim_line(out: &mut Stdout, text: &str) -> Result<()> {
    write!(out, "{DIM}{text}{RESET}\r\n\r\n")?;
    out.flush()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Non-TTY fallback (piped stdin/stdout, e.g. scripted smoke tests)

/// Plain line-per-message loop, one reply per line, no terminal control.
fn pipe_repl(generator: &mut Generator, max_tokens: usize, show_thinking: bool) -> Result<()> {
    use std::io::BufRead;

    let opts = ChatOptions { enable_thinking: true };
    let mut messages: Vec<Message> = Vec::new();
    let stdin = std::io::stdin();
    let mut out = stdout();

    loop {
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            break;
        }
        let line = line.trim_end_matches('\n').to_string();
        if line.trim() == "exit" {
            break;
        }
        if line.trim().is_empty() {
            continue;
        }

        messages.push(Message::User(line));
        let prompt = build_prompt(&messages, &opts)?;

        let mut full = String::new();
        let mut in_think = true;
        generator.generate(
            &prompt,
            max_tokens,
            &mut |chunk| {
                full.push_str(chunk);
                if !in_think {
                    print!("{chunk}");
                } else {
                    match chunk.split_once("</think>") {
                        Some((think, rest)) => {
                            if show_thinking {
                                print!("{DIM}{think}{RESET}");
                            }
                            in_think = false;
                            print!("{rest}");
                        }
                        None => {
                            if show_thinking {
                                print!("{DIM}{chunk}{RESET}");
                            }
                        }
                    }
                }
                let _ = out.flush();
            },
            &mut || false,
        )?;
        println!();

        let (reasoning, content) = split_thinking(&full);
        messages.push(Message::Assistant {
            content: content.to_string(),
            reasoning: if reasoning.is_empty() { None } else { Some(reasoning.to_string()) },
        });
    }
    Ok(())
}
