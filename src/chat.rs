/// Hand-rolled prompt builder for Laguna's GLM-style template
/// (reference/chat_template.jinja): <system>/<user>/<assistant> turns,
/// <think> reasoning blocks (preserved across turns when thinking is on),
/// <tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>.
/// Rendered text must match `llama-server --jinja` byte-for-byte (fixtures).
use anyhow::Result;

/// BOS token text the template emits verbatim (tokenizer added-token id 2).
/// The angle brackets are U+3008/U+3009 (CJK ANGLE BRACKET), which look like but
/// are distinct from U+2329/U+232A; they match tokenizer.json's id-2 content so
/// the tokenizer maps this literal back to a single BOS id.
const BOS: &str = "\u{3008}|EOS|\u{3009}";

/// Default system message (chat_template.jinja line 9), used whenever the caller
/// supplies no leading system message.
const DEFAULT_SYSTEM: &str = "You are a helpful, conversationally-fluent assistant made by Poolside. You are here to be helpful to users through natural language conversations.";

#[derive(Debug, Clone)]
pub enum Message {
    System(String),
    User(String),
    Assistant { content: String, reasoning: Option<String> },
    ToolResponse(String),
}

#[derive(Debug, Clone)]
pub struct ChatOptions {
    pub enable_thinking: bool,
}

impl Default for ChatOptions {
    fn default() -> Self {
        Self { enable_thinking: true }
    }
}

/// Render a conversation into the exact prompt string, ending with the
/// generation prompt (`<assistant><think>` or `<assistant></think>`).
///
/// Faithful to reference/chat_template.jinja with `add_generation_prompt=true`.
/// Tool support (the `tools` list and assistant `tool_calls`) is absent from the
/// `Message` model, so the tool-conditioned branches of the header and main loop
/// are not rendered; every other branch is reproduced verbatim.
pub fn build_prompt(messages: &[Message], opts: &ChatOptions) -> Result<String> {
    let thinking = opts.enable_thinking;
    let mut out = String::new();

    // Line 3: the template always opens with the BOS text.
    out.push_str(BOS);

    // Header (lines 7-36). A leading system message overrides the default and is
    // consumed here; the remaining messages are rendered by the main loop.
    let (system_message, body) = match messages.first() {
        Some(Message::System(content)) => (content.as_str(), &messages[1..]),
        _ => (DEFAULT_SYSTEM, messages),
    };

    // Line 15: `has_sys` tests the message stripped of both-end whitespace.
    let has_sys = !system_message.trim().is_empty();
    // Line 16: without tools or a tool list, the block appears when there is a
    // real system message or when thinking is enabled (an empty leading system
    // message with thinking off therefore emits no <system> block at all).
    if has_sys || thinking {
        out.push_str("<system>");
        if has_sys {
            // Line 20: content is right-stripped (leading whitespace preserved).
            out.push_str(system_message.trim_end());
        }
        out.push_str("</system>\n");
    }

    // Main loop (lines 39-83).
    for message in body {
        match message {
            Message::User(content) => {
                // Line 42.
                out.push_str("<user>");
                out.push_str(content);
                out.push_str("</user>\n");
            }
            Message::Assistant { content, reasoning } => {
                // Lines 45-75.
                out.push_str("<assistant>");
                // Lines 54-58: with thinking on, the (possibly empty) reasoning is
                // wrapped in <think>…</think>; with thinking off, a bare </think>
                // is emitted and any reasoning is dropped.
                if thinking {
                    out.push_str("<think>");
                    out.push_str(reasoning.as_deref().unwrap_or(""));
                    out.push_str("</think>");
                } else {
                    out.push_str("</think>");
                }
                // Lines 60-62: main content follows the reasoning.
                out.push_str(content);
                out.push_str("</assistant>\n");
            }
            Message::ToolResponse(content) => {
                // Line 78.
                out.push_str("<tool_response>");
                out.push_str(content);
                out.push_str("</tool_response>\n");
            }
            Message::System(content) => {
                // Lines 79-81: a system message past the first is rendered inline.
                out.push_str("<system>");
                out.push_str(content);
                out.push_str("</system>\n");
            }
        }
    }

    // Generation prompt (lines 84-93), always appended by this builder.
    out.push_str("<assistant>");
    out.push_str(if thinking { "<think>" } else { "</think>" });

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn thinking(on: bool) -> ChatOptions {
        ChatOptions { enable_thinking: on }
    }

    // (a) system + user, thinking on: header with the supplied system message,
    // one user turn, generation prompt opening a <think> block.
    #[test]
    fn system_and_user_thinking_on() {
        let msgs = [Message::System("You are a pirate.".into()), Message::User("Hi".into())];
        let expected =
            "\u{3008}|EOS|\u{3009}<system>You are a pirate.</system>\n<user>Hi</user>\n<assistant><think>";
        assert_eq!(build_prompt(&msgs, &thinking(true)).unwrap(), expected);
    }

    // (b) no system message: the default system message fills the header.
    #[test]
    fn no_system_message_uses_default() {
        let msgs = [Message::User("Hello".into())];
        let expected = format!(
            "\u{3008}|EOS|\u{3009}<system>{DEFAULT_SYSTEM}</system>\n<user>Hello</user>\n<assistant><think>"
        );
        assert_eq!(build_prompt(&msgs, &thinking(true)).unwrap(), expected);
    }

    // (c) empty leading system message + thinking off: the <system> block is
    // suppressed entirely (jinja line 16 — none of has_sys/tools/thinking hold).
    #[test]
    fn empty_system_message_opts_out() {
        let msgs = [Message::System(String::new()), Message::User("Hey".into())];
        let expected = "\u{3008}|EOS|\u{3009}<user>Hey</user>\n<assistant></think>";
        assert_eq!(build_prompt(&msgs, &thinking(false)).unwrap(), expected);
    }

    // (d) multi-turn, thinking on: a prior assistant turn keeps its reasoning
    // inside <think>…</think> ahead of its content.
    #[test]
    fn multi_turn_reasoning_preserved() {
        let msgs = [
            Message::System("You are a pirate.".into()),
            Message::User("2+2?".into()),
            Message::Assistant {
                content: "4".into(),
                reasoning: Some("The user asks arithmetic.".into()),
            },
            Message::User("thanks".into()),
        ];
        let expected = "\u{3008}|EOS|\u{3009}<system>You are a pirate.</system>\n\
                        <user>2+2?</user>\n\
                        <assistant><think>The user asks arithmetic.</think>4</assistant>\n\
                        <user>thanks</user>\n\
                        <assistant><think>";
        assert_eq!(build_prompt(&msgs, &thinking(true)).unwrap(), expected);
    }

    // (e) same conversation, thinking off: the assistant turn emits a bare
    // </think> and its reasoning is dropped, matching the header staying present.
    #[test]
    fn multi_turn_thinking_off_drops_reasoning() {
        let msgs = [
            Message::System("You are a pirate.".into()),
            Message::User("2+2?".into()),
            Message::Assistant {
                content: "4".into(),
                reasoning: Some("The user asks arithmetic.".into()),
            },
            Message::User("thanks".into()),
        ];
        let expected = "\u{3008}|EOS|\u{3009}<system>You are a pirate.</system>\n\
                        <user>2+2?</user>\n\
                        <assistant></think>4</assistant>\n\
                        <user>thanks</user>\n\
                        <assistant></think>";
        assert_eq!(build_prompt(&msgs, &thinking(false)).unwrap(), expected);
    }

    // (f) the trailing generation prompt differs only by the think tag.
    #[test]
    fn generation_prompt_suffix_both_modes() {
        let msgs = [Message::User("x".into())];
        assert!(build_prompt(&msgs, &thinking(true)).unwrap().ends_with("<assistant><think>"));
        assert!(build_prompt(&msgs, &thinking(false)).unwrap().ends_with("<assistant></think>"));
    }

    // A tool response turn renders between the <tool_response> markers.
    #[test]
    fn tool_response_turn() {
        let msgs = [
            Message::System(String::new()),
            Message::User("weather?".into()),
            Message::ToolResponse("sunny".into()),
        ];
        let expected = "\u{3008}|EOS|\u{3009}<system></system>\n\
                        <user>weather?</user>\n\
                        <tool_response>sunny</tool_response>\n\
                        <assistant><think>";
        assert_eq!(build_prompt(&msgs, &thinking(true)).unwrap(), expected);
    }
}
