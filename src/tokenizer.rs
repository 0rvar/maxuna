use std::path::Path;

use anyhow::{Result, anyhow};

/// Wrapper around the HF `tokenizers` runtime for the Laguna BPE vocab.
/// Special ids: BOS 2 (`〈|EOS|〉`), EOG {2, 24} where 24 = `</assistant>`, pad 9.
///
/// The chat template (see `chat.rs`) emits every structural marker as literal
/// text, including the leading BOS `〈|EOS|〉` and `<assistant>` / `</think>` etc.
/// Some of those strings are entries in the tokenizer's added-vocabulary (BOS is
/// id 2, `<assistant>` id 23, `</assistant>` id 24, `<think>`/`</think>` 18/19,
/// `<tool_call>`/`</tool_call>` 25/26); the rest (`<system>`, `<user>`, …) are not
/// added tokens and fall through to ordinary byte-level BPE. Prompt text produced
/// by `chat.rs` must therefore be encoded so those literal added-token strings map
/// to their single ids — which `encode` does — while NOT letting the tokenizer's
/// post-processor prepend its own BOS on top of the one the template already wrote.
pub struct LagunaTokenizer {
    inner: tokenizers::Tokenizer,
}

impl LagunaTokenizer {
    /// Beginning-of-sequence token (`〈|EOS|〉`), doubling as end-of-sequence.
    pub const BOS: u32 = 2;
    /// Padding token (`〈|PAD|〉`).
    pub const PAD: u32 = 9;
    /// End-of-generation tokens: `〈|EOS|〉` (2) and `</assistant>` (24).
    pub const EOG: [u32; 2] = [2, 24];

    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path.as_ref())
            .map_err(|e| anyhow!("failed to load tokenizer from {:?}: {e}", path.as_ref()))?;
        Ok(Self { inner })
    }

    /// Load the vocab embedded in the GGUF metadata (no tokenizer.json needed).
    pub fn from_gguf(content: &candle_core::quantized::gguf_file::Content) -> Result<Self> {
        let _ = content;
        // Reconstructing the byte-level BPE model (merges, byte-level pre-tokenizer
        // and decoder, and the 70-entry added vocabulary) from the flat
        // `tokenizer.ggml.*` arrays would duplicate a large slice of the
        // `tokenizers` builders for no parity benefit while `tokenizer.json` ships
        // alongside every checkpoint. Use `--tokenizer <tokenizer.json>` instead.
        Err(anyhow!(
            "building the tokenizer from GGUF metadata is not supported; \
             pass the tokenizer.json path via --tokenizer"
        ))
    }

    /// Encode prompt text into token ids.
    ///
    /// `add_special_tokens` is `false`: the added-vocabulary matcher still maps any
    /// literal added-token string in the text to its single id (this is gated by
    /// `encode_special_tokens`, left at its default `false`, not by this flag), so
    /// `〈|EOS|〉`, `<assistant>`, `</think>`, … resolve to ids 2/23/19/…. Passing
    /// `false` only suppresses the `TemplateProcessing` post-processor, which would
    /// otherwise prepend a second BOS on top of the one `chat.rs` already emitted.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| anyhow!("encode failed: {e}"))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode ids back to text. Special tokens are rendered verbatim (lossless):
    /// the generation loop stops on an EOG before it would decode one, and the
    /// structural markers callers care about (`<think>` etc.) are non-special
    /// added tokens that render as their literal text regardless.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner.decode(ids, false).map_err(|e| anyhow!("decode failed: {e}"))
    }

    /// The literal string for a token id, or `None` if out of range.
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }

    /// Incremental decoder for streaming (handles multi-token UTF-8).
    pub fn decode_stream(&self) -> DecodeStream<'_> {
        DecodeStream { tokenizer: self, ids: Vec::new(), prefix: String::new(), prefix_index: 0 }
    }
}

/// Streaming, UTF-8-safe decoder.
///
/// Mirrors the prefix-diff algorithm of `tokenizers::DecodeStream`: it keeps a
/// rolling suffix of ids around the last emitted `prefix` so a decode of the buffer
/// always reproduces `prefix` as a leading substring; the freshly finalized text is
/// whatever the current decode adds beyond `prefix`. Bytes that would land mid
/// UTF-8 sequence decode to the replacement char `U+FFFD` and are withheld (return
/// `None`) until a later token completes them.
pub struct DecodeStream<'a> {
    tokenizer: &'a LagunaTokenizer,
    ids: Vec<u32>,
    prefix: String,
    prefix_index: usize,
}

impl DecodeStream<'_> {
    /// Feed one token; returns text newly finalized by it, if any.
    pub fn step(&mut self, id: u32) -> Result<Option<String>> {
        if self.prefix.is_empty() && !self.ids.is_empty() {
            let new_prefix = self.tokenizer.decode(&self.ids)?;
            if !new_prefix.ends_with('\u{fffd}') {
                self.prefix = new_prefix;
                self.prefix_index = self.ids.len();
            }
        }

        self.ids.push(id);
        let string = self.tokenizer.decode(&self.ids)?;
        if string.len() > self.prefix.len() && !string.ends_with('\u{fffd}') {
            if !string.starts_with(&self.prefix) {
                return Err(anyhow!(
                    "streaming decode produced {string:?}, which does not extend prefix {:?}",
                    self.prefix
                ));
            }
            let new_text = string[self.prefix.len()..].to_string();
            let new_prefix_index = self.ids.len() - self.prefix_index;
            self.ids = self.ids.split_off(self.prefix_index);
            self.prefix = self.tokenizer.decode(&self.ids)?;
            self.prefix_index = new_prefix_index;
            Ok(Some(new_text))
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokenizer() -> LagunaTokenizer {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/reference/tokenizer.json");
        LagunaTokenizer::from_file(path).expect("load reference tokenizer")
    }

    #[test]
    fn special_ids_map_to_expected_strings() {
        let t = tokenizer();
        assert_eq!(t.id_to_token(2).as_deref(), Some("\u{3008}|EOS|\u{3009}"));
        assert_eq!(t.id_to_token(9).as_deref(), Some("\u{3008}|PAD|\u{3009}"));
        assert_eq!(t.id_to_token(24).as_deref(), Some("</assistant>"));
    }

    #[test]
    fn bos_literal_encodes_to_single_id_without_post_processor_doubling() {
        let t = tokenizer();
        // The template writes the BOS as literal text; encoding it must yield the
        // single added-token id 2, not a doubled [2, 2] from the post-processor.
        assert_eq!(t.encode("\u{3008}|EOS|\u{3009}").unwrap(), vec![2]);
    }

    #[test]
    fn structural_markers_map_to_added_token_ids() {
        let t = tokenizer();
        // <assistant>=23, <think>=18, </think>=19, </assistant>=24 are added tokens.
        let ids = t.encode("<assistant><think></think></assistant>").unwrap();
        assert_eq!(ids, vec![23, 18, 19, 24]);
    }

    #[test]
    fn decode_roundtrips_plain_text() {
        let t = tokenizer();
        let ids = t.encode("Hello, world!").unwrap();
        assert_eq!(t.decode(&ids).unwrap(), "Hello, world!");
    }

    #[test]
    fn decode_stream_matches_whole_decode() {
        let t = tokenizer();
        let ids = t.encode("The quick brown fox jumps over the lazy dog.").unwrap();
        let mut stream = t.decode_stream();
        let mut streamed = String::new();
        for &id in &ids {
            if let Some(chunk) = stream.step(id).unwrap() {
                streamed.push_str(&chunk);
            }
        }
        assert_eq!(streamed, t.decode(&ids).unwrap());
    }

    #[test]
    fn decode_stream_withholds_partial_utf8() {
        let t = tokenizer();
        // A multi-byte grapheme ("界", U+754C) generally spans several byte-level
        // tokens; the stream must never emit a lone replacement char and, summed,
        // must reproduce the full decode.
        let ids = t.encode("世界").unwrap();
        let mut stream = t.decode_stream();
        let mut streamed = String::new();
        for &id in &ids {
            if let Some(chunk) = stream.step(id).unwrap() {
                assert!(!chunk.contains('\u{fffd}'), "emitted a partial-UTF8 chunk: {chunk:?}");
                streamed.push_str(&chunk);
            }
        }
        assert_eq!(streamed, "世界");
    }
}
