// Byte-exact validation of src/chat.rs `build_prompt` fixtures against the
// llama.cpp-laguna fork's jinja renderer via `llama-server`'s /apply-template.
//
// The expected strings below are the exact fixture literals asserted in
// src/chat.rs tests (which already pass, so expected == build_prompt output).
// A match here therefore proves build_prompt == the fork's renderer.
//
// Prereq: llama-server running with --jinja on 127.0.0.1:8087, serving a model
// whose embedded chat template is byte-identical to reference/chat_template.jinja
// (verified separately via /props).

const BASE = "http://127.0.0.1:8087";
const BOS = "\u{3008}|EOS|\u{3009}";
const DEFAULT_SYSTEM =
  "You are a helpful, conversationally-fluent assistant made by Poolside. You are here to be helpful to users through natural language conversations.";

type Case = {
  name: string;
  messages: any[];
  enable_thinking: boolean;
  expected: string;
};

const cases: Case[] = [
  {
    name: "(a) system+user, thinking on",
    messages: [
      { role: "system", content: "You are a pirate." },
      { role: "user", content: "Hi" },
    ],
    enable_thinking: true,
    expected: `${BOS}<system>You are a pirate.</system>\n<user>Hi</user>\n<assistant><think>`,
  },
  {
    name: "(b) no system -> default, thinking on",
    messages: [{ role: "user", content: "Hello" }],
    enable_thinking: true,
    expected: `${BOS}<system>${DEFAULT_SYSTEM}</system>\n<user>Hello</user>\n<assistant><think>`,
  },
  {
    name: "(c) empty system opt-out, thinking off",
    messages: [
      { role: "system", content: "" },
      { role: "user", content: "Hey" },
    ],
    enable_thinking: false,
    expected: `${BOS}<user>Hey</user>\n<assistant></think>`,
  },
  {
    name: "(d) multi-turn, reasoning preserved, thinking on",
    messages: [
      { role: "system", content: "You are a pirate." },
      { role: "user", content: "2+2?" },
      { role: "assistant", content: "4", reasoning_content: "The user asks arithmetic." },
      { role: "user", content: "thanks" },
    ],
    enable_thinking: true,
    expected:
      `${BOS}<system>You are a pirate.</system>\n` +
      `<user>2+2?</user>\n` +
      `<assistant><think>The user asks arithmetic.</think>4</assistant>\n` +
      `<user>thanks</user>\n` +
      `<assistant><think>`,
  },
  {
    name: "(e) multi-turn, thinking off, reasoning dropped",
    messages: [
      { role: "system", content: "You are a pirate." },
      { role: "user", content: "2+2?" },
      { role: "assistant", content: "4", reasoning_content: "The user asks arithmetic." },
      { role: "user", content: "thanks" },
    ],
    enable_thinking: false,
    expected:
      `${BOS}<system>You are a pirate.</system>\n` +
      `<user>2+2?</user>\n` +
      `<assistant></think>4</assistant>\n` +
      `<user>thanks</user>\n` +
      `<assistant></think>`,
  },
  {
    name: "(tool) tool_response turn, thinking on",
    messages: [
      { role: "system", content: "" },
      { role: "user", content: "weather?" },
      { role: "tool", content: "sunny" },
    ],
    enable_thinking: true,
    expected:
      `${BOS}<system></system>\n` +
      `<user>weather?</user>\n` +
      `<tool_response>sunny</tool_response>\n` +
      `<assistant><think>`,
  },
];

function hexView(s: string): string {
  return [...s].map((c) => c.codePointAt(0)!.toString(16).padStart(4, "0")).join(" ");
}

function firstDiff(a: string, b: string): number {
  const n = Math.min(a.length, b.length);
  let i = 0;
  for (; i < n && a[i] === b[i]; i++);
  return i;
}

async function render(messages: any[], enable_thinking: boolean): Promise<string> {
  const res = await fetch(`${BASE}/apply-template`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      messages,
      add_generation_prompt: true,
      chat_template_kwargs: { enable_thinking },
    }),
  });
  if (!res.ok) throw new Error(`/apply-template ${res.status}: ${await res.text()}`);
  const j = await res.json();
  return j.prompt;
}

// The fork's /apply-template returns the template BODY without the leading BOS:
// llama.cpp strips the BOS the jinja emits on line 3 and re-adds token id 2 at
// tokenize time (add_bos_token / bos_token = 〈|EOS|〉, verified via llama-tokenize).
// build_prompt keeps the BOS in the string (matching the raw jinja / HF
// apply_chat_template) and relies on encode(add_special_tokens=false) to map it
// to a single id 2 without doubling. The two paths therefore produce the same
// token stream; the byte-exact invariant we check is: build_prompt == BOS + body.
let pass = 0;
let fail = 0;
for (const c of cases) {
  const got = await render(c.messages, c.enable_thinking);
  const expectedBody = c.expected.startsWith(BOS) ? c.expected.slice(BOS.length) : c.expected;
  const bosOk = c.expected.startsWith(BOS);
  if (bosOk && got === expectedBody) {
    console.log(`PASS ${c.name}`);
    pass++;
  } else {
    fail++;
    console.log(`FAIL ${c.name}`);
    if (!bosOk) console.log(`  build_prompt fixture does not start with the BOS token text`);
    const i = firstDiff(got, expectedBody);
    console.log(`  body divergence at index ${i} (got len ${got.length}, expected-body len ${expectedBody.length})`);
    console.log(`  expected-body around: ${JSON.stringify(expectedBody.slice(Math.max(0, i - 30), i + 30))}`);
    console.log(`  got around:           ${JSON.stringify(got.slice(Math.max(0, i - 30), i + 30))}`);
    console.log(`  expected-body hex: ${hexView(expectedBody.slice(Math.max(0, i - 8), i + 8))}`);
    console.log(`  got hex:           ${hexView(got.slice(Math.max(0, i - 8), i + 8))}`);
    console.log(`  --- full expected (with BOS) ---\n${JSON.stringify(c.expected)}`);
    console.log(`  --- full got (body, no BOS) ---\n${JSON.stringify(got)}`);
  }
}
console.log(`\n${pass}/${cases.length} body byte-exact (BOS handled at tokenize-time), ${fail} mismatch`);
process.exit(fail === 0 ? 0 : 1);
