# Native Google Antigravity provider

ARTEX has an experimental first-class `antigravity` LLM profile format. It delegates each model request to Google's official `agy` CLI and does not implement Google's OAuth protocol or read OAuth credential files.

## Server prerequisites

1. Install the official Antigravity CLI (`agy`) on the machine/container that runs ARTEX.
2. Authenticate that CLI as the operating-system account used by the ARTEX process. Run `agy` interactively and complete Google's sign-in before starting ARTEX.
3. Verify the session with `agy models`. Copy an exact model ID from that command into the ARTEX profile.
4. Create an LLM profile with format **Google Antigravity (agy CLI, OAuth)**, enter the exact model ID, and activate the profile.

If ARTEX runs in Docker, the CLI and its authenticated account must be available inside the ARTEX container/runtime. Signing in on the host does not automatically authenticate a container with a different HOME or credential store.

The provider checks `ANTIGRAVITY_COMMAND`, `AGY_CLI_PATH`, and `ANTIGRAVITY_CLI_PATH`, then PATH and common user-local install locations. Set one of those variables if the executable is installed elsewhere.

## Security and execution model

- ARTEX invokes `agy` as a child process with a bounded request timeout.
- The CLI owns Google authentication. ARTEX does not parse, copy, print, or persist OAuth tokens.
- The prompt explicitly asks the model to request ARTEX tools using `<tool_call>` JSON blocks. ARTEX parses those blocks into Norma tool-call events.
- If the CLI reports a native tool execution step, ARTEX terminates the process and fails closed. The CLI must not execute host tools on behalf of ARTEX.
- Model IDs are passed through unchanged. A separate `--effort` flag is used only for suffixless Gemini model IDs and supported effort values.

## Current limitations

- Model discovery is manual for now: run `agy models` on the ARTEX server and paste the exact ID. The ARTEX model-list endpoint does not yet proxy `agy models`.
- The current adapter starts a one-shot CLI process per completion. It replays conversation history in the prompt; it does not keep a persistent agy worker session.
- Norma's request-level output-token cap, temperature, stop sequences, and provider-specific thinking toggle are not mapped to agy CLI flags. Profile HTTP proxy, session-header, and custom User-Agent settings also do not apply to the local CLI process.
- The CLI runs under the ARTEX service account's normal HOME, so agy's own session/config files remain under that account's Antigravity CLI data directory. ARTEX itself does not read or copy OAuth token contents.
- The ARTEX profile's rate limits are enforced between CLI process starts; they do not account for Google-side quota or other agy processes.
- Live integration tests require an authenticated Google Antigravity account and an installed `agy` binary. Unit tests cover prompt serialization and model/effort argument mapping without contacting Google.

This provider is experimental until CI and an authenticated end-to-end run pass.
