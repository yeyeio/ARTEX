package agent

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/Autumn-27/norma/llm"
)

func TestAntigravityArgsUseExactModelAndSupportedEffort(t *testing.T) {
	tests := []struct {
		name, model, effort string
		wantEffort bool
	}{
		{"gemini suffixless", "gemini-3.8-flash", "high", true},
		{"native effort suffix", "gemini-3.8-flash-medium", "high", false},
		{"claude model", "claude-sonnet-4", "high", false},
		{"gpt model", "gpt-oss-120b", "medium", false},
		{"invalid effort", "gemini-3.8-pro", "extreme", false},
		{"none maps to low", "gemini-3.8-pro", "none", true},
		{"minimal maps to low", "gemini-3.8-pro", "minimal", true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &antigravityProvider{model: tt.model, effort: tt.effort}
			args := strings.Join(p.args(), " ")
			if !strings.Contains(args, "--model "+tt.model) {
				t.Fatalf("exact model ID missing from args: %q", args)
			}
			got := strings.Contains(args, "--effort ")
			if got != tt.wantEffort {
				t.Fatalf("effort flag present=%t, want %t: %q", got, tt.wantEffort, args)
			}
			if tt.effort == "none" || tt.effort == "minimal" {
				if !strings.Contains(args, "--effort low") { t.Fatalf("effort was not normalized to low: %q", args) }
			}
		})
	}
}

func fakeAgy(t *testing.T, events ...string) string {
	t.Helper()
	if runtime.GOOS == "windows" { t.Skip("fake shell CLI test is POSIX-only") }
	path := filepath.Join(t.TempDir(), "agy")
	var script strings.Builder
	script.WriteString("#!/bin/sh\ncat >/dev/null\n")
	for _, event := range events {
		script.WriteString("printf '%s\\n' '")
		script.WriteString(strings.ReplaceAll(event, "'", "'\\''"))
		script.WriteString("'\n")
	}
	if err := os.WriteFile(path, []byte(script.String()), 0o700); err != nil { t.Fatal(err) }
	return path
}

func TestAntigravityCompleteParsesTextAndUsage(t *testing.T) {
	command := fakeAgy(t,
		`{"event":"step_update","step_update":{"step_type":"model","text_delta":"hello"}}`,
		`{"event":"result","result":{"status":"SUCCESS","response":"hello","usage":{"input_tokens":12,"output_tokens":3,"cache_read_tokens":2}}}`,
	)
	p := &antigravityProvider{command: command, model: "gemini-test", timeout: time.Second * 5}
	msg, stop, usage, err := p.Complete(context.Background(), llm.CompletionRequest{Messages: []llm.Message{llm.UserText("hi")}})
	if err != nil { t.Fatal(err) }
	if msg.Text() != "hello" { t.Fatalf("text = %q, want hello", msg.Text()) }
	if stop != "end_turn" { t.Fatalf("stop = %q, want end_turn", stop) }
	if usage.InputTokens != 12 || usage.OutputTokens != 3 || usage.CacheReadTokens != 2 {
		t.Fatalf("unexpected usage: %+v", usage)
	}
}

func TestAntigravityCompleteMapsTextToolCall(t *testing.T) {
	callJSON, err := json.Marshal(map[string]any{
		"id": "call-1",
		"type": "function",
		"function": map[string]any{"name": "list_assets", "arguments": `{"limit":10}`},
	})
	if err != nil { t.Fatal(err) }
	stepJSON, err := json.Marshal(map[string]any{
		"event": "step_update",
		"step_update": map[string]any{"step_type": "model", "text_delta": "<tool_call>" + string(callJSON) + "</tool_call>"},
	})
	if err != nil { t.Fatal(err) }
	resultJSON := `{"event":"result","result":{"status":"SUCCESS","response":""}}`
	command := fakeAgy(t, string(stepJSON), resultJSON)
	p := &antigravityProvider{command: command, model: "gemini-test", timeout: time.Second * 5}
	msg, stop, _, err := p.Complete(context.Background(), llm.CompletionRequest{
		Messages: []llm.Message{llm.UserText("list assets")},
		Tools: []llm.ToolSchema{{Name: "list_assets", InputSchema: map[string]any{"type": "object"}}},
	})
	if err != nil { t.Fatal(err) }
	if stop != "tool_use" { t.Fatalf("stop = %q, want tool_use", stop) }
	if len(msg.ToolUses()) != 1 { t.Fatalf("tool uses = %d, want 1: %#v", len(msg.ToolUses()), msg.Content) }
	call := msg.ToolUses()[0]
	if call.Name != "list_assets" || string(call.Input) != `{"limit":10}` {
		t.Fatalf("unexpected tool call: %#v", call)
	}
}

func TestAntigravityPromptPreservesSystemMessagesAndToolSchemas(t *testing.T) {
	req := llm.CompletionRequest{
		System: []string{"Follow the ARTEX policy."},
		Messages: []llm.Message{
			llm.UserText("Inspect the target."),
		},
		Tools: []llm.ToolSchema{{
			Name: "list_assets",
			Description: "List known assets",
			InputSchema: map[string]any{"type": "object", "properties": map[string]any{}},
		}},
	}
	prompt, err := antigravityPrompt(req)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"Follow the ARTEX policy.", "Inspect the target.", "list_assets", "<tool_call>", "never execute tools yourself"} {
		if !strings.Contains(prompt, want) {
			t.Errorf("prompt missing %q", want)
		}
	}
}

func TestAntigravityPromptSerializesToolResults(t *testing.T) {
	req := llm.CompletionRequest{Messages: []llm.Message{{
		Role: llm.RoleAssistant,
		Content: []llm.ContentBlock{{
			Type: llm.BlockToolUse, ID: "call-1", Name: "list_assets", Input: []byte("{\"limit\":10}"),
		}},
	}, {
		Role: llm.RoleUser,
		Content: []llm.ContentBlock{{
			Type: llm.BlockToolResult, ToolUseID: "call-1", Content: []llm.ContentBlock{llm.TextBlock("asset-a")},
		}},
	}}}
	prompt, err := antigravityPrompt(req)
	if err != nil {
		t.Fatal(err)
	}
	// Tool arguments are represented as a JSON string in the OpenAI-style
	// function-call envelope, so the inner JSON quotes are escaped once.
	for _, want := range []string{"call-1", "list_assets", `\\"limit\\":10`, "Tool Result (call-1)", "asset-a"} {
		if !strings.Contains(prompt, want) {
			t.Errorf("prompt missing %q", want)
		}
	}
}
