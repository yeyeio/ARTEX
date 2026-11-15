package agent

import (
	"strings"
	"testing"

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
	for _, want := range []string{"call-1", "list_assets", "\"limit\":10", "Tool Result (call-1)", "asset-a"} {
		if !strings.Contains(prompt, want) {
			t.Errorf("prompt missing %q", want)
		}
	}
}
