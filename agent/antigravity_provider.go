package agent

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"os"
	"os/exec"
	"strings"
	"sync"
	"time"

	"github.com/Autumn-27/norma/llm"
)

// antigravityProvider delegates inference to Google's official agy CLI. It never
// reads, copies, or persists OAuth credentials; agy owns authentication.
type antigravityProvider struct {
	command string
	model   string
	effort  string
	timeout time.Duration

	rateInterval time.Duration
	rateMu       sync.Mutex
	lastStart    time.Time
}

func newAntigravityProvider(c Config) (llm.Provider, error) {
	command := strings.TrimSpace(os.Getenv("ANTIGRAVITY_COMMAND"))
	if command == "" { command = "agy" }
	if _, err := exec.LookPath(command); err != nil {
		return nil, fmt.Errorf("antigravity: agy CLI not found (%s): %w", command, err)
	}
	interval := time.Duration(0)
	if c.RatePerSecond > 0 {
		interval = time.Duration(float64(time.Second) / c.RatePerSecond)
	}
	if c.RatePerMinute > 0 {
		minuteInterval := time.Duration(float64(time.Minute) / c.RatePerMinute)
		if minuteInterval > interval { interval = minuteInterval }
	}
	return &antigravityProvider{command: command, model: c.Model, effort: c.ReasoningEffort, timeout: 5 * time.Minute, rateInterval: interval}, nil
}

func (p *antigravityProvider) args() []string {
	args := []string{"--output-format", "stream-json", "--disable-slash-commands"}
	if p.model != "" { args = append(args, "--model", p.model) }
	// agy only accepts a separate effort flag for suffixless Gemini model IDs.
	// Native model variants (for example, gemini-*-medium) already encode effort.
	if strings.HasPrefix(p.model, "gemini-") && !hasNativeAgyEffortSuffix(p.model) && isNativeAgyEffort(p.effort) {
		args = append(args, "--effort", p.effort)
	}
	return args
}

func hasNativeAgyEffortSuffix(model string) bool {
	switch model[strings.LastIndex(model, "-")+1:] {
	case "low", "medium", "high", "xhigh", "max":
		return true
	default:
		return false
	}
}

func isNativeAgyEffort(effort string) bool {
	switch strings.ToLower(strings.TrimSpace(effort)) {
	case "low", "medium", "high", "xhigh", "max":
		return true
	default:
		return false
	}
}

func (p *antigravityProvider) Stream(ctx context.Context, req llm.CompletionRequest) iter.Seq2[llm.StreamEvent, error] {
	return func(yield func(llm.StreamEvent, error) bool) {
		if err := p.run(ctx, req, yield); err != nil { yield(llm.StreamEvent{}, err) }
	}
}

func (p *antigravityProvider) Complete(ctx context.Context, req llm.CompletionRequest) (llm.Message, string, llm.Usage, error) {
	var message llm.Message
	message.Role = llm.RoleAssistant
	var usage llm.Usage
	stop := "end_turn"
	err := p.run(ctx, req, func(ev llm.StreamEvent, err error) bool {
		if err != nil { return false }
		switch ev.Type {
		case llm.SETextDelta:
			if len(message.Content) == 0 || message.Content[len(message.Content)-1].Type != llm.BlockText {
				message.Content = append(message.Content, llm.TextBlock(""))
			}
			message.Content[len(message.Content)-1].Text += ev.Text
		case llm.SEThinkingDelta:
			if len(message.Content) == 0 || message.Content[len(message.Content)-1].Type != llm.BlockThinking {
				message.Content = append(message.Content, llm.ContentBlock{Type: llm.BlockThinking})
			}
			message.Content[len(message.Content)-1].Thinking += ev.Text
		case llm.SEToolUseStart:
			stop = "tool_use"
			message.Content = append(message.Content, llm.ContentBlock{Type: llm.BlockToolUse, ID: ev.ToolID, Name: ev.ToolName, Input: json.RawMessage("{}")})
		case llm.SEToolInputJSON:
			if len(message.Content) > 0 {
				last := &message.Content[len(message.Content)-1]
				if last.Type == llm.BlockToolUse { last.Input = append(last.Input[:0], ev.Text...) }
			}
		case llm.SEMessageDelta:
			usage.MergeEcho(ev.Usage)
			if ev.StopReason != "" { stop = ev.StopReason }
		}
		return true
	})
	return message, stop, usage, err
}

func (p *antigravityProvider) run(ctx context.Context, req llm.CompletionRequest, yield func(llm.StreamEvent, error) bool) error {
	if ctx == nil { ctx = context.Background() }
	callCtx, cancel := context.WithTimeout(ctx, p.timeout)
	defer cancel()
	if p.rateInterval > 0 {
		p.rateMu.Lock()
		wait := time.Until(p.lastStart.Add(p.rateInterval))
		if wait > 0 {
			timer := time.NewTimer(wait)
			select {
			case <-callCtx.Done():
				timer.Stop()
				p.rateMu.Unlock()
				return callCtx.Err()
			case <-timer.C:
			}
		}
		p.lastStart = time.Now()
		p.rateMu.Unlock()
	}
	prompt, err := antigravityPrompt(req)
	if err != nil { return err }
	cmd := exec.CommandContext(callCtx, p.command, p.args()...)
	cmd.Stdin = strings.NewReader(prompt)
	var stderr bytes.Buffer
	cmd.Stderr = &limitedWriter{buf: &stderr, max: 64 * 1024}
	stdout, err := cmd.StdoutPipe()
	if err != nil { return fmt.Errorf("antigravity: stdout pipe: %w", err) }
	if err = cmd.Start(); err != nil { return fmt.Errorf("antigravity: start agy: %w", err) }
	if !yield(llm.StreamEvent{Type: llm.SEMessageStart}, nil) { _ = cmd.Process.Kill(); _ = cmd.Wait(); return nil }

	scanner := bufio.NewScanner(stdout)
	scanner.Buffer(make([]byte, 64*1024), 4*1024*1024)
	sawResult, sawOutput := false, false
	textPending := ""
	emitText := func(delta string, final bool) bool {
		textPending += delta
		for len(textPending) > 0 {
			start := strings.Index(textPending, "<tool_call>")
			if start < 0 {
				if final {
					if textPending != "" && !yield(llm.StreamEvent{Type: llm.SETextDelta, Text: textPending}, nil) { return false }
					textPending = ""
					return true
				}
				keep := 0
				max := len("<tool_call>") - 1
				if len(textPending) < max { max = len(textPending) }
				for n := max; n > 0; n-- {
					if strings.HasSuffix(textPending, "<tool_call>"[:n]) { keep = n; break }
				}
				safe := textPending[:len(textPending)-keep]
				textPending = textPending[len(safe):]
				if safe != "" && !yield(llm.StreamEvent{Type: llm.SETextDelta, Text: safe}, nil) { return false }
				return true
			}
			if start > 0 {
				plain := textPending[:start]
				textPending = textPending[start:]
				if !yield(llm.StreamEvent{Type: llm.SETextDelta, Text: plain}, nil) { return false }
			}
			end := strings.Index(textPending, "</tool_call>")
			if end < 0 {
				if final {
					if !yield(llm.StreamEvent{Type: llm.SETextDelta, Text: textPending}, nil) { return false }
					textPending = ""
				}
				return true
			}
			body := strings.TrimSpace(textPending[len("<tool_call>"):end])
			textPending = textPending[end+len("</tool_call>"):]
			var call struct {
				ID string `json:"id"`
				Function struct {
					Name string `json:"name"`
					Arguments json.RawMessage `json:"arguments"`
				} `json:"function"`
			}
			if err := json.Unmarshal([]byte(body), &call); err != nil || call.Function.Name == "" {
				if !yield(llm.StreamEvent{Type: llm.SETextDelta, Text: "<tool_call>"+body+"</tool_call>"}, nil) { return false }
				continue
			}
			args := call.Function.Arguments
			if len(args) == 0 { args = json.RawMessage("{}") }
			if len(args) > 0 && args[0] == '"' {
				var encoded string
				if json.Unmarshal(args, &encoded) == nil { args = json.RawMessage(encoded) }
			}
			if !json.Valid(args) { args = json.RawMessage("{}") }
			id := call.ID
			if id == "" { id = fmt.Sprintf("agy-call-%d", time.Now().UnixNano()) }
			if !yield(llm.StreamEvent{Type: llm.SEToolUseStart, ToolID: id, ToolName: call.Function.Name}, nil) { return false }
			if !yield(llm.StreamEvent{Type: llm.SEToolInputJSON, Text: string(args)}, nil) { return false }
		}
		return true
	}
	for scanner.Scan() {
		if callCtx.Err() != nil { break }
		line := strings.TrimSpace(scanner.Text())
		if line == "" || !strings.HasPrefix(line, "{") { continue }
		var event map[string]any
		if json.Unmarshal([]byte(line), &event) != nil { continue }
		switch event["event"] {
		case "step_update":
			step, _ := event["step_update"].(map[string]any)
			if step == nil { continue }
			if step["step_type"] == "tool" {
				_ = cmd.Process.Kill()
				_ = cmd.Wait()
				yield(llm.StreamEvent{}, errors.New("antigravity: agy attempted a native tool; native CLI tools are disabled, use ARTEX function-call output"))
				return nil
			}
			if s, ok := step["thinking_delta"].(string); ok && s != "" {
				sawOutput = true
				if !yield(llm.StreamEvent{Type: llm.SEThinkingDelta, Text: s}, nil) { _ = cmd.Process.Kill(); _ = cmd.Wait(); return nil }
			}
			if s, ok := step["text_delta"].(string); ok && s != "" {
				sawOutput = true
				if !emitText(s, false) { _ = cmd.Process.Kill(); _ = cmd.Wait(); return nil }
			}
			if u, ok := step["usage"].(map[string]any); ok {
				if !yield(llm.StreamEvent{Type: llm.SEMessageDelta, Usage: agyUsage(u)}, nil) { _ = cmd.Process.Kill(); _ = cmd.Wait(); return nil }
			}
		case "result":
			res, _ := event["result"].(map[string]any)
			if res == nil { continue }
			sawResult = true
			if u, ok := res["usage"].(map[string]any); ok {
				if !yield(llm.StreamEvent{Type: llm.SEMessageDelta, Usage: agyUsage(u)}, nil) { _ = cmd.Process.Kill(); _ = cmd.Wait(); return nil }
			}
			if msg, _ := res["error"].(string); msg != "" { _ = cmd.Wait(); return fmt.Errorf("antigravity: %s", msg) }
			if status, _ := res["status"].(string); status != "SUCCESS" { _ = cmd.Wait(); return fmt.Errorf("antigravity: unexpected result status %q", status) }
			if response, _ := res["response"].(string); response != "" && !sawOutput {
				if !emitText(response, false) { _ = cmd.Process.Kill(); _ = cmd.Wait(); return nil }
				sawOutput = true
			}
		}
	}
	scanErr := scanner.Err()
	waitErr := cmd.Wait()
	if callCtx.Err() != nil { return callCtx.Err() }
	if scanErr != nil { return fmt.Errorf("antigravity: read stream: %w", scanErr) }
	if !emitText("", true) { return nil }
	if waitErr != nil { return fmt.Errorf("antigravity: agy exited: %w: %s", waitErr, strings.TrimSpace(stderr.String())) }
	if !sawResult || !sawOutput { return fmt.Errorf("antigravity: CLI returned no complete response (result=%t output=%t): %s", sawResult, sawOutput, strings.TrimSpace(stderr.String())) }
	yield(llm.StreamEvent{Type: llm.SEMessageStop}, nil)
	return nil
}

func antigravityPrompt(req llm.CompletionRequest) (string, error) {
	var b strings.Builder
	b.WriteString("You are an AI model operating inside ARTEX. Follow the system instructions and conversation.\n")
	for _, part := range req.System { if strings.TrimSpace(part) != "" { b.WriteString("\nSYSTEM:\n"); b.WriteString(part); b.WriteString("\n") } }
	if len(req.Tools) > 0 {
		b.WriteString("\nARTEX TOOLS (you must request tools only using the exact <tool_call> JSON format; never execute tools yourself):\n")
		raw, err := json.Marshal(req.Tools); if err != nil { return "", err }; b.Write(raw)
		b.WriteString("\nEmit: <tool_call>{\"id\":\"call_id\",\"type\":\"function\",\"function\":{\"name\":\"tool_name\",\"arguments\":\"{}\"}}</tool_call>. Stop after the tool call.\n")
	}
	for _, msg := range req.Messages {
		role := string(msg.Role)
		if role == "" { role = "unknown" }
		b.WriteString("\n"); b.WriteString(strings.ToUpper(role)); b.WriteString(":\n")
		for _, block := range msg.Content {
			switch block.Type {
			case llm.BlockText: b.WriteString(block.Text)
			case llm.BlockThinking: b.WriteString("[prior reasoning omitted]")
			case llm.BlockToolUse:
				b.WriteString("<tool_call>")
				raw, err := json.Marshal(map[string]any{"id":block.ID,"type":"function","function":map[string]any{"name":block.Name,"arguments":string(block.Input)}}); if err != nil { return "", err }; b.Write(raw); b.WriteString("</tool_call>")
			case llm.BlockToolResult:
				b.WriteString("Tool Result ("); b.WriteString(block.ToolUseID); b.WriteString("): ")
				for _, part := range block.Content { b.WriteString(part.Text) }
			}
		}
		b.WriteString("\n")
	}
	b.WriteString("\nRespond to the latest message. If a tool is needed, emit only a valid <tool_call> block and stop.\n")
	return b.String(), nil
}

func agyUsage(m map[string]any) llm.Usage {
	num := func(k string) int { if v, ok := m[k].(float64); ok { return int(v) }; return 0 }
	return llm.Usage{InputTokens:num("input_tokens"), OutputTokens:num("output_tokens"), CacheReadTokens:num("cache_read_tokens"), CacheWriteTokens:num("cache_write_tokens")}
}

type limitedWriter struct { buf *bytes.Buffer; max int }
func (w *limitedWriter) Write(p []byte) (int, error) {
	n := len(p)
	left := w.max - w.buf.Len()
	if left > 0 { if len(p) > left { p = p[:left] }; _, _ = w.buf.Write(p) }
	return n, nil
}
