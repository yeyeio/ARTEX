package server

import (
	"context"
	"fmt"
	"log"
	"strconv"
	"sync"
	"time"

	"github.com/Autumn-27/artex/db"
	"github.com/Autumn-27/norma/llm"
)

// TaskLLMRecoveryConfig defines the timing and retry rules for auto-resuming tasks
// after LLM quota exhaustion.
type TaskLLMRecoveryConfig struct {
	Enabled      bool          `json:"enabled"`
	InitialCheck time.Duration `json:"initial_check"`
	NextInterval time.Duration `json:"next_interval"`
	MaxInterval  time.Duration `json:"max_interval"`
	ProbeTimeout time.Duration `json:"probe_timeout"`
}

// DefaultTaskLLMRecoveryConfig provides the recommended defaults:
// 30s initial recovery check, 60s next interval, 5m maximum interval.
func DefaultTaskLLMRecoveryConfig() TaskLLMRecoveryConfig {
	return TaskLLMRecoveryConfig{
		Enabled:      true,
		InitialCheck: 30 * time.Second,
		NextInterval: 60 * time.Second,
		MaxInterval:  5 * time.Minute,
		ProbeTimeout: 15 * time.Second,
	}
}

type taskRecoveryState struct {
	attempts    int
	exhaustedAt time.Time
	lastCheckAt time.Time
	nextCheckAt time.Time
	lastError   string
	probing     bool
}

// LLMRecoverySupervisor monitors tasks whose LLM chains have been exhausted,
// tests candidate profiles periodically with lightweight inference probes,
// and automatically resumes tasks from their previous checkpoints once a profile recovers.
type LLMRecoverySupervisor struct {
	s          *Server
	mu         sync.Mutex
	cfg        TaskLLMRecoveryConfig
	taskStates map[string]*taskRecoveryState
	probeFn    func(ctx context.Context, profileID int64) error
}

func newLLMRecoverySupervisor(s *Server) *LLMRecoverySupervisor {
	return &LLMRecoverySupervisor{
		s:          s,
		cfg:        DefaultTaskLLMRecoveryConfig(),
		taskStates: make(map[string]*taskRecoveryState),
	}
}

// SetProbeFunc overrides the inference probe implementation (primarily for unit testing).
func (sup *LLMRecoverySupervisor) SetProbeFunc(fn func(ctx context.Context, profileID int64) error) {
	sup.mu.Lock()
	defer sup.mu.Unlock()
	sup.probeFn = fn
}

// SetConfig updates the supervisor configuration.
func (sup *LLMRecoverySupervisor) SetConfig(cfg TaskLLMRecoveryConfig) {
	sup.mu.Lock()
	defer sup.mu.Unlock()
	sup.cfg = cfg
}

// Config returns a copy of the current configuration.
func (sup *LLMRecoverySupervisor) Config() TaskLLMRecoveryConfig {
	sup.mu.Lock()
	defer sup.mu.Unlock()
	return sup.cfg
}

// Run starts the periodic background sweep for tasks waiting on LLM recovery.
func (sup *LLMRecoverySupervisor) Run(ctx context.Context) {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			sup.CheckTasks(ctx)
		}
	}
}

// NextCheckFor returns the scheduled next recovery check time for a task, if any.
func (sup *LLMRecoverySupervisor) NextCheckFor(taskID string) (time.Time, bool) {
	sup.mu.Lock()
	defer sup.mu.Unlock()
	st, ok := sup.taskStates[taskID]
	if !ok {
		return time.Time{}, false
	}
	return st.nextCheckAt, true
}

// probe performs a lightweight inference check against the specified profile.
func (sup *LLMRecoverySupervisor) probe(ctx context.Context, profileID int64) error {
	sup.mu.Lock()
	customProbe := sup.probeFn
	timeout := sup.cfg.ProbeTimeout
	sup.mu.Unlock()

	if customProbe != nil {
		return customProbe(ctx, profileID)
	}

	if sup.s == nil {
		return fmt.Errorf("server instance is nil")
	}
	return sup.s.probeLLMProfileWithTimeout(ctx, profileID, timeout)
}

// CheckTasks inspects all loaded tasks and probes candidate profiles for those in Waiting for LLM.
func (sup *LLMRecoverySupervisor) CheckTasks(ctx context.Context) {
	sup.mu.Lock()
	enabled := sup.cfg.Enabled
	sup.mu.Unlock()

	if !enabled || sup.s == nil || sup.s.m == nil || sup.s.m.pg == nil {
		return
	}

	tasks := sup.s.m.List()
	now := time.Now()

	for _, task := range tasks {
		select {
		case <-ctx.Done():
			return
		default:
		}
		sup.checkSingleTask(ctx, task, now, false)
	}
}

// TriggerRecovery forces an immediate check and recovery attempt for a task, bypassing the backoff timer.
func (sup *LLMRecoverySupervisor) TriggerRecovery(ctx context.Context, taskID string) (bool, error) {
	task, ok := sup.s.m.Task(taskID)
	if !ok {
		return false, fmt.Errorf("task %s not found", taskID)
	}
	return sup.checkSingleTask(ctx, task, time.Now(), true)
}

func (sup *LLMRecoverySupervisor) checkSingleTask(ctx context.Context, task *Task, now time.Time, force bool) (bool, error) {
	taskNum, err := strconv.ParseInt(task.ID, 10, 64)
	if err != nil || taskNum <= 0 {
		return false, nil
	}

	lifecycle := task.lifecycleSnapshot()
	if isTerminalStatus(lifecycle.Status) {
		sup.mu.Lock()
		delete(sup.taskStates, task.ID)
		sup.mu.Unlock()
		return false, nil
	}

	llmState := task.llmStateSnapshot()
	isExhausted := llmState.FailoverState == db.LLMFailoverStateChainExhausted ||
		llmState.FailoverState == db.LLMFailoverStateWaitingForLLM

	if !isExhausted {
		sup.mu.Lock()
		delete(sup.taskStates, task.ID)
		sup.mu.Unlock()
		return false, nil
	}

	profiles, err := sup.s.m.pg.TaskLLMProfiles(taskNum)
	if err != nil || len(profiles) == 0 {
		return false, err
	}

	// Find exhausted profiles in chain order
	var candidate *db.TaskLLMProfile
	for i := range profiles {
		if profiles[i].Status == "quota_exhausted" {
			candidate = &profiles[i]
			break
		}
	}
	if candidate == nil {
		return false, nil
	}

	sup.mu.Lock()
	cfg := sup.cfg
	st, exists := sup.taskStates[task.ID]
	if !exists {
		exhaustedTime := now
		if candidate.ExhaustedAt != nil {
			exhaustedTime = *candidate.ExhaustedAt
		}
		st = &taskRecoveryState{
			attempts:    0,
			exhaustedAt: exhaustedTime,
			nextCheckAt: exhaustedTime.Add(cfg.InitialCheck),
		}
		sup.taskStates[task.ID] = st
	}

	if !force && now.Before(st.nextCheckAt) {
		sup.mu.Unlock()
		return false, nil
	}

	if st.probing {
		sup.mu.Unlock()
		return false, nil
	}
	st.probing = true
	sup.mu.Unlock()

	defer func() {
		sup.mu.Lock()
		if s, ok := sup.taskStates[task.ID]; ok {
			s.probing = false
		}
		sup.mu.Unlock()
	}()

	// Probe candidate profile
	probeErr := sup.probe(ctx, candidate.ProfileID)
	if probeErr == nil {
		// Recovery SUCCESS! Reactivate this profile for the task
		transition, reopened, recErr := sup.s.m.RecoverTaskLLMProfile(task.ID, candidate.ProfileID)
		if recErr != nil {
			log.Printf("[llm-recovery] task %s recover profile #%d in db failed: %v", task.ID, candidate.ProfileID, recErr)
			return false, recErr
		}

		// Reset circuit breaker and consecutive fail streak
		if sup.s.llmHealth != nil {
			sup.s.llmHealth.Reset(candidate.ProfileID)
		}
		resetLLMFailStreak(task.ID, candidate.ProfileID)

		summary := fmt.Sprintf("LLM profile #%d recovered, task resuming from previous progress (%d intents reopened)", candidate.ProfileID, reopened)
		sup.s.engine.emitActivity(task, db.Activity{
			Worker:  "system",
			Kind:    "llm_recovery",
			Summary: summary,
			Detail:  fmt.Sprintf("Transition: %+v", transition),
		})
		log.Printf("[llm-recovery] task %s: %s", task.ID, summary)

		// Wake the task engine/workers
		task.Notify()

		sup.mu.Lock()
		delete(sup.taskStates, task.ID)
		sup.mu.Unlock()
		return true, nil
	}

	// Probe failed — advance backoff schedule
	sup.mu.Lock()
	if s, ok := sup.taskStates[task.ID]; ok {
		s.attempts++
		interval := cfg.NextInterval * time.Duration(1<<min(s.attempts-1, 4))
		if interval > cfg.MaxInterval {
			interval = cfg.MaxInterval
		}
		s.nextCheckAt = now.Add(interval)
		s.lastCheckAt = now
		s.lastError = probeErr.Error()
		log.Printf("[llm-recovery] task %s profile #%d probe failed: %v; next recovery check in %v (attempt %d)",
			task.ID, candidate.ProfileID, probeErr, interval, s.attempts)
	}
	sup.mu.Unlock()

	return false, probeErr
}

// probeLLMProfileWithTimeout sends a minimal 1-token completion request to test inference capability.
func (s *Server) probeLLMProfileWithTimeout(ctx context.Context, profileID int64, timeout time.Duration) error {
	prov, _, ok := s.providerForProfile(profileID)
	if !ok {
		return fmt.Errorf("profile #%d not found or invalid", profileID)
	}

	probeCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	// Dry-run inference probe: a minimal request to verify inference works without side effects
	req := llm.CompletionRequest{
		Messages: []llm.Message{
			llm.UserText("ping"),
		},
		MaxTokens: 1,
	}

	_, _, _, err := prov.Complete(probeCtx, req)
	return err
}
