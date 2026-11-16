package server

import (
	"context"
	"errors"
	"fmt"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Autumn-27/artex/db"
)

func TestLLMRecoverySupervisorBackoffAndRecovery(t *testing.T) {
	m, err := NewManager(t.TempDir(), "")
	if err != nil {
		t.Skipf("postgres unavailable (%v) - skipping", err)
	}
	defer m.Close()

	suffix := time.Now().UnixNano()
	pID1, err := m.pg.SaveProfile(&db.LLMProfile{
		Name: fmt.Sprintf("recov-p1-%d", suffix), Format: "openai", Model: "gpt-4", APIKey: "k1",
	})
	if err != nil {
		t.Fatal(err)
	}
	pID2, err := m.pg.SaveProfile(&db.LLMProfile{
		Name: fmt.Sprintf("recov-p2-%d", suffix), Format: "openai", Model: "gpt-3.5", APIKey: "k2",
	})
	if err != nil {
		t.Fatal(err)
	}

	task, err := m.CreateTaskWithOptions("test recovery", "testing auto resume on recovery", db.TaskCreateOptions{
		LLMProfileIDs: []int64{pID1, pID2},
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_, _ = m.DeleteTask(task.ID, DeleteTaskOptions{})
		_ = m.pg.DeleteProfile(pID1)
		_ = m.pg.DeleteProfile(pID2)
	})

	taskNum, err := parseTaskID(task.ID)
	if err != nil {
		t.Fatal(err)
	}

	// Exhaust pID1 -> advances to pID2
	step1, err := m.pg.MarkTaskLLMProfileQuotaExhausted(taskNum, pID1, "quota exceeded 429")
	if err != nil || !step1.Advanced || step1.NextProfileID == nil || *step1.NextProfileID != pID2 {
		t.Fatalf("failover to p2 failed: transition=%+v err=%v", step1, err)
	}

	// Exhaust pID2 -> whole chain exhausted
	step2, err := m.pg.MarkTaskLLMProfileQuotaExhausted(taskNum, pID2, "quota exceeded 429")
	if err != nil || !step2.ChainExhausted {
		t.Fatalf("exhausting chain failed: transition=%+v err=%v", step2, err)
	}

	pt, err := m.pg.GetTask(taskNum)
	if err != nil || pt.ActiveLLMProfileID != nil || pt.LLMFailoverState != "chain_exhausted" {
		t.Fatalf("task is not chain_exhausted: %+v", pt)
	}

	// Setup Server and Supervisor
	s := &Server{m: m, engine: NewEngine(m)}
	sup := newLLMRecoverySupervisor(s)
	sup.SetConfig(TaskLLMRecoveryConfig{
		Enabled:      true,
		InitialCheck: 10 * time.Millisecond,
		NextInterval: 50 * time.Millisecond,
		MaxInterval:  200 * time.Millisecond,
		ProbeTimeout: 1 * time.Second,
	})

	task.setLLMState(pt.LLMProfileID, pt.ActiveLLMProfileID, pt.LLMProfileIDs, pt.LLMChainRevision, pt.LLMFailoverState, pt.LLMFailoverReason)

	// Add a blocked intent
	intentID, err := task.Store.AddIntent(map[string]any{"action": "test probe recovery"}, 1, nil, "planner")
	if err != nil {
		t.Fatal(err)
	}
	if err := task.Store.SetIntentBlockedReason(intentID, db.IntentBlockedLLMQuota); err != nil {
		t.Fatal(err)
	}

	var probeCalled atomic.Int32
	var probeReturnQuota atomic.Bool
	probeReturnQuota.Store(true) // initially probe fails with quota error

	sup.SetProbeFunc(func(ctx context.Context, profileID int64) error {
		probeCalled.Add(1)
		if probeReturnQuota.Load() {
			return errors.New("429 Too Many Requests: quota exhausted")
		}
		return nil
	})

	// First check: probe fails
	ctx := context.Background()
	recovered, err := sup.TriggerRecovery(ctx, task.ID)
	if err == nil || recovered {
		t.Fatalf("expected probe failure, got recovered=%v err=%v", recovered, err)
	}
	if probeCalled.Load() != 1 {
		t.Fatalf("probe calls=%d, want 1", probeCalled.Load())
	}

	// Task should still be chain_exhausted
	if task.llmStateSnapshot().FailoverState != "chain_exhausted" {
		t.Fatalf("failover state=%q, want chain_exhausted", task.llmStateSnapshot().FailoverState)
	}

	// Now simulate LLM recovering!
	probeReturnQuota.Store(false)

	// Trigger recovery again: probe succeeds
	recovered, err = sup.TriggerRecovery(ctx, task.ID)
	if err != nil || !recovered {
		t.Fatalf("expected recovery success, got recovered=%v err=%v", recovered, err)
	}

	// Profile pID1 should now be ready and active!
	updatedState := task.llmStateSnapshot()
	if updatedState.ActiveID == nil || *updatedState.ActiveID != pID1 {
		t.Fatalf("active profile=%v, want %d", updatedState.ActiveID, pID1)
	}
	if updatedState.FailoverState != "ready" {
		t.Fatalf("failover state=%q, want ready", updatedState.FailoverState)
	}

	// Verify blocked intent was reopened from checkpoint
	reopenedNode, err := task.Store.GetNode(intentID)
	if err != nil {
		t.Fatal(err)
	}
	if reopenedNode.State != "open" {
		t.Fatalf("intent state=%q, want open (reopened from checkpoint)", reopenedNode.State)
	}
}
