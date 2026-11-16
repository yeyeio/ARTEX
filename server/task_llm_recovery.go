package server

import (
	"log"
	"strconv"
	"time"

	"github.com/Autumn-27/artex/db"
)

const (
	taskLLMRecoveryPollInterval = time.Minute
	taskLLMRecoveryCooldown    = 5 * time.Minute
)

// monitorExhaustedTaskLLMChains periodically re-arms one cooled-down profile for
// each task whose explicit chain is exhausted. It deliberately does not issue a
// synthetic LLM request: the next real planner/worker request is the probe, so
// recovery does not spend quota for work the task did not need to do.
func (s *Server) monitorExhaustedTaskLLMChains() {
	if s == nil || s.ctx == nil || s.m == nil || s.m.pg == nil {
		return
	}
	ticker := time.NewTicker(taskLLMRecoveryPollInterval)
	defer ticker.Stop()
	for {
		select {
		case <-s.ctx.Done():
			return
		case <-ticker.C:
			s.recoverExhaustedTaskLLMChains()
		}
	}
}

func (s *Server) recoverExhaustedTaskLLMChains() {
	if s == nil || s.m == nil || s.m.pg == nil {
		return
	}
	for _, task := range s.m.List() {
		lifecycle := task.lifecycleSnapshot()
		if lifecycle.Paused || isTerminalStatus(lifecycle.Status) || lifecycle.Queued {
			continue
		}
		state := task.llmStateSnapshot()
		if len(state.ProfileIDs) == 0 || state.ActiveID != nil {
			continue
		}

		taskID, err := strconv.ParseInt(task.ID, 10, 64)
		if err != nil {
			continue
		}
		profileID, recovered, err := s.m.pg.RecoverTaskLLMProfileAfterCooldown(taskID, taskLLMRecoveryCooldown)
		if err != nil {
			log.Printf("[llm-recovery] task %s: recover cooled-down profile: %v", task.ID, err)
			continue
		}
		if !recovered {
			continue
		}

		pt, err := s.m.pg.GetTask(taskID)
		if err != nil || pt == nil {
			log.Printf("[llm-recovery] task %s: reload recovered chain: %v", task.ID, err)
			continue
		}
		s.syncTaskLLMState(pt)

		// Building the provider validates the stored configuration without
		// sending a billable request. The first real task call then verifies
		// upstream health; if quota is still exhausted, normal failover marks
		// this profile exhausted again and the cooldown starts over.
		if _, _, ok := s.providerForProfile(profileID); !ok {
			transition, markErr := s.m.pg.MarkTaskLLMProfileQuotaExhaustedAtRevision(
				taskID, profileID, pt.LLMChainRevision,
				"recovery probe: LLM profile configuration is invalid or unavailable",
			)
			if markErr != nil {
				log.Printf("[llm-recovery] task %s profile %d: mark unavailable profile exhausted: %v", task.ID, profileID, markErr)
			} else if refreshed, getErr := s.m.pg.GetTask(taskID); getErr == nil && refreshed != nil {
				s.syncTaskLLMState(refreshed)
			}
			_ = transition
			continue
		}

		if _, err := task.Store.ReopenIntentsByBlockedReason(db.IntentBlockedLLMQuota); err != nil {
			log.Printf("[llm-recovery] task %s profile %d: reopen quota-blocked intents: %v", task.ID, profileID, err)
			continue
		}
		s.engine.emitActivity(task, db.Activity{
			Worker:  "system",
			Kind:    "llm_switch",
			Summary: "A cooled-down LLM profile was re-enabled; the task is resuming from its saved progress",
			Detail:  "Profile #" + strconv.FormatInt(profileID, 10) + " was returned to the task chain. Its next real request will verify upstream availability.",
		})
		task.Notify()
		log.Printf("[llm-recovery] task %s re-armed profile %d after cooldown", task.ID, profileID)
	}
}
