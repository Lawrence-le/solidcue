# Chat UI Behavior Spec

This document defines the expected behavior for the routed chat UI in `studio`.
It is the source of truth for message rendering, streaming, reconnect, resume,
and task-state interactions.

## Goals

- Keep chat state predictable across refresh, reconnect, and resume.
- Avoid duplicate message bubbles.
- Never reset task timing unless a new task actually starts.
- Make the input bar reflect the real task state.
- Keep node rail, timer, and message stream aligned with the same run.

## Core Concepts

- `conversation_id` identifies the chat session.
- `thread_id` identifies the checkpointed graph thread.
- `run_id` identifies the current in-flight execution.
- The UI never starts backend execution on refresh.
- The UI only starts backend execution for a new chat or an explicit resume.
- `runState` controls the user-facing UI mode.
- `messages` is the rendered chat transcript.
- `nodeEvents` is the right-side task trace.

## UI States

### 1. Fresh Chat

No conversation is loaded yet.

Expected behavior:

- No message bubbles are shown.
- The input bar is enabled.
- The send button is enabled when text is present.
- No timer is shown.
- No node rail activity is shown.

### 2. New User Send

The user submits a new message.

Expected behavior:

- The user bubble appears immediately.
- The input bar becomes disabled.
- The button changes to a stop button.
- The placeholder changes to indicate that the task is running.
- The node rail starts updating when backend node events arrive.
- The timer starts only when the backend run actually begins.

### 3. Live Run In Progress

The backend graph is active and streaming.

Expected behavior:

- Assistant text streams into the active assistant bubble.
- The input bar stays disabled.
- The stop button remains visible.
- The timer continues increasing.
- The node rail shows the current node sequence.
- Refresh does not start a second run.

### 4. Refresh During Live Run

The browser refreshes while the run is still active.

Expected behavior:

- The UI restores the conversation snapshot first.
- Existing chat history is displayed once.
- The UI does not start backend execution.
- The UI does not open a new stream.
- The UI reflects the session as snapshot-only until the user explicitly resumes.
- The timer value is restored from snapshot state.
- No duplicate user bubble is created.
- No duplicate assistant bubble is created.
- The input bar stays disabled while the task is still active.

### 5. Resume After Cancel

The user intentionally cancelled the task and then resumes it.

Expected behavior:

- The conversation keeps its existing history.
- The timer resumes from the saved checkpoint value.
- The node rail preserves completed work where appropriate.
- The input bar remains disabled during the resumed run.
- The stop button is available again.

### 6. Disconnected But Resumable

The browser disconnected and the run can still be resumed.

Expected behavior:

- Show a disconnected/rejoin banner.
- Keep the existing conversation visible.
- Keep the timer value from the checkpoint.
- Disable the input bar while disconnected.
- Rejoin should start a new backend continuation from the latest checkpoint.

### 7. Interrupted / Approval Required

The graph pauses for user approval.

Expected behavior:

- Render the interrupt card.
- Disable normal chat input unless the approval flow explicitly requires it.
- Show the run as paused or interrupted.
- Preserve the timer state for the interrupted thread.
- Do not duplicate the assistant transcript.

### 8. Completed Run

The graph reaches its final output.

Expected behavior:

- The final assistant response is visible.
- The timer is finalized.
- The node rail marks all nodes complete.
- The input bar is re-enabled.
- No reconnect banner is shown.

### 9. Error

The run fails.

Expected behavior:

- Show an error message in the chat.
- Finalize or stop the timer.
- Clear active streaming state.
- Re-enable the input bar.
- Clear transient reconnect status messages.

## Rendering Rules

- User messages render on the right.
- Assistant messages render on the left.
- System messages are centered and styled as lightweight status text.
- Interrupts render as a special approval card.
- Error messages render with error styling.

## Input Bar Rules

- Disable the input bar while a task is running.
- Disable the input bar while reconnecting.
- Show `Task is running...` when the task is active.
- Show a stop button while the task is active.
- Keep the dropdown and other controls disabled when task execution is active.

## Timer Rules

- The timer starts when the backend run starts.
- The timer does not restart from zero on refresh reconnect.
- The timer should continue from the saved `worked_seconds` and `timer_started_at`.
- The timer finalizes on completion, cancellation, interruption, or error.

## Node Rail Rules

- The node rail should reflect the current backend run, not a synthetic UI state.
- Do not clear node history on refresh reconnect unless the run truly restarted.
- Do not duplicate nodes when reconnecting.
- Keep node events visible after completion unless the user switches conversations.

## Refresh Rules

- Refresh should rehydrate from snapshot.
- Refresh should not create a live task.
- Refresh should not duplicate chat bubbles.
- Refresh should not reset timer state.
- Refresh should not change the backend runtime.

## Implementation Notes

- Separate snapshot hydration from live stream attachment.
- Treat refresh as a render/state recovery concern, not a runtime trigger.
- Keep resumed and refreshed runs anchored to the same `conversation_id`.
- Use the same state transition path for agent and router flows where possible.

## Invariants

- One active run per thread.
- One user bubble per user turn.
- One assistant bubble per assistant turn.
- One timer per active task.
- One node rail timeline per run.
