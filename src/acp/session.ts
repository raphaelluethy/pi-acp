import type {
  AgentSideConnection,
  ContentBlock,
  McpServer,
  SessionUpdate,
  ToolCallContent
} from '@agentclientprotocol/sdk'
import { RequestError } from '@agentclientprotocol/sdk'
import { maybeAuthRequiredError } from './auth-required.js'
import { readFileSync } from 'node:fs'
import { isAbsolute, resolve as resolvePath } from 'node:path'
import { PiRpcProcess, PiRpcSpawnError, type PiRpcEvent } from '../pi-rpc/process.js'
import { SessionStore } from './session-store.js'
import { expandSlashCommand, type FileSlashCommand } from './slash-commands.js'
import {
  toolInfoFromPiToolCall,
  provisionalDiffContentFromFileToolArgs,
  toolUpdateFromPiToolResult,
  shellTerminalId,
  shellExitMeta,
  formatShellToolResponse,
  shellToolPresentation,
  isRejectedToolResult
} from './tools.js'

type SessionCreateParams = {
  cwd: string
  mcpServers: McpServer[]
  conn: AgentSideConnection
  fileCommands?: import('./slash-commands.js').FileSlashCommand[]
  piCommand?: string
}

export type StopReason = 'end_turn' | 'cancelled' | 'error'

type PendingTurn = {
  resolve: (reason: StopReason) => void
  reject: (err: unknown) => void
}

type QueuedTurn = {
  message: string
  images: unknown[]
  resolve: (reason: StopReason) => void
  reject: (err: unknown) => void
}

function findUniqueLineNumber(text: string, needle: string): number | undefined {
  if (!needle) return undefined

  const first = text.indexOf(needle)
  if (first < 0) return undefined

  const second = text.indexOf(needle, first + needle.length)
  if (second >= 0) return undefined

  let line = 1
  for (let i = 0; i < first; i += 1) {
    if (text.charCodeAt(i) === 10) line += 1
  }
  return line
}

export class SessionManager {
  private sessions = new Map<string, PiAcpSession>()
  private readonly store = new SessionStore()

  /** Dispose all sessions and their underlying pi subprocesses. */
  disposeAll(): void {
    for (const [id] of this.sessions) this.close(id)
  }

  /** Get a registered session if it exists (no throw). */
  maybeGet(sessionId: string): PiAcpSession | undefined {
    return this.sessions.get(sessionId)
  }

  /**
   * Dispose a session's underlying pi process and remove it from the manager.
   * Used when clients explicitly reload a session and we want a fresh pi subprocess.
   */
  close(sessionId: string): void {
    const s = this.sessions.get(sessionId)
    if (!s) return
    try {
      s.proc.dispose?.()
    } catch {
      // ignore
    }
    this.sessions.delete(sessionId)
  }

  /** Close all sessions except the one with `keepSessionId`. */
  closeAllExcept(keepSessionId: string): void {
    for (const [id] of this.sessions) {
      if (id === keepSessionId) continue
      this.close(id)
    }
  }

  async create(params: SessionCreateParams): Promise<PiAcpSession> {
    // Let pi manage session persistence in its default location (~/.pi/agent/sessions/...)
    // so sessions are visible to the regular `pi` CLI.
    let proc: PiRpcProcess
    try {
      proc = await PiRpcProcess.spawn({
        cwd: params.cwd,
        piCommand: params.piCommand
      })
    } catch (e) {
      if (e instanceof PiRpcSpawnError) {
        throw RequestError.internalError({ code: e.code }, e.message)
      }
      throw e
    }

    let state: any = null
    try {
      state = (await proc.getState()) as any
    } catch {
      state = null
    }

    const sessionId = typeof state?.sessionId === 'string' ? state.sessionId : crypto.randomUUID()
    const sessionFile = typeof state?.sessionFile === 'string' ? state.sessionFile : null

    if (sessionFile) {
      this.store.upsert({ sessionId, cwd: params.cwd, sessionFile })
    }

    const session = new PiAcpSession({
      sessionId,
      cwd: params.cwd,
      mcpServers: params.mcpServers,
      proc,
      conn: params.conn,
      fileCommands: params.fileCommands ?? []
    })

    this.sessions.set(sessionId, session)
    return session
  }

  get(sessionId: string): PiAcpSession {
    const s = this.sessions.get(sessionId)
    if (!s) throw RequestError.invalidParams(`Unknown sessionId: ${sessionId}`)
    return s
  }

  /**
   * Used by session/load: create a session object bound to an existing sessionId/proc
   * if it isn't already registered.
   */
  getOrCreate(sessionId: string, params: SessionCreateParams & { proc: PiRpcProcess }): PiAcpSession {
    const existing = this.sessions.get(sessionId)
    if (existing) return existing

    const session = new PiAcpSession({
      sessionId,
      cwd: params.cwd,
      mcpServers: params.mcpServers,
      proc: params.proc,
      conn: params.conn,
      fileCommands: params.fileCommands ?? []
    })

    this.sessions.set(sessionId, session)
    return session
  }
}

export class PiAcpSession {
  readonly sessionId: string
  readonly cwd: string
  readonly mcpServers: McpServer[]

  private startupInfo: string | null = null
  private startupInfoSent = false

  readonly proc: PiRpcProcess
  private readonly conn: AgentSideConnection
  private readonly fileCommands: FileSlashCommand[]

  // Used to map abort semantics to ACP stopReason.
  // Applies to the currently running turn.
  private cancelRequested = false

  // Current in-flight turn (if any). Additional prompts are queued.
  private pendingTurn: PendingTurn | null = null
  private readonly turnQueue: QueuedTurn[] = []
  // Track tool call statuses and ensure they are monotonic (pending -> in_progress -> completed).
  // Some pi events can arrive out of order (e.g. late toolcall_* deltas after execution starts),
  // and clients may hide progress if we ever downgrade back to `pending`.
  private currentToolCalls = new Map<string, 'pending' | 'in_progress'>()

  // pi can emit multiple `turn_end` events for a single user prompt (e.g. after tool_use).
  // The overall agent loop completes when `agent_end` is emitted.
  private inAgentLoop = false

  // For ACP diff support: capture file contents before edits, then emit ToolCallContent {type:"diff"}.
  // This is due to pi sending diff as a string as opposed to ACP expected diff format.
  // Compatible format may need to be implemented in pi in the future.
  private editSnapshots = new Map<string, { path: string; oldText: string }>()

  // Ensure `session/update` notifications are sent in order and can be awaited
  // before completing a `session/prompt` request.
  private lastEmit: Promise<void> = Promise.resolve()

  constructor(opts: {
    sessionId: string
    cwd: string
    mcpServers: McpServer[]
    proc: PiRpcProcess
    conn: AgentSideConnection
    fileCommands?: FileSlashCommand[]
  }) {
    this.sessionId = opts.sessionId
    this.cwd = opts.cwd
    this.mcpServers = opts.mcpServers
    this.proc = opts.proc
    this.conn = opts.conn
    this.fileCommands = opts.fileCommands ?? []

    this.proc.onEvent(ev => this.handlePiEvent(ev))
  }

  setStartupInfo(text: string) {
    this.startupInfo = text
  }

  /**
   * Best-effort attempt to send startup info outside of a prompt turn.
   * Some clients (e.g. Zed) may only render agent messages once the UI is ready;
   * callers can invoke this shortly after session/new returns.
   */
  sendStartupInfoIfPending(): void {
    if (this.startupInfoSent || !this.startupInfo) return
    this.startupInfoSent = true

    this.emit({
      sessionUpdate: 'agent_message_chunk',
      content: { type: 'text', text: this.startupInfo }
    })
  }

  async prompt(message: string, images: unknown[] = []): Promise<StopReason> {

    // pi RPC mode disables slash command expansion, so we do it here.
    const expandedMessage = expandSlashCommand(message, this.fileCommands)

    const turnPromise = new Promise<StopReason>((resolve, reject) => {
      const queued: QueuedTurn = { message: expandedMessage, images, resolve, reject }

      // If a turn is already running, enqueue.
      if (this.pendingTurn) {
        this.turnQueue.push(queued)

        // Best-effort: notify client that a prompt was queued.
        // This doesn't work in Zed yet, needs to be revisited
        this.emit({
          sessionUpdate: 'agent_message_chunk',
          content: {
            type: 'text',
            text: `Queued message (position ${this.turnQueue.length}).`
          }
        })

        // Also publish queue depth via session info metadata.
        // This also not visible in the client
        this.emit({
          sessionUpdate: 'session_info_update',
          _meta: { piAcp: { queueDepth: this.turnQueue.length, running: true } }
        })

        return
      }

      // No turn is running; start immediately.
      this.startTurn(queued)
    })

    return turnPromise
  }

  async cancel(): Promise<void> {
    // Cancel current and clear any queued prompts.
    this.cancelRequested = true

    if (this.turnQueue.length) {
      const queued = this.turnQueue.splice(0, this.turnQueue.length)
      for (const t of queued) t.resolve('cancelled')

      this.emit({
        sessionUpdate: 'agent_message_chunk',
        content: { type: 'text', text: 'Cleared queued prompts.' }
      })
      this.emit({
        sessionUpdate: 'session_info_update',
        _meta: { piAcp: { queueDepth: 0, running: Boolean(this.pendingTurn) } }
      })
    }

    // Abort the currently running turn (if any). If nothing is running, this is a no-op.
    await this.proc.abort()
  }

  wasCancelRequested(): boolean {
    return this.cancelRequested
  }

  private emit(update: SessionUpdate): void {
    // Serialize update delivery.
    this.lastEmit = this.lastEmit
      .then(() =>
        this.conn.sessionUpdate({
          sessionId: this.sessionId,
          update
        })
      )
      .catch(() => {
        // Ignore notification errors (client may have gone away). We still want
        // prompt completion.
      })
  }

  private async flushEmits(): Promise<void> {
    await this.lastEmit
  }

  private startTurn(t: QueuedTurn): void {
    this.cancelRequested = false
    this.inAgentLoop = false

    this.pendingTurn = { resolve: t.resolve, reject: t.reject }

    // Publish queue depth (0 because we're starting the turn now).
    this.emit({
      sessionUpdate: 'session_info_update',
      _meta: { piAcp: { queueDepth: this.turnQueue.length, running: true } }
    })

    // Kick off pi, but completion is determined by pi events, not the RPC response.
    // Important: pi may emit multiple `turn_end` events (e.g. when the model requests tools).
    // The full prompt is finished when we see `agent_end`.
    this.proc.prompt(t.message, t.images).catch(err => {
      // If the subprocess errors before we get an `agent_end`, treat as error unless cancelled.
      // Also ensure we flush any already-enqueued updates first.
      void this.flushEmits().finally(() => {
        // If this looks like an auth/config issue, surface AUTH_REQUIRED so clients can offer terminal login.
        const authErr = maybeAuthRequiredError(err)
        if (authErr) {
          this.pendingTurn?.reject(authErr)
        } else {
          const reason: StopReason = this.cancelRequested ? 'cancelled' : 'error'
          this.pendingTurn?.resolve(reason)
        }

        this.pendingTurn = null
        this.inAgentLoop = false

        // If the prompt failed, do not automatically proceed—pi may be unhealthy.
        // But we still clear the queueDepth metadata.
        this.emit({
          sessionUpdate: 'session_info_update',
          _meta: { piAcp: { queueDepth: this.turnQueue.length, running: false } }
        })
      })
      void err
    })
  }

  private handlePiEvent(ev: PiRpcEvent) {
    const type = String((ev as any).type ?? '')

    switch (type) {
      case 'message_update': {
        const ame = (ev as any).assistantMessageEvent

        // Stream assistant text.
        if (ame?.type === 'text_delta' && typeof ame.delta === 'string') {
          this.emit({
            sessionUpdate: 'agent_message_chunk',
            content: { type: 'text', text: ame.delta } satisfies ContentBlock
          })
          break
        }

        if (ame?.type === 'thinking_delta' && typeof ame.delta === 'string') {
          this.emit({
            sessionUpdate: 'agent_thought_chunk',
            content: { type: 'text', text: ame.delta } satisfies ContentBlock
          })
          break
        }

        // Surface tool calls ASAP so clients (e.g. Zed) can show a tool-in-use/loading UI
        // while the model is still streaming tool call args.
        if (ame?.type === 'toolcall_start' || ame?.type === 'toolcall_delta' || ame?.type === 'toolcall_end') {
          const toolCall =
            // pi sometimes includes the tool call directly on the event
            (ame as any)?.toolCall ??
            // ...and always includes it in the partial assistant message at contentIndex
            (ame as any)?.partial?.content?.[(ame as any)?.contentIndex ?? 0]

          const toolCallId = String((toolCall as any)?.id ?? '')
          const toolName = String((toolCall as any)?.name ?? 'tool')

          if (toolCallId) {
            const rawInput =
              (toolCall as any)?.arguments && typeof (toolCall as any).arguments === 'object'
                ? (toolCall as any).arguments
                : (() => {
                    const s = String((toolCall as any)?.partialArgs ?? '')
                    if (!s) return undefined
                    try {
                      return JSON.parse(s)
                    } catch {
                      return { partialArgs: s }
                    }
                  })()

            const isShellTool = toolName === 'bash'
            const info = toolInfoFromPiToolCall(toolName, rawInput ?? {}, this.cwd)
            const isFileMutationTool = toolName === 'edit' || toolName === 'write'

            // Generate provisional diff content for file mutations
            const provisional = isFileMutationTool
              ? provisionalDiffContentFromFileToolArgs(toolName, rawInput ?? {})
              : []

            // Generate shell presentation if needed
            const terminalId = isShellTool ? shellTerminalId(toolCallId) : undefined
            const shellPresentation = isShellTool ? shellToolPresentation(rawInput ?? {}, terminalId) : null

            const existingStatus = this.currentToolCalls.get(toolCallId)
            const status = existingStatus ?? (isFileMutationTool || isShellTool ? 'in_progress' : 'pending')

            if (!existingStatus) {
              this.currentToolCalls.set(toolCallId, status)
              this.emit({
                sessionUpdate: 'tool_call',
                toolCallId,
                title: info.title,
                kind: info.kind,
                status,
                locations: info.locations,
                rawInput,
                content: [...info.content, ...provisional],
                _meta: {
                  ...(isShellTool && shellPresentation
                    ? {
                        terminal_info: {
                          terminal_id: terminalId!,
                          ...(shellPresentation.cwd ? { cwd: shellPresentation.cwd } : {})
                        }
                      }
                    : {})
                }
              })
            } else {
              // Best-effort: keep rawInput and title updated while args are streaming.
              this.emit({
                sessionUpdate: 'tool_call_update',
                toolCallId,
                status,
                title: info.title,
                locations: info.locations,
                rawInput,
                _meta: {
                  ...(isShellTool && terminalId
                    ? {
                        terminal_info: {
                          terminal_id: terminalId,
                          cwd: (rawInput as any)?.cd
                        }
                      }
                    : {})
                }
              })
            }
          }

          break
        }

        // Ignore other delta/event types for now.
        break
      }

      case 'tool_execution_start': {
        const toolCallId = String((ev as any).toolCallId ?? crypto.randomUUID())
        const toolName = String((ev as any).toolName ?? 'tool')
        const args = (ev as any).args as Record<string, unknown>

        let line: number | undefined

        // Capture pre-edit file contents so we can emit a structured ACP diff on completion.
        // Also compute line number for edit tools when oldText matches uniquely.
        if (toolName === 'edit' && args?.path) {
          const p = String(args.path)
          try {
            const abs = isAbsolute(p) ? p : resolvePath(this.cwd, p)
            const oldText = readFileSync(abs, 'utf8')
            this.editSnapshots.set(toolCallId, { path: p, oldText })

            const needle = typeof args.oldText === 'string' ? args.oldText : ''
            line = findUniqueLineNumber(oldText, needle)
          } catch {
            // Ignore snapshot failures; we'll fall back to plain text output.
          }
        }

        const info = toolInfoFromPiToolCall(toolName, args ?? {}, this.cwd, line)
        const isShellTool = toolName === 'bash'
        const isFileMutationTool = toolName === 'edit' || toolName === 'write'
        const terminalId = isShellTool ? shellTerminalId(toolCallId) : undefined

        // Generate provisional diff for file mutations
        const provisional = isFileMutationTool
          ? provisionalDiffContentFromFileToolArgs(toolName, args ?? {})
          : []

        const existingStatus = this.currentToolCalls.get(toolCallId)

        // If we already surfaced the tool call while the model streamed it, just transition.
        if (!existingStatus) {
          this.currentToolCalls.set(toolCallId, 'in_progress')
          this.emit({
            sessionUpdate: 'tool_call',
            toolCallId,
            title: info.title,
            kind: info.kind,
            status: 'in_progress',
            locations: info.locations,
            rawInput: args,
            content: [...info.content, ...provisional],
            _meta: {
              ...(isShellTool
                ? {
                    terminal_info: {
                      terminal_id: terminalId!,
                      cwd: (args as any)?.cd
                    }
                  }
                : {})
            }
          })
        } else {
          this.currentToolCalls.set(toolCallId, 'in_progress')
          this.emit({
            sessionUpdate: 'tool_call_update',
            toolCallId,
            status: 'in_progress',
            locations: info.locations,
            rawInput: args
          })
        }

        break
      }

      case 'tool_execution_update': {
        const toolCallId = String((ev as any).toolCallId ?? '')
        if (!toolCallId) break

        const partial = (ev as any).partialResult as Record<string, unknown> | undefined
        const toolName = String((ev as any).toolName ?? '')
        const isShellTool = toolName === 'bash'

        // For shell tools, emit terminal output updates
        if (isShellTool && partial) {
          const details = partial.details as Record<string, unknown> | undefined
          const outputText = String(details?.stdout ?? partial.stdout ?? '')
          const stderrText = String(details?.stderr ?? partial.stderr ?? '')
          const combined = [outputText, stderrText].filter(Boolean).join('\n')

          if (combined) {
            this.emit({
              sessionUpdate: 'tool_call_update',
              toolCallId,
              status: 'in_progress',
              _meta: {
                terminal_output: {
                  terminal_id: shellTerminalId(toolCallId),
                  data: combined
                }
              }
            })
          }
        } else if (partial) {
          // For non-shell tools, use the toolUpdateFromPiToolResult for structured content
          const info = toolUpdateFromPiToolResult(toolName, {}, partial)
          if (info.content) {
            this.emit({
              sessionUpdate: 'tool_call_update',
              toolCallId,
              status: 'in_progress',
              content: info.content,
              rawOutput: partial
            })
          }
        }
        break
      }

      case 'tool_execution_end': {
        const toolCallId = String((ev as any).toolCallId ?? '')
        if (!toolCallId) break

        const result = (ev as any).result as Record<string, unknown>
        const isError = Boolean((ev as any).isError)
        // toolName may not be present on tool_execution_end events, so try to infer from snapshot
        const toolName = String((ev as any).toolName ?? '')
        const args = (ev as any).args as Record<string, unknown>
        const isShellTool = toolName === 'bash'
        // Infer edit tool from snapshot presence if toolName is missing
        const isEditTool = toolName === 'edit' || (!toolName && this.editSnapshots.has(toolCallId))

        // Get structured update info
        const info = toolUpdateFromPiToolResult(toolName, args ?? {}, result)
        const isRejected = isRejectedToolResult(result)
        const status = isError || isRejected ? 'failed' : 'completed'

        // Build content array
        let content: ToolCallContent[] | undefined = info.content

        // For edit tools with snapshots, prefer the diff content
        const snapshot = this.editSnapshots.get(toolCallId)
        if (!isError && snapshot && isEditTool) {
          try {
            const abs = isAbsolute(snapshot.path) ? snapshot.path : resolvePath(this.cwd, snapshot.path)
            const newText = readFileSync(abs, 'utf8')
            if (newText !== snapshot.oldText) {
              content = [
                {
                  type: 'diff',
                  path: snapshot.path,
                  oldText: snapshot.oldText,
                  newText
                }
              ]
            }
          } catch {
            // ignore; fall back to toolUpdateFromPiToolResult content
          }
        }

        // Build _meta for shell tools
        const shellOutputText = isShellTool
          ? formatShellToolResponse(result, null)
          : null

        this.emit({
          sessionUpdate: 'tool_call_update',
          toolCallId,
          status,
          content,
          locations: info.locations,
          rawOutput: result,
          _meta: {
            ...(isShellTool
              ? {
                  terminal_exit: shellExitMeta(toolCallId, result),
                  ...(shellOutputText
                    ? {
                        toolResponse: [
                          {
                            type: 'text',
                            text: shellOutputText
                          }
                        ]
                      }
                    : {})
                }
              : {})
          }
        })

        this.currentToolCalls.delete(toolCallId)
        this.editSnapshots.delete(toolCallId)
        break
      }

      case 'auto_retry_start': {
        this.emit({
          sessionUpdate: 'agent_message_chunk',
          content: { type: 'text', text: formatAutoRetryMessage(ev) } satisfies ContentBlock
        })
        break
      }

      case 'auto_retry_end': {
        this.emit({
          sessionUpdate: 'agent_message_chunk',
          content: { type: 'text', text: 'Retry finished, resuming.' } satisfies ContentBlock
        })
        break
      }

      case 'auto_compaction_start': {
        this.emit({
          sessionUpdate: 'agent_message_chunk',
          content: { type: 'text', text: 'Context nearing limit, running automatic compaction...' } satisfies ContentBlock
        })
        break
      }

      case 'auto_compaction_end': {
        this.emit({
          sessionUpdate: 'agent_message_chunk',
          content: {
            type: 'text',
            text: 'Automatic compaction finished; context was summarized to continue the session.'
          } satisfies ContentBlock
        })
        break
      }

      case 'agent_start': {
        this.inAgentLoop = true
        break
      }

      case 'turn_end': {
        // pi uses `turn_end` for sub-steps (e.g. tool_use) and will often start another turn.
        // Do NOT resolve the ACP `session/prompt` here; wait for `agent_end`.
        break
      }

      case 'agent_end': {
        // Ensure all updates derived from pi events are delivered before we resolve
        // the ACP `session/prompt` request.
        void this.flushEmits().finally(() => {
          const reason: StopReason = this.cancelRequested ? 'cancelled' : 'end_turn'
          this.pendingTurn?.resolve(reason)
          this.pendingTurn = null
          this.inAgentLoop = false

          // Start next queued prompt, if any.
          const next = this.turnQueue.shift()
          if (next) {
            this.emit({
              sessionUpdate: 'agent_message_chunk',
              content: { type: 'text', text: `Starting queued message. (${this.turnQueue.length} remaining)` }
            })
            this.startTurn(next)
          } else {
            this.emit({
              sessionUpdate: 'session_info_update',
              _meta: { piAcp: { queueDepth: 0, running: false } }
            })
          }
        })
        break
      }

      default:
        break
    }
  }
}

function formatAutoRetryMessage(ev: PiRpcEvent): string {
  const attempt = Number((ev as any).attempt)
  const maxAttempts = Number((ev as any).maxAttempts)
  const delayMs = Number((ev as any).delayMs)

  if (!Number.isFinite(attempt) || !Number.isFinite(maxAttempts) || !Number.isFinite(delayMs)) {
    return 'Retrying...'
  }

  let delaySeconds = Math.round(delayMs / 1000)
  if (delayMs > 0 && delaySeconds === 0) delaySeconds = 1

  return `Retrying (attempt ${attempt}/${maxAttempts}, waiting ${delaySeconds}s)...`
}
