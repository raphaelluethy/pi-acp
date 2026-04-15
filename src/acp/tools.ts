import type { ToolCallContent, ToolCallLocation, ToolKind } from '@agentclientprotocol/sdk'
import { isAbsolute, resolve as resolvePath } from 'node:path'

export interface ToolInfo {
  title: string
  kind: ToolKind
  content: ToolCallContent[]
  locations?: ToolCallLocation[]
}

export interface ShellToolPresentation {
  title: string
  content: ToolCallContent[]
  cwd?: string
}

export function markdownEscape(text: string): string {
  let fence = '```'
  for (const [m] of text.matchAll(/^```+/gm)) {
    while (m.length >= fence.length) {
      fence += '`'
    }
  }
  return `${fence}\n${text}${text.endsWith('\n') ? '' : '\n'}${fence}`
}

function textContent(text: string): ToolCallContent[] {
  return [
    {
      type: 'content',
      content: {
        type: 'text',
        text: markdownEscape(text)
      }
    }
  ]
}

function isObject(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v)
}

function extractTextFromToolResult(result: Record<string, unknown> | undefined): string | null {
  if (!result) return null

  const content = result.content
  if (Array.isArray(content)) {
    const texts = content
      .map((c: any) => (c?.type === 'text' && typeof c.text === 'string' ? c.text : ''))
      .filter(Boolean)
    if (texts.length) return texts.join('')
  }

  const details = result.details

  const detailsObj = isObject(details) ? details : undefined

  const stdout =
    (typeof detailsObj?.stdout === 'string' ? detailsObj.stdout : undefined) ??
    (typeof result.stdout === 'string' ? result.stdout : undefined) ??
    (typeof detailsObj?.output === 'string' ? detailsObj.output : undefined) ??
    (typeof result.output === 'string' ? result.output : undefined)

  const stderr =
    (typeof detailsObj?.stderr === 'string' ? detailsObj.stderr : undefined) ??
    (typeof result.stderr === 'string' ? result.stderr : undefined)

  if (stdout || stderr) {
    const parts: string[] = []
    if (stdout) parts.push(stdout)
    if (stderr) parts.push(`stderr:\n${stderr}`)
    return parts.join('\n\n').trimEnd()
  }

  return null
}

export function shellToolPresentation(
  args: Record<string, unknown>,
  terminalId?: string
): ShellToolPresentation {
  // pi sends the command as 'cmd', but we also support 'command' for compatibility
  const command =
    (typeof args.cmd === 'string' ? args.cmd : undefined) ??
    (typeof args.command === 'string' ? args.command : '')
  const cd = typeof args.cd === 'string' && args.cd.length > 0 ? args.cd : undefined

  const content: ToolCallContent[] = []
  if (terminalId) {
    content.push({
      type: 'terminal',
      terminalId
    })
  }

  return {
    title: command ? `\`${command.split('`').join('\\`')}\`` : 'Shell',
    content,
    cwd: cd
  }
}

export function toolInfoFromPiToolCall(
  toolName: string,
  args: Record<string, unknown>,
  cwd: string,
  line?: number
): ToolInfo {
  switch (toolName) {
    case 'bash': {
      const shell = shellToolPresentation(args)
      return {
        kind: 'execute',
        title: shell.title,
        content: shell.content,
        ...(shell.cwd ? { locations: [{ path: shell.cwd }] } : {})
      }
    }

    case 'read': {
      const path = typeof args.path === 'string' ? args.path : ''
      const resolvedPath = path && !isAbsolute(path) ? resolvePath(cwd, path) : path
      return {
        kind: 'read',
        title: path ? `Read ${path}` : 'Read',
        content: [],
        locations: resolvedPath ? [{ path: resolvedPath }] : undefined
      }
    }

    case 'write': {
      const path = typeof args.path === 'string' ? args.path : ''
      const resolvedPath = path && !isAbsolute(path) ? resolvePath(cwd, path) : path
      return {
        kind: 'edit',
        title: path ? `Write ${path}` : 'Write',
        content: [],
        locations: resolvedPath ? [{ path: resolvedPath }] : undefined
      }
    }

    case 'edit': {
      const path = typeof args.path === 'string' ? args.path : ''
      const resolvedPath = path && !isAbsolute(path) ? resolvePath(cwd, path) : path
      const locations = resolvedPath
        ? [{ path: resolvedPath, ...(typeof line === 'number' ? { line } : {}) }]
        : undefined
      return {
        kind: 'edit',
        title: path ? `Edit ${path}` : 'Edit',
        content: [],
        locations
      }
    }

    default:
      return {
        kind: 'other',
        title: toolName || 'Tool',
        content: []
      }
  }
}

export function provisionalDiffContentFromFileToolArgs(
  toolName: string,
  args: Record<string, unknown>
): ToolCallContent[] {
  const path = typeof args.path === 'string' ? args.path : undefined
  if (!path) return []

  if (toolName === 'write') {
    const newText =
      typeof args.content === 'string'
        ? args.content
        : typeof args.contents === 'string'
          ? args.contents
          : undefined
    if (newText === undefined) return []

    const oldText =
      typeof args.previousContent === 'string'
        ? args.previousContent
        : typeof args.previousContents === 'string'
          ? args.previousContents
          : ''

    return [{ type: 'diff', path, oldText, newText }]
  }

  if (toolName === 'edit') {
    const oldText =
      typeof args.oldText === 'string'
        ? args.oldText
        : typeof args.old_string === 'string'
          ? args.old_string
          : typeof args.oldString === 'string'
            ? args.oldString
            : undefined

    const newText =
      typeof args.newText === 'string'
        ? args.newText
        : typeof args.new_string === 'string'
          ? args.new_string
          : typeof args.newString === 'string'
            ? args.newString
            : undefined

    if (oldText !== undefined && newText !== undefined) {
      return [{ type: 'diff', path, oldText, newText }]
    }
  }

  return []
}

export function diffContentFromEditResult(
  args: Record<string, unknown>,
  result: Record<string, unknown> | undefined,
  oldText: string
): ToolCallContent[] {
  const path = typeof args.path === 'string' ? args.path : undefined
  if (!path) return []

  const details = result ? (result.details as Record<string, unknown> | undefined) : undefined

  const diffString = typeof details?.diff === 'string' ? details.diff : null
  const newTextFromResult = typeof details?.newText === 'string' ? details.newText : null

  if (diffString) {
    return [
      {
        type: 'content',
        content: {
          type: 'text',
          text: markdownEscape(diffString)
        }
      }
    ]
  }

  if (newTextFromResult && oldText !== newTextFromResult) {
    return [{ type: 'diff', path, oldText, newText: newTextFromResult }]
  }

  return []
}

export function toolUpdateFromPiToolResult(
  toolName: string,
  args: Record<string, unknown>,
  result: Record<string, unknown> | undefined,
  terminalId?: string
): { content?: ToolCallContent[]; locations?: ToolCallLocation[] } {
  const isObject = (v: unknown): v is Record<string, unknown> =>
    typeof v === 'object' && v !== null && !Array.isArray(v)

  switch (toolName) {
    case 'bash': {
      if (terminalId) {
        return {
          content: [{ type: 'terminal', terminalId }]
        }
      }
      const outputText = extractTextFromToolResult(result ?? undefined)
      return {
        content: outputText ? textContent(outputText) : undefined
      }
    }

    case 'edit':
    case 'write': {
      const content: ToolCallContent[] = []

      const path = typeof args.path === 'string' ? args.path : undefined

      const details = result && isObject(result) ? (result.details as Record<string, unknown>) : undefined
      const diffString = typeof details?.diff === 'string' ? details.diff : null

      if (diffString) {
        content.push({
          type: 'content',
          content: {
            type: 'text',
            text: markdownEscape(diffString)
          }
        })
      }

      const text = extractTextFromToolResult(result ?? undefined)
      if (text && !diffString) {
        content.push({
          type: 'content',
          content: {
            type: 'text',
            text: markdownEscape(text)
          }
        })
      }

      return {
        content: content.length > 0 ? content : undefined,
        locations: path ? [{ path }] : undefined
      }
    }

    default: {
      const text = extractTextFromToolResult(result ?? undefined)
      if (text) {
        return {
          content: textContent(text)
        }
      }
      return {}
    }
  }
}

export function shellTerminalId(toolCallId: string): string {
  return `pi-shell-${toolCallId}`
}

export function shellExitMeta(
  toolCallId: string,
  result: Record<string, unknown> | undefined
): { terminal_id: string; exit_code: number; signal: string | null } {
  const terminalId = shellTerminalId(toolCallId)

  const isObject = (v: unknown): v is Record<string, unknown> =>
    typeof v === 'object' && v !== null && !Array.isArray(v)

  const details = result && isObject(result) ? (result.details as Record<string, unknown>) : undefined

  const exitCode =
    typeof details?.exitCode === 'number'
      ? details.exitCode
      : typeof details?.code === 'number'
        ? details.code
        : 0

  return {
    terminal_id: terminalId,
    exit_code: exitCode,
    signal: null
  }
}

export function formatShellToolResponse(
  result: Record<string, unknown> | undefined,
  outputText: string | null
): string {
  const text = outputText ?? 'Command completed with no output.'

  const isObject = (v: unknown): v is Record<string, unknown> =>
    typeof v === 'object' && v !== null && !Array.isArray(v)

  const details = result && isObject(result) ? (result.details as Record<string, unknown>) : undefined

  const exitCode =
    typeof details?.exitCode === 'number'
      ? details.exitCode
      : typeof details?.code === 'number'
        ? details.code
        : undefined

  let prefix = ''
  if (typeof exitCode === 'number') {
    prefix += `Exited with code ${exitCode}.`
  }

  if (prefix) {
    prefix += ' Output:\n\n'
    return `${prefix}${text}`
  }

  return text
}

export function isRejectedToolResult(result: Record<string, unknown> | undefined): boolean {
  if (!result) return false
  const isObject = (v: unknown): v is Record<string, unknown> =>
    typeof v === 'object' && v !== null && !Array.isArray(v)
  return isObject(result.error) || Boolean(result.rejected)
}
