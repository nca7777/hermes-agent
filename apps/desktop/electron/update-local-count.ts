'use strict'

/**
 * update-local-count.ts
 *
 * Honest behind-count for a remote tip the GitHub compare endpoint refuses.
 *
 * The passive check is API-first: it asks GitHub for the branch tip, then
 * calls `/compare/<local-head>...<tip>` for the exact count. That compare
 * 404s whenever local HEAD is a commit GitHub has never seen — which is the
 * normal state of any checkout that carries local commits on top of origin
 * (every in-app update merges `origin/main` into the current branch and mints
 * a local merge commit, so HEAD stops existing upstream). A 404 there used to
 * fall back to `behind: null, updateAvailable: true`, and the statusbar
 * renders a count-less "unknown" as `(update)`. On a diverged-but-current
 * checkout nothing can ever be applied, HEAD never moves, and nothing ever
 * clears the badge: the user clicks Update forever and the app reopens saying
 * an update is available.
 *
 * The local object graph still knows the answer whenever the tip is already on
 * disk (any earlier fetch/merge leaves `origin/<branch>` in the object DB), and
 * local git also reaches the broader "commits on disk, not reachable from tip"
 * case. So: when the compare call fails, count locally before declaring an
 * uncountable update. `null` still means "genuinely unknown" — an unfetched
 * tip, a shallow clone, no git at all — and callers keep the old honest
 * present-but-uncountable state for those.
 *
 * Dependency-injected (the caller passes `git`) so the counting rules are
 * unit-testable without booting Electron or touching a repository.
 */

import type { CompareCommit } from './update-api-check'

export interface GitResult {
  code: number
  stdout: string
  stderr: string
}

export type GitRunner = (args: string[]) => Promise<GitResult>

export interface LocalBehind {
  behind: number
  commits: CompareCommit[]
}

// Unit-separated fields (0x1f) rather than a printable delimiter: commit
// summaries are free text and can contain anything short of NUL/newline.
const LOG_FORMAT = '%H%x1f%s%x1f%an%x1f%ct'
const LOG_LIMIT = 250

/**
 * Parse `git log --format=<LOG_FORMAT>` output into the overlay's shape.
 * `git log` walks newest first, which is the order the overlay renders, so no
 * reversal here (unlike parseCompare, whose payload arrives oldest first).
 */
export function parseLocalLog(stdout: string): CompareCommit[] {
  return (stdout || '')
    .split('\n')
    .map(line => line.split('\u001f'))
    .filter(fields => /^[0-9a-f]{40}$/i.test(fields[0] || ''))
    .map(fields => ({
      sha: fields[0],
      summary: fields[1] || '',
      author: fields[2] || '',
      // %ct is UNIX seconds; the API path carries milliseconds.
      at: Number.parseInt(fields[3] || '', 10) * 1000
    }))
    .map(commit => ({ ...commit, at: Number.isFinite(commit.at) ? commit.at : 0 }))
}

/**
 * How far `targetSha` is ahead of local HEAD, counted in the local graph.
 * Returns null when the answer isn't knowable locally — the tip isn't a commit
 * we have (never fetched, or a partial clone) or git refused to answer — so
 * the caller can keep the honest "update available, count unknown" state.
 */
export async function localBehind(targetSha: string, git: GitRunner): Promise<LocalBehind | null> {
  if (!/^[0-9a-f]{40}$/i.test(targetSha || '')) {
    return null
  }

  const known = await git(['cat-file', '-e', `${targetSha}^{commit}`])

  if (known.code !== 0) {
    return null
  }

  // Reachable from HEAD means local is AHEAD of the tip, not behind it —
  // exactly the case the API can't express and the one that must not nudge
  // the user into wiping their own commits.
  const ancestor = await git(['merge-base', '--is-ancestor', targetSha, 'HEAD'])

  if (ancestor.code === 0) {
    return { behind: 0, commits: [] }
  }

  const counted = await git(['rev-list', '--count', `HEAD..${targetSha}`])

  if (counted.code !== 0) {
    return null
  }

  const behind = Number.parseInt(counted.stdout.trim(), 10)

  if (!Number.isInteger(behind) || behind < 0) {
    return null
  }

  // The count is the load-bearing part; a missing list still renders "+N".
  const log = await git(['log', `--max-count=${LOG_LIMIT}`, `--format=${LOG_FORMAT}`, `HEAD..${targetSha}`])

  return { behind, commits: log.code === 0 ? parseLocalLog(log.stdout) : [] }
}
