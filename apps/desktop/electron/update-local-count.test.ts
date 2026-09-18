'use strict'

/**
 * Tests for electron/update-local-count.ts — the local-graph fallback for a
 * compare endpoint that can't answer.
 *
 * Why this exists: the API-first check calls GitHub's compare endpoint, which
 * 404s for any local HEAD the upstream repo has never seen. Every in-app
 * update merges `origin/main` into the current branch, so a checkout that
 * carries local commits is permanently in that state — and the old fallback
 * (`behind: null, updateAvailable: true`) renders as a count-less "(update)"
 * badge that no update can ever clear. These pin the rules that make the
 * badge honest again: reachable tip = 0 behind (ahead, not behind), held tip =
 * real count, unheld tip = still unknown.
 */

import assert from 'node:assert/strict'

import { test } from 'vitest'

import { localBehind, parseLocalLog } from './update-local-count'

const SHA_A = 'a'.repeat(40)
const SHA_B = 'b'.repeat(40)
const TIP = 'c'.repeat(40)

const REC = { code: 0, stdout: '', stderr: '' }

/** Fake git: a map from the first two args to a result. */
function fakeGit(table: Record<string, { code: number; stdout?: string; stderr?: string }>) {
  const calls: string[][] = []

  const runner = async (args: string[]) => {
    calls.push(args)

    for (const [key, value] of Object.entries(table)) {
      if (key === args.slice(0, 2).join(' ')) {
        return { code: value.code, stdout: value.stdout ?? '', stderr: value.stderr ?? '' }
      }
    }

    // Anything unasked-for is a test bug, not a silent miss.
    throw new Error(`unexpected git call: ${args.join(' ')}`)
  }

  return { calls, runner }
}

test('a tip reachable from HEAD is 0 behind — the diverged-but-current case that used to stick', async () => {
  const { runner } = fakeGit({ 'cat-file -e': REC, 'merge-base --is-ancestor': REC })

  assert.deepEqual(await localBehind(TIP, runner), { behind: 0, commits: [] })
})

test('a tip we do not hold stays unknown so callers keep the honest present-but-uncountable state', async () => {
  const { runner } = fakeGit({ 'cat-file -e': { code: 1, stderr: 'not a valid object' } })

  assert.equal(await localBehind(TIP, runner), null)
})

test('a diverged tip we do hold is counted locally, with the commit list the overlay renders', async () => {
  const log = [
    [SHA_B, 'feat: newer', 'B', '1789000200'].join('\u001f'),
    [SHA_A, 'fix: older', 'A', '1789000100'].join('\u001f'),
    ''
  ].join('\n')

  const { calls, runner } = fakeGit({
    'cat-file -e': REC,
    'merge-base --is-ancestor': { code: 1 },
    'rev-list --count': { ...REC, stdout: '2\n' },
    'log --max-count=250': { ...REC, stdout: log }
  })

  const result = await localBehind(TIP, runner)

  assert.equal(result?.behind, 2)
  assert.deepEqual(
    result?.commits.map(c => [c.sha, c.summary, c.author, c.at]),
    [
      [SHA_B, 'feat: newer', 'B', 1789000200000],
      [SHA_A, 'fix: older', 'A', 1789000100000]
    ]
  )
  // Counted against the tip, never the tracked branch name (a fork's default
  // branch may not be the branch being followed).
  assert.deepEqual(calls.at(-1), [
    'log',
    '--max-count=250',
    '--format=%H%x1f%s%x1f%an%x1f%ct',
    `HEAD..${TIP}`
  ])
})

test('an uncomputable count stays unknown rather than reporting zero', async () => {
  const failedRevList = fakeGit({
    'cat-file -e': REC,
    'merge-base --is-ancestor': { code: 1 },
    'rev-list --count': { code: 128, stderr: 'fatal: bad revision' }
  })

  assert.equal(await localBehind(TIP, failedRevList.runner), null)

  const garbage = fakeGit({
    'cat-file -e': REC,
    'merge-base --is-ancestor': { code: 1 },
    'rev-list --count': { ...REC, stdout: 'not-a-number\n' }
  })

  assert.equal(await localBehind(TIP, garbage.runner), null)
})

test('a malformed tip is unknown without shelling out', async () => {
  const { calls, runner } = fakeGit({})

  for (const bad of ['', 'HEAD', 'abc123', `${'a'.repeat(39)}`, `${'a'.repeat(41)}`]) {
    assert.equal(await localBehind(bad, runner), null)
  }

  assert.equal(calls.length, 0)
})

test('a missing commit list still yields the count — the count is the load-bearing part', async () => {
  const { runner } = fakeGit({
    'cat-file -e': REC,
    'merge-base --is-ancestor': { code: 1 },
    'rev-list --count': { ...REC, stdout: '7\n' },
    'log --max-count=250': { code: 128, stderr: 'fatal: bad object' }
  })

  assert.deepEqual(await localBehind(TIP, runner), { behind: 7, commits: [] })
})

test('parseLocalLog tolerates junk lines and keeps newest-first order', () => {
  const log = [
    [SHA_A, 'feat: first line only', 'A', '1789000100'].join('\u001f'),
    'warning: something on stderr',
    [SHA_B, 'fix: second', 'B', '1789000200'].join('\u001f'),
    ''
  ].join('\n')

  assert.deepEqual(
    parseLocalLog(log).map(c => [c.sha, c.summary]),
    [
      [SHA_A, 'feat: first line only'],
      [SHA_B, 'fix: second']
    ]
  )
  assert.deepEqual(parseLocalLog(''), [])
})
