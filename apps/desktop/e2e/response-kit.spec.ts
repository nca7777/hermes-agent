/**
 * response-kit E2E — proves the desktop cards render and answer for real.
 *
 * Added 2026-09-23 with the response-kit plugin (agent half:
 * ~/.hermes/plugins/response-kit/, desktop half:
 * ~/.hermes/desktop-plugins/response-kit/plugin.js). This spec is the ONLY file
 * in this repo that belongs to that work; deleting it removes the test.
 *
 * It boots the real Electron app against the mock inference server (no real
 * provider, sandboxed HERMES_HOME) with the response-kit desktop plugin copied
 * into the sandbox home, has the fake model answer with an `::ask` card and an
 * `::detail` section, and asserts on what the renderer actually paints.
 *
 * Each test is self-contained: a failing test recycles the Playwright worker, so
 * every test sends its own prompt and waits for its own card instead of leaning on
 * the previous test's transcript.
 *
 * Run (from apps/desktop, dist/ built, hidden display so nothing pops):
 *   DISPLAY=:9 npx playwright test e2e/response-kit.spec.ts --reporter=list
 */

import * as fs from 'node:fs'
import * as path from 'node:path'

import { startMockServer } from '../../../tests-js/scripts/mock-server'

import {
  buildAppEnv,
  createSandbox,
  launchDesktop,
  type Sandbox,
  waitForAppReady,
  writeEnvFile,
  writeMockProviderConfig
} from './fixtures'
import { expect, test } from './test'

const PROOF_DIR = '/home/axwel/.hermes/response-kit/proof'
const SOURCE_PLUGIN = '/home/axwel/.hermes/desktop-plugins/response-kit'

const DETAIL = [
  '',
  'Two things are worth checking before this lands; they are folded below.',
  '',
  '::detail{title="Evidence and implementation notes" body="The flow was read end to end.\\n- messages and decisions are preserved\\n- archived conversations stay searchable"}'
].join('\n')

/** The archive card — the id below is reused on purpose by the prompt-keyed replies. */
const ARCHIVE_REPLY = [
  'Archive handling has three sensible settings. Pick one and I will wire it.',
  '',
  '::ask{id="archive-policy" q="How should completed conversations be handled?" type="single" rec="a" why="Keeps you in control until you choose Archive." opts="a=Mark ready to archive|b=Archive automatically|c=Keep active"}',
  DETAIL
].join('\n')

/** A different question with its OWN id, for the typed-letter path. */
const COMPRESS_REPLY = [
  'Compression timing is a judgement call. Pick one.',
  '',
  '::ask{id="compress-policy" q="When should long chats be compressed?" type="single" rec="b" why="Cheapest moment, no waiting mid-thought." opts="a=Compress weekly|b=Compress on demand|c=Never compress"}',
  DETAIL
].join('\n')

/**
 * A message shaped exactly like the one that leaked in a real chat: one `::detail` far
 * over the renderer's 1200-character cap (the core refuses it and paints the raw text),
 * followed by a valid ask card that must still become a card.
 */
const OVERSIZED_REPLY = [
  'The audit is in. Two of the four landed; the rest is below.',
  '',
  `::detail{title="What the checks turned up" body="${'x'.repeat(1300)}"}`,
  '',
  '::ask{id="oversized-check" q="Should the verbose line above still take a card?" type="single" rec="a" why="The card and the long line are separate paragraphs." opts="a=Card only, fold the rest|b=Prose instead|c=Split into two smaller folds"}',
].join('\n')

/**
 * The live leak case, verbatim from a real chat (2026-09-23): the model invented an
 * option marker — `a2=Archive item 4 anyway…` — and the row rendered as "C  a2=Archive
 * item 4 anyway…". The marker must be consumed so the row reads like a sentence.
 */
const MESSY_REPLY = [
  'Where things actually stand: 23 items left to walk.',
  '',
  '::ask{id="item4-email-fix" q="Item 4 — the email regression: how do you want it handled?" type="single" rec="a" why="Real defect in live code; cheapest to fix now." opts="a=Fix it now, then keep walking the list|b=Keep walking, batch all fixes at the end|a2=Archive item 4 anyway and file a card for the fix"}',
].join('\n')

/**
 * The reply whose stream the mock HOLDS two tokens in — the exact window where a
 * half-written `::ask{…` paragraph used to paint as raw markup in a normal chat
 * (2026-09-23) before snapping into a card. Token 2 carries `::ask{id="slow-flash"`
 * with the prose of token 1 already on screen, so the paragraph is incomplete and
 * visible at the same time.
 */
const SLOW_FLASH_REPLY =
  'One moment.\n\n::ask{id="slow-flash" q="Should the raw markup ever be painted?" type="single" rec="a" why="It flashed before the card resolved." opts="a=Never show raw markup|b=Show it while the card writes"}'

/** What the fake model answers with, chosen by the prompt. */
const replyFor = (prompt: string): string =>
  /messy/i.test(prompt)
    ? MESSY_REPLY
    : /oversized/i.test(prompt)
      ? OVERSIZED_REPLY
      : /slow-flash/i.test(prompt)
        ? SLOW_FLASH_REPLY
        : /compress/i.test(prompt)
          ? COMPRESS_REPLY
          : ARCHIVE_REPLY

interface Fixture {
  app: Awaited<ReturnType<typeof launchDesktop>>['app']
  page: Awaited<ReturnType<typeof launchDesktop>>['page']
  mock: Awaited<ReturnType<typeof startMockServer>>
  sandbox: Sandbox
  cleanup: () => Promise<void>
}

let fixture: Fixture | null = null

test.beforeAll(async () => {
  // The fake model returns the same reply for every prompt, so every turn shows a card.
  const mock = await startMockServer({
    holdFirstStreamForPrompt: 'slow-flash',
    holdStreamAfterWords: 2,
    replyForPrompt: prompt => replyFor(prompt)
  })

  const sandbox = createSandbox('response-kit')

  // The desktop plugin must exist in the sandbox home BEFORE the app boots.
  const target = path.join(sandbox.hermesHome, 'desktop-plugins', 'response-kit')
  fs.mkdirSync(path.dirname(target), { recursive: true })
  fs.cpSync(SOURCE_PLUGIN, target, { recursive: true })
  fs.mkdirSync(PROOF_DIR, { recursive: true })

  writeMockProviderConfig(sandbox.hermesHome, mock.url)
  writeEnvFile(sandbox.hermesHome)
  const { app, page } = await launchDesktop(buildAppEnv(sandbox))

  // The plugin's own probes (console.error → desktop.log) and any renderer error,
  // surfaced into the test output: this is how an answer that never submits is told
  // apart from a click that never happened.
  page.on('console', message => {
    const text = message.text()

    if (text.includes('response-kit') || message.type() === 'error') {
       
      console.log(`[page:${message.type()}] ${text}`)
    }
  })
  page.on('pageerror', error => {
     
    console.log(`[pageerror] ${String(error)}`)
  })

  fixture = {
    app,
    page,
    mock,
    sandbox,
    cleanup: async () => {
      await app.close().catch(() => undefined)
      await mock.close()
      sandbox.cleanup()
    }
  }
  await waitForAppReady(fixture, 120_000)
})

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null
})

/**
 * Start each test in its OWN chat. Sharing one transcript made a later test race the
 * previous turn (a prompt submitted while the last one was still running gets queued
 * and then drained, which shows up as a second copy of the same card).
 */
async function freshSession(page: Fixture['page']): Promise<void> {
  await page.getByRole('button', { name: /New session/ }).first().click()
  await page.locator('[contenteditable="true"]').first().waitFor({ state: 'visible', timeout: 60_000 })
  await page.waitForTimeout(500)
}

/** The focused chat's status chip — Working / Ready / Needs your answer / Offline. */
function statusChip(page: Fixture['page']) {
  return page.locator('button', { hasText: /^(Ready|Working|Offline|Needs your answer|\d+ answers needed)/ })
}

/** Bounded wait for the focused turn to stop running: the chip paints Working while it runs. */
async function waitIdle(page: Fixture['page']): Promise<void> {
  const chip = statusChip(page).first()

  for (let attempt = 0; attempt < 60; attempt += 1) {
    const text = (await chip.innerText().catch(() => '')) || ''

    if (!text.trim().startsWith('Working')) {return}
    await page.waitForTimeout(500)
  }
}

/** Send one message through the real composer. */
async function send(page: Fixture['page'], text: string): Promise<void> {
  const composer = page.locator('[contenteditable="true"]').first()
  await composer.waitFor({ state: 'visible', timeout: 90_000 })
  await composer.click()
  await composer.type(text, { delay: 10 })
  await page.keyboard.press('Enter')
}

/** One turn → one card, asserted present. Returns the card's option row locator helper. */
async function askCard(
  page: Fixture['page'],
  prompt: string,
  question = 'How should completed conversations be handled?'
) {
  await send(page, prompt)
  await expect(page.getByText(question).last()).toBeVisible({ timeout: 90_000 })

  return {
    option: (label: string) => page.getByRole('button', { name: new RegExp(`^[A-D] ${label}`) }).last(),
    input: () => page.getByPlaceholder('Or type your answer…').last(),
    answerButton: () => page.getByRole('button', { name: 'Answer', exact: true }).last()
  }
}

/** Locate an already-rendered card by its question (no prompt sent). Every test starts in
 *  its own chat, so the option rows in the transcript are that card's — they are the only
 *  buttons whose name begins with a letter and a space. */
function cardIn(page: Fixture['page'], question: string) {
  return {
    question: () => page.getByText(question).last(),
    options: () => page.getByRole('button', { name: /^[A-D] \S/ }),
    option: (label: string) => page.getByRole('button', { name: new RegExp(`^[A-D] ${label}`) }).last(),
    optionButton: (letter: string) =>
      page.getByRole('button', { name: new RegExp(`^${letter.toUpperCase()} `) }).last(),
    input: () => page.getByPlaceholder('Or type your answer…').last(),
    answerButton: () => page.getByRole('button', { name: 'Answer', exact: true }).last()
  }
}

test.describe('response-kit in the real desktop app', () => {
  test('renders the question card, the folded detail and the status chip', async () => {
    const page = fixture!.page
    await freshSession(page)
    const card = await askCard(page, 'how should archive handling work?')

    // The card: question, lettered options, the recommended badge and its reason.
    await expect(card.option('Mark ready to archive')).toBeVisible()
    await expect(card.option('Archive automatically')).toBeVisible()
    await expect(page.getByText('Recommended')).toBeVisible()
    await expect(page.getByText('Keeps you in control until you choose Archive.')).toBeVisible()
    await expect(card.input()).toBeVisible()

    // The folded section starts closed: title visible, body hidden.
    await expect(page.getByText('Evidence and implementation notes')).toBeVisible()
    await expect(page.getByText('archived conversations stay searchable')).toHaveCount(0)

    // A real status, not prose.
    const chip = page.locator('button', {
      hasText: /^(Ready|Working|Offline|Needs your answer|\d+ answers needed)$/
    })

    await expect(chip.first()).toBeVisible({ timeout: 30_000 })
     
    console.log(`[response-kit e2e] status chip reads: ${(await chip.first().innerText()).trim()}`)

    await page.screenshot({ path: path.join(PROOF_DIR, '01-card.png') })

    // Opening the folded section reveals its body — the proof stays one click away.
    await page.getByRole('button', { name: /Evidence and implementation notes/ }).last().click()
    await expect(page.getByText('archived conversations stay searchable')).toBeVisible({ timeout: 15_000 })
    await page.screenshot({ path: path.join(PROOF_DIR, '02-detail-open.png') })
  })

  test('clicking an option answers once, and the card locks', async () => {
    const page = fixture!.page
    await freshSession(page)
    const card = await askCard(page, 'archive handling again please')
    await card.option('Archive automatically').click()

    // The answer lands in the transcript as a normal user turn.
    await expect(page.getByText('B — Archive automatically').last()).toBeVisible({ timeout: 45_000 })

    // The owning card locks and says what was chosen — no second submit from it.
    await expect(page.getByText('Answered: B — Archive automatically').last()).toBeVisible({ timeout: 45_000 })

    const disabled = await page
      .getByRole('button', { name: /^B Archive automatically/ })
      .last()
      .evaluate(node => (node as HTMLButtonElement).disabled)

    expect(disabled).toBe(true)

    await page.screenshot({ path: path.join(PROOF_DIR, '03-answered.png') })
  })

  test('typing a letter answers the card the same way a click does', async () => {
    const page = fixture!.page
    await freshSession(page)
    const card = await askCard(page, 'tell me about compress timing', 'When should long chats be compressed?')

    await waitIdle(page)
    await card.input().click()
    await card.input().type('c', { delay: 20 })
    await expect(card.answerButton()).toBeEnabled({ timeout: 10_000 })
    await card.answerButton().click()

    await expect(page.getByText('C — Never compress').last()).toBeVisible({ timeout: 45_000 })
    await page.screenshot({ path: path.join(PROOF_DIR, '04-typed-letter.png') })
  })

  test('an oversized directive does not take the rest of the message down with it', async () => {
    const page = fixture!.page
    await freshSession(page)
    await send(page, 'show me the oversized one')
    const card = cardIn(page, 'Should the verbose line above still take a card?')

    // The card renders even though the paragraph above it was refused by the renderer.
    await expect(card.options().first()).toBeVisible({ timeout: 30_000 })
    await expect(card.options()).toHaveCount(3)

    // And the refused line is shown as raw text, not swallowed — that is the symptom the
    // recipe's length limits exist to prevent.
    await expect(page.getByText('::detail{title="What the checks turned up"', { exact: false })).toBeVisible()
    await page.screenshot({ path: `${PROOF_DIR}/06-oversized-detail.png`, fullPage: false })

    await card.optionButton('a').click()
    await expect(page.getByText('Answered: A — Card only, fold the rest')).toBeVisible({ timeout: 20_000 })
  })

  test('an invented option marker renders a clean row, not "a2=…"', async () => {
    const page = fixture!.page
    await freshSession(page)
    await send(page, 'show me the messy one')
    const card = cardIn(page, 'Item 4 — the email regression: how do you want it handled?')

    // Three real rows, lettered by position — and the invented marker is gone: a row
    // reading "C  a2=Archive item 4 anyway…" is the model's typo shown to the owner.
    await expect(card.options()).toHaveCount(3, { timeout: 30_000 })
    await expect(
      page.getByRole('button', { name: /^C Archive item 4 anyway and file a card for the fix/ })
    ).toBeVisible()
    await expect(page.getByText('a2=Archive', { exact: false })).toHaveCount(0)

    await card.optionButton('c').click()
    await expect(
      page.getByText('C — Archive item 4 anyway and file a card for the fix').last()
    ).toBeVisible({ timeout: 45_000 })
    await page.screenshot({ path: `${PROOF_DIR}/07-invented-marker.png`, fullPage: false })
  })

  test('a reused question id stays answerable; the older card steps aside', async () => {
    const page = fixture!.page
    await freshSession(page)
    // The same question (and the same id) asked twice — the case that once left the
    // second card permanently locked.
    await askCard(page, 'archive policy once more')
    await askCard(page, 'archive policy a third time')

    const options = page.getByRole('button', { name: /^A Mark ready to archive/ })
    await expect(options.last()).toBeEnabled({ timeout: 30_000 })
    await expect(options.first()).toBeDisabled()

    await options.last().click()
    await expect(page.getByText('A — Mark ready to archive').last()).toBeVisible({ timeout: 45_000 })
    await expect(page.getByText('Superseded by a newer question in this chat.')).toBeVisible({ timeout: 30_000 })
    await page.screenshot({ path: path.join(PROOF_DIR, '05-reused-id.png') })
  })

  /**
   * The live bug this spec exists to keep closed: while a directive paragraph is still
   * being written, the renderer used to paint the bare markup — the owner photographed
   * `::ask{id="app-raw-flash" …}` sitting in his transcript in a NORMAL chat. The mock
   * holds the stream two tokens in — mid-directive, with the prose before it already
   * painted — so this asserts the exact window instead of racing it.
   */
  test('a half-written directive is never painted as raw markup while the turn streams', async () => {
    const page = fixture!.page
    await freshSession(page)

    await send(page, 'slow-flash please')
    await expect(page.getByText('slow-flash please').last()).toBeVisible({ timeout: 30_000 })

    // The stream is paused mid-directive: the turn is live and the paragraph is incomplete.
    await fixture!.mock.waitForHeldStream()
    await expect(statusChip(page).first()).toContainText('Working', { timeout: 30_000 })
    await page.waitForTimeout(1500)

    // Nothing of the markup is on screen — not the raw `::ask`, not its attributes.
    await expect(page.locator('body')).not.toContainText('::ask')
    await page.screenshot({ path: path.join(PROOF_DIR, '08-while-held.png'), fullPage: false })

    // Release: the rest of the directive streams in and settles into a real card.
    fixture!.mock.releaseHeldStream()
    await expect(page.getByText('Should the raw markup ever be painted?').last()).toBeVisible({ timeout: 45_000 })
    await expect(page.getByRole('button', { name: /^A Never show raw markup/ }).last()).toBeVisible()
    await expect(page.locator('body')).not.toContainText('::ask')
    await page.screenshot({ path: path.join(PROOF_DIR, '09-resolved-card.png'), fullPage: false })
  })
})
