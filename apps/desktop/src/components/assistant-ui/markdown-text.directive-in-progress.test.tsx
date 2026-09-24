// @vitest-environment jsdom
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { registry } from '@/contrib/registry'
import { TRANSCRIPT_DIRECTIVE_AREA, type TranscriptDirectiveContribution } from '@/lib/transcript-directives'

import { MarkdownTextContent } from './markdown-text'

/**
 * The raw-text flash: deltas land a few characters at a time, so a paragraph
 * carrying a directive is unparseable for the whole time its closing `}` has
 * not arrived. It used to paint as the bare markup (`::ask{id="x" q="Which…
 * `) in EVERY desktop chat and then snap into a card on settle — reported live
 * from a normal chat on 2026-09-23. The paragraph must hold its slot empty
 * while the turn streams and still be shown (as the authoring bug it is) once
 * the stream ends.
 */

/** Claim `::ask` for one test — the core only lifts a directive a plugin claims. */
function claimAsk() {
  return registry.register({
    id: 'test:ask',
    area: TRANSCRIPT_DIRECTIVE_AREA,
    source: 'plugin:test',
    data: {
      name: 'ask',
      render: ({ attrs }) => <div data-testid="ask-card">{attrs.q}</div>
    } satisfies TranscriptDirectiveContribution
  })
}

const HALF_WRITTEN = ['One moment.', '', '::ask{id="x" q="Which way?'].join('\n')
const SETTLED = 'Standing by.\n\n::ask{id="x" q="Which way?"}'

afterEach(cleanup)

describe('a directive paragraph on a streaming turn', () => {
  it('withholds the half-written markup instead of painting it', () => {
    const { container } = render(<MarkdownTextContent isRunning text={HALF_WRITTEN} />)

    expect(container.textContent).not.toContain('::')
    expect(screen.getByText('One moment.')).toBeTruthy()
  })

  it('withholds a lone leading colon — the same line one delta earlier', () => {
    const { container } = render(<MarkdownTextContent isRunning text={'One moment.\n\n:'} />)

    expect(container.textContent).not.toContain(':')
    expect(screen.getByText('One moment.')).toBeTruthy()
  })

  it('renders the card when the directive completes mid-stream', () => {
    const dispose = claimAsk()

    try {
      render(<MarkdownTextContent isRunning text={SETTLED} />)
      expect(screen.getByTestId('ask-card').textContent).toBe('Which way?')
    } finally {
      dispose()
    }
  })

  it('shows the markup once the stream ends, so an authoring bug stays visible', () => {
    const { container } = render(<MarkdownTextContent isRunning={false} text={HALF_WRITTEN} />)

    expect(container.textContent).toContain('::ask{id="x" q="Which way?')
  })

  it('leaves prose that only mentions a scope operator alone', () => {
    const { container } = render(<MarkdownTextContent isRunning text={'Use std::vector for that.'} />)

    expect(container.textContent).toContain('std::vector')
  })
})
