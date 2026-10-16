import { describe, expect, it } from 'vitest'
import type { MoodboardConnector, MoodboardItem } from '../types'
import { wiredCharacterFile, wiredCharacterFiles } from './wiredCharacter'

describe('wiredCharacterFile', () => {
  const CONNECT = (from: string): MoodboardConnector[] =>
    [
      { toItemId: 'gen', fromItemId: from, data: { targetHandle: 'character' } },
    ] as unknown as MoodboardConnector[]

  function source(id: string, type: string, params: Record<string, unknown>): MoodboardItem {
    return { id, type: 'core', data: { core: { type, params } } } as unknown as MoodboardItem
  }

  it('reads Load Character and Write .char, which name the file differently', () => {
    // Mirrors Core: `character/load` keeps it under `file`, `character/write` under `filename`.
    expect(
      wiredCharacterFile('gen', CONNECT('a'), [source('a', 'character/load', { file: 'e.char' })]),
    ).toBe('e.char')
    expect(
      wiredCharacterFile('gen', CONNECT('b'), [source('b', 'character/write', { filename: 'e' })]),
    ).toBe('e.char')
  })

  it('reduces a typed path the way the library does', () => {
    expect(
      wiredCharacterFile('gen', CONNECT('c'), [
        source('c', 'character/write', { filename: '  /core/models/emmy-s500-v6' }),
      ]),
    ).toBe('emmy-s500-v6.char')
  })

  it('names nothing for a character that has never been written', () => {
    // Encode emits an identity that only exists inside a running graph - it has no file to count.
    expect(
      wiredCharacterFile('gen', CONNECT('d'), [source('d', 'character/encode', { name: 'e' })]),
    ).toBeNull()
    expect(wiredCharacterFile('gen', [], [])).toBeNull()
  })
})

describe('wiredCharacterFiles', () => {
  function source(id: string, file: string): MoodboardItem {
    return {
      id,
      type: 'core',
      data: { core: { type: 'character/load', params: { file } } },
    } as unknown as MoodboardItem
  }

  // Wire order is what the prompt numbers, so the list keeps it rather than sorting or deduping.
  it('returns every character on the port, in wire order', () => {
    const connectors = ['a', 'b', 'c'].map((from) => ({
      toItemId: 'gen',
      fromItemId: from,
      data: { targetHandle: 'character' },
    })) as unknown as MoodboardConnector[]
    const items = [source('a', 'ada.char'), source('b', 'bo.char'), source('c', 'cy.char')]
    expect(wiredCharacterFiles('gen', connectors, items)).toEqual([
      'ada.char',
      'bo.char',
      'cy.char',
    ])
    // The singular is the first of them, which is what a one-character caller still gets.
    expect(wiredCharacterFile('gen', connectors, items)).toBe('ada.char')
  })

  it('is empty when nothing is wired', () => {
    expect(wiredCharacterFiles('gen', [], [])).toEqual([])
  })
})
