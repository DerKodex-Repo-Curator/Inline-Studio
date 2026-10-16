import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { CastPlanError, MAX_CAST, planCast, sheetGroups, type CastMode } from './castPlan'

/** The same file `core/tests/test_cast.py` reads: if the two drift, one of them fails here. */
const FIXTURE = join(__dirname, '../../core/tests/fixtures/cast_plan.json')

interface Case {
  counts: number[]
  budget: number
  mode: CastMode
  split?: boolean
  expect?: { mode: string; shares: number[]; slots: number }
  error?: string
}

const cases: Case[] = JSON.parse(readFileSync(FIXTURE, 'utf8')).cases

describe('the cast plan Core will use', () => {
  it.each(cases)(
    '$counts into $budget slots as $mode',
    ({ counts, budget, mode, split, expect: want, error }) => {
      const sheet = split === undefined ? {} : { split }
      if (error) {
        // Case-insensitive, matching `pytest.raises(match=f"(?i){...}")` on the Python side: the
        // fixture names the reason, not the exact wording.
        expect(() => planCast(counts, budget, mode, sheet)).toThrow(CastPlanError)
        expect(() => planCast(counts, budget, mode, sheet)).toThrow(new RegExp(error, 'i'))
        return
      }
      expect(planCast(counts, budget, mode, sheet)).toEqual(want)
    },
  )

  // The reason this module exists: `auto` is never what the run uses, so the canvas must not show it.
  it('always resolves auto to the mode the render will actually use', () => {
    expect(planCast([3, 3], 4).mode).toBe('refs')
    expect(planCast([3, 3, 3], 4).mode).toBe('sheet')
  })

  it('refuses an unknown mode rather than guessing one', () => {
    expect(() => planCast([3], 4, 'wide-shot' as CastMode)).toThrow(/Unknown cast mode/)
  })

  it('refuses a cast past the ceiling, naming the ceiling', () => {
    const counts = Array.from({ length: MAX_CAST + 1 }, () => 3)
    expect(() => planCast(counts, 9)).toThrow(/At most 5 characters/)
  })
})

describe('how a sheet is split', () => {
  // Above three characters, two half-sheets in two slots, so each face keeps more pixels.
  it('splits above three characters, and only when asked', () => {
    expect(sheetGroups(4, { split: true })).toEqual([
      [0, 1],
      [2, 3],
    ])
    expect(sheetGroups(3, { split: true })).toEqual([[0, 1, 2]])
    expect(sheetGroups(4, {})).toEqual([[0, 1, 2, 3]])
  })
})
