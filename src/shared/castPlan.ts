/**
 * How a cast fits a model's reference slots, mirrored from Core so the canvas can show what a run
 * will actually send before it starts.
 *
 * The same reason `clipGrid.ts` exists: `cast_mode` defaults to auto and silently resolves to refs
 * or to a sheet, and a control that snaps without saying so reads as broken. Both sides are pinned
 * to `core/tests/fixtures/cast_plan.json`, so a change to one that the other does not follow fails
 * a test rather than quietly promising a layout the render will not honour.
 *
 * Mirrors `characters/cast.py::plan_cast`.
 */

/** More than this in one shot is refused: the gate has not shown a larger cast holds. */
export const MAX_CAST = 5

/** Below two references a character is one view of a face, which the gate has not shown holds. */
export const MIN_REFS_PER_CHARACTER = 2

export const CAST_MODES = ['auto', 'refs', 'face-each', 'sheet'] as const

export type CastMode = (typeof CAST_MODES)[number]

/** What the render gate varies on a sheet; the defaults match Core's `SheetOptions`. */
export interface SheetOptions {
  captions?: boolean
  split?: boolean
  facesOnly?: boolean
}

export interface CastPlan {
  /** The mode it resolved to - never `auto`, which is the whole point of showing it. */
  mode: Exclude<CastMode, 'auto'>
  /** References per character, in cast order. Empty for a sheet. */
  shares: number[]
  slots: number
}

/** A cast that cannot be applied as asked; the message says what to change. */
export class CastPlanError extends Error {}

/** Which cast positions each sheet holds. Mirrors `characters/sheet.py::groups`. */
export function sheetGroups(count: number, options: SheetOptions = {}): number[][] {
  if (options.split && count > 3) {
    const half = Math.ceil(count / 2)
    return [range(0, half), range(half, count)]
  }
  return [range(0, count)]
}

/** Round-robin, so the split is even and any remainder goes to the earliest characters. */
function divide(counts: readonly number[], budget: number): number[] {
  const shares = counts.map(() => 0)
  let spare = budget
  while (spare > 0) {
    let moved = false
    counts.forEach((available, index) => {
      if (spare > 0 && shares[index]! < available) {
        shares[index] += 1
        spare -= 1
        moved = true
      }
    })
    if (!moved) break
  }
  return shares
}

export function planCast(
  refCounts: readonly number[],
  budget: number,
  mode: CastMode = 'auto',
  sheet: SheetOptions = {},
): CastPlan {
  const count = refCounts.length
  if (count === 0) throw new CastPlanError('No characters are wired.')
  if (count > MAX_CAST) {
    throw new CastPlanError(
      `At most ${MAX_CAST} characters per generation, and ${count} are wired. Unwire one.`,
    )
  }
  if (!CAST_MODES.includes(mode)) {
    throw new CastPlanError(`Unknown cast mode '${mode}'; use one of ${CAST_MODES.join(', ')}.`)
  }
  if (budget <= 0) {
    throw new CastPlanError(
      "The wired images already fill this model's reference slots, so the characters have none " +
        'left. Unwire one to make room.',
    )
  }
  let resolved: CastMode = mode
  if (resolved === 'auto') {
    // One person never goes on a sheet: it would spend a slot to show what one reference shows.
    const enough = count === 1 || count * MIN_REFS_PER_CHARACTER <= budget
    resolved = enough ? 'refs' : 'sheet'
  }
  if (resolved === 'sheet') {
    const slots = sheetGroups(count, sheet).length
    if (slots > budget) {
      throw new CastPlanError(
        `The character sheets need ${slots} reference slots and ${budget} are free. ` +
          'Unwire an image, or turn off split sheets.',
      )
    }
    return { mode: 'sheet', shares: [], slots }
  }
  if (count > budget) {
    throw new CastPlanError(
      `${count} characters need at least ${count} reference slots and ${budget} are free. ` +
        'Use a character sheet, or unwire an image.',
    )
  }
  const shares =
    resolved === 'face-each' ? refCounts.map((n) => Math.min(1, n)) : divide(refCounts, budget)
  return { mode: resolved, shares, slots: shares.reduce((a, b) => a + b, 0) }
}

function range(start: number, stop: number): number[] {
  return Array.from({ length: stop - start }, (_, i) => start + i)
}
