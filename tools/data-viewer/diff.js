// Word-level diff (longest common subsequence), good enough for prompts of a few hundred words.
// Returns [{type: "same" | "added" | "removed", text}] with whitespace kept attached to words.

export function wordDiff(before, after) {
  const a = tokenize(before);
  const b = tokenize(after);
  const rows = a.length + 1;
  const cols = b.length + 1;
  const lengths = new Uint16Array(rows * cols);
  for (let i = a.length - 1; i >= 0; i--) {
    for (let j = b.length - 1; j >= 0; j--) {
      lengths[i * cols + j] =
        a[i] === b[j]
          ? lengths[(i + 1) * cols + j + 1] + 1
          : Math.max(lengths[(i + 1) * cols + j], lengths[i * cols + j + 1]);
    }
  }
  const parts = [];
  const push = (type, text) => {
    const last = parts[parts.length - 1];
    if (last?.type === type) last.text += text;
    else parts.push({ type, text });
  };
  let i = 0;
  let j = 0;
  while (i < a.length && j < b.length) {
    if (a[i] === b[j]) {
      push("same", b[j]);
      i++;
      j++;
    } else if (lengths[(i + 1) * cols + j] >= lengths[i * cols + j + 1]) {
      push("removed", a[i++]);
    } else {
      push("added", b[j++]);
    }
  }
  while (i < a.length) push("removed", a[i++]);
  while (j < b.length) push("added", b[j++]);
  return parts;
}

function tokenize(text) {
  return (text ?? "").match(/\S+\s*|\s+/g) ?? [];
}
