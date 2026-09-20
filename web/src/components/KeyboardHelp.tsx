import { Modal } from "./Modal";

const SHORTCUTS: [string[], string][] = [
  [["←", "→"], "previous / next head"],
  [["↑", "↓"], "previous / next layer"],
  [["d"], "flip read direction (destination ⇄ source)"],
  [["Esc"], "clear the pinned token, then close the head view"],
  [["/"], "focus the prompt box"],
  [["?"], "open this panel"],
];

export function KeyboardHelp({ onClose }: { onClose: () => void }) {
  return (
    <Modal title="Keyboard shortcuts" onClose={onClose}>
      <table className="shortcuts">
        <tbody>
          {SHORTCUTS.map(([keys, desc]) => (
            <tr key={keys.join()}>
              <td>
                {keys.map((k) => (
                  <kbd key={k} style={{ marginRight: 4 }}>
                    {k}
                  </kbd>
                ))}
              </td>
              <td>{desc}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p style={{ marginTop: 16 }} className="muted">
        Shortcuts are ignored while you're typing in the prompt box, so <kbd>/</kbd> and <kbd>?</kbd> insert
        characters there as normal.
      </p>
    </Modal>
  );
}
