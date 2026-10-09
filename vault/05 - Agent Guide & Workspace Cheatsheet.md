# 🤖 05 - Agent Guide & Workspace Cheatsheet

> Quick orientation and rules for AI agents operating in this workspace.

---

## 📁 Workspace Layout
* **Workspace Path:** `c:\Users\aadit\Desktop\AIAnime`
* **Python Runtime:** Python 3.13.13 (x64) with CUDA 13.4 driver / RTX 4060 Laptop (8GB).

---

## 🛑 Operational Rules & Boundaries
1. **Ask Before Installing Dependencies:** Always check before adding packages. State the package, necessity, and lightweight alternatives.
2. **Ask Before Architecture/Schema Changes:** Verify major restructuring or destructive file operations before execution.
3. **Zero Git Modifications Without Approval:** Run only read-only git commands (`status`, `diff`, `log`) unless explicitly authorized.
4. **No Automated Browser Launching:** Dev servers, test runners, and browsers must never launch automatically.

---

## ⚙️ Standard Commands Cheatsheet
* Inspect GPU Status: `nvidia-smi`
* Run Local Verification: `python inference.py --input ./val --output ./val_output`
* Test Single Clip: `python inference.py --input ./val_sample --output ./val_output`