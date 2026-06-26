# Pre-publication checklist for github.com/Kamapond/Ueno-lab2

Work through this once before making the repository "reviewer-ready" and minting the DOI.
Commands assume a local clone and Git Bash / PowerShell on Windows.

## 0. Clone a fresh copy
```bash
git clone https://github.com/Kamapond/Ueno-lab2.git
cd Ueno-lab2
```

## 1. Add the helper files (from output/repo_assets/)
Copy these into the repo root and commit:
`README.md`, `LICENSE`, `requirements.txt`, `.gitignore`, `.env.example`.
Then fill in every `[PLACEHOLDER]` in `README.md` and confirm the copyright holder in `LICENSE`.

## 2. Scan CURRENT files for hardcoded secrets
The OpenAI calls already use `os.environ.get("OPENAI_API_KEY")` (good). Verify nothing else is inline —
focus on the Neo4j connection (most likely culprit), in `F.2 Neo4j Ingenstion`, `F.1`, `F.3`:
```bash
grep -rEni "sk-[A-Za-z0-9]{20,}|neo4j\+s://|bolt://|password\s*=\s*[\"'][^\"']|NEO4J_PASSWORD\s*=\s*[\"'][^\"']|api_key\s*=\s*[\"'][^\"']" .
```
- Expected: matches only on `os.environ.get(...)` (safe). Any literal key/URI/password is a finding.
- Refactor every finding to `os.environ.get("NAME")` + an entry in `.env.example`.

## 3. Scan the FULL git HISTORY (all 42 commits)
A secret can hide in history even if it is gone from the current tree. Use one of:
```bash
# Option A: gitleaks (recommended)
gitleaks detect --source . --redact -v

# Option B: trufflehog
trufflehog git file://. --only-verified
```

## 4. If ANY secret is or was committed
1. **Rotate the credential immediately** — generate a new OpenAI API key (revoke the old one) and reset
   the Neo4j AuraDB password. Rotation is mandatory even after cleaning history, because the old value
   may already be cached/scraped.
2. **Remove it from history.** Because this repo has 0 forks and 0 stars, the simplest guaranteed-clean
   option is to **re-publish from a fresh, secret-free snapshot**:
   ```bash
   # from a clean working tree with secrets removed and .gitignore in place
   rm -rf .git
   git init && git add . && git commit -m "Clean public release"
   git branch -M main
   git remote add origin https://github.com/Kamapond/Ueno-lab2.git
   git push -f origin main
   ```
   (Alternatively, scrub with `git filter-repo` or BFG if you must keep history.)

## 5. Confirm NO copyrighted data is tracked
```bash
git ls-files | grep -Ei "\.(json|csv|xlsx)$"
```
- Expected: **no output**. The benchmark questions and knowledge-source documents (JIS/NDIS, JSNDI texts)
  must not be in the repo. `.gitignore` now blocks them going forward.

## 6. Sanity-check the environment
```bash
python -m venv .venv && source .venv/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt
```
Pin versions for the release: `pip freeze > requirements.lock` (optional but recommended).

## 7. Mint a citable DOI (Zenodo)
1. Sign in at https://zenodo.org with your GitHub account.
2. Zenodo → **GitHub** settings → toggle **Kamapond/Ueno-lab2 = ON**.
3. On GitHub, **Releases → Draft a new release**, tag `v1.0.0`, publish. Zenodo auto-archives the
   snapshot and mints a DOI.
4. Copy the DOI into `README.md` (badge) and into the paper's Code Availability statement and reference [28].

## 8. Final polish (optional but nice)
- Fix the folder typo `F.2 Neo4j Ingenstion` → only if you choose to rename later (we are keeping names).
- Set a clear repository **Description** and **Topics** on GitHub (e.g., `graphrag`, `rag`, `knowledge-graph`,
  `item-response-theory`, `neo4j`).
- Make sure the default branch is `main` and the repo is **Public**.
