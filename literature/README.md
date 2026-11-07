# Literature & references

Sources collected from the `llm-lying-detection` Discord channel. Add new entries here.

## Papers

### Telling FIBS: Probes Effectively Detect Sabotage and Catch Unverbalized Deception
- Authors: Oskar J. Hollinsworth, Alex F. Spies, Tigist Diriba, Adam Gleave, Chris Cundy (FAR.AI)
- arXiv: https://arxiv.org/abs/2610.12445 (v4, 20 Oct 2026)
- Code: https://github.com/AlignmentResearch/fibs-probes
- File: [`papers/2610.12445_telling-fibs-probes-detect-sabotage.pdf`](papers/2610.12445_telling-fibs-probes-detect-sabotage.pdf)
- Shared by: soenkebartling
- Summary: Trains white-box probes on model activations using FIBS ("Falsehoods In Broad
  Settings"), the largest deception dataset so far. It targets *knowing falsehoods*:
  statements the model believes are false. Probes reach 98.7% AUROC on SHADE-Arena and
  98.0% on BashArena, beating action-only black-box text monitors. They also catch
  *introspective* deception, e.g. a denied hidden goal (up to 99.7% AUROC), and lies about
  politically sensitive topics in open-weight models.

## Articles / blog posts

### Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights
- Authors: Rikiya Takehi, Jayden Teoh, Alex Zhang, Omar Khattab (MIT CSAIL), 2 Nov 2026
- URL: https://rikiyat.github.io/blog/weight-space-programming/
- File: [`articles/weight-space-programming.html`](articles/weight-space-programming.html)
- Shared by: Miko
- Summary: Frontier LLMs can read raw network weights (e.g. say which game a policy
  network was trained on) and edit them directly, without gradient training. Example:
  editing 0.016% of Qwen3.5-0.8B's weights to fix reasoning loops raises MATH-500 from
  7.9% to 48.1%. Possible relevance: LLM-driven inspection or editing of
  deception-related weights or probes.

## Repositories

| Repo | Shared by | Notes |
|---|---|---|
| https://github.com/msverma101/LLM_lie_detector | Snoozesne | This repo |
| https://github.com/AlignmentResearch/fibs-probes | (from FIBS paper) | Dataset + probe code |
| https://github.com/Fredo220/verified-reasoning-monitoring | fred | Internal model signals for AI reliability; controlled experiments on false success reports |
| https://github.com/CorvinLabs/CorvinOS | Shumway82 | Self-hosted agentic OS; connects Claude Code agents to Discord/Slack/etc. |
| https://github.com/veegee82 | Shumway82 | GitHub profile (26 repos) |
