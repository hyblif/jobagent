# Codex Project Workflow

This project is developed in daily focused Codex conversations.

## Core Rhythm

- Use `PROJECT_PLAN.md` as the source of truth for the full project plan.
- Complete one clear project section per day.
- Start a new Codex conversation for each next section to keep context clean.
- At the beginning of each new conversation, ask Codex to read the current project files before making changes.
- At the end of each day, update project progress files so the next conversation can continue smoothly.

## Recommended Daily Prompt

```text
这是项目第 N 天。请先阅读 PROJECT_PLAN.md、CODEX_WORKFLOW.md、README.md 和当前代码。
昨天已完成：[简要列出完成内容]。
今天目标：[明确写出今天要完成的部分]。
请按现有代码风格实现，并在完成后运行测试。
完成后请更新进度记录，方便下一个对话继续。
```

## Progress Memory

Use one or more of these files to preserve context between conversations:

- `PROJECT_PLAN.md`: full plan and completion checklist
- `PROGRESS.md`: daily completed work and decisions
- `TODO.md`: remaining tasks, blockers, and next steps
- `README.md`: stable project setup and usage instructions

## Git Practice

- Prefer one branch per daily section or feature.
- Commit at the end of each completed section.
- Keep commits focused on the day's implementation.

Example:

```bash
git checkout -b day-03-auth
git commit -m "Implement authentication flow"
```

## Guiding Principle

Conversations push the work forward. Files preserve the memory.
