# Claude Code Project Instructions

## Critical Restrictions

**NEVER run git commit commands.** This applies to:
- `git commit`
- `git commit -m`
- `git commit --amend`
- Any variation of commit commands

If the user asks you to commit, remind them that committing is disabled for this project and they should commit manually.

**NEVER run git push commands.** This applies to:
- `git push`
- `git push origin`
- `git push --force`
- Any variation of push commands

## Allowed Git Operations

You MAY use these git commands:
- `git status`
- `git diff`
- `git log`
- `git branch`
- `git stash`
- `git add` (for staging, but not committing)
- `git checkout` (for switching branches or restoring files)

## Project Context

This is an MPI Poisson Solver project for DTU course 02616 Large-scale Modelling.

Key tools:
- `uv` for dependency management
- `mpiexec -n N uv run python` for MPI execution
- `pytest` for testing
- `sphinx` for documentation

## Available Agents

- `/course-assistant` - Search course materials and library documentation
- `/code-review` - Review code against MPI best practices and Assignment 1 feedback
- `/architect` - Project structure and organization advice
