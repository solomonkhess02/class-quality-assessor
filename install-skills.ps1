# Installs a small, focused set of agent skills for Claude Code into this project (.claude/skills).
# Needs Node.js (npx). Run from the repo root:  ./install-skills.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$skills = @(
  @("obra/superpowers", "writing-plans"),                  # plan milestones before coding
  @("obra/superpowers", "test-driven-development"),        # parser / stats / scoring tests first
  @("obra/superpowers", "systematic-debugging"),           # root-cause instead of guess-and-retry
  @("obra/superpowers", "verification-before-completion"), # show evidence before claiming done
  @("mattpocock/skills", "grill-me")                       # stress-test a design decision on request
)
foreach ($s in $skills) {
  Write-Host "Installing $($s[1]) from $($s[0])"
  npx -y skills add $s[0] --skill $s[1] -a claude-code
}
npx -y skills list
