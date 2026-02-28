# PhoebeBot auto-backup — runs weekly via Task Scheduler
# Commits any changes and pushes to GitHub

$repoPath = "C:\PhoebeLocal\PhoebeBot"
$logFile  = "C:\PhoebeLocal\PhoebeBot\backup.log"

Set-Location $repoPath

$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm"
Add-Content $logFile "[$timestamp] Starting backup..."

# Stage all changes
git add .

# Only commit if there's something new
$status = git status --porcelain
if ($status) {
    git commit -m "Auto-backup $timestamp"
    git push origin master
    Add-Content $logFile "[$timestamp] Pushed changes."
} else {
    Add-Content $logFile "[$timestamp] Nothing new to commit."
}
