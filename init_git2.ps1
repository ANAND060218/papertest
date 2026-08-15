Write-Host "Resetting Git Repository..."
Remove-Item -Recurse -Force .git

git init
git add .
git commit -m "Initial commit for V15 and V19 Paper Trading Automation with Gitignore"
git branch -M main
git remote add origin https://github.com/ANAND060218/papertest.git
git push -u origin main --force

Write-Host "Successfully pushed to GitHub."
