Write-Host "Initializing Git Repository..."
echo "# papertest" >> README.md

git init
git add .
git commit -m "Initial commit for V15 and V19 Paper Trading Automation"
git branch -M main
git remote add origin https://github.com/ANAND060218/papertest.git
git push -u origin main

Write-Host "Successfully pushed to GitHub."
