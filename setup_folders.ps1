# Create folder structure
$folders = @(
    "data/iSAID/train",
    "data/iSAID/val", 
    "data/iSAID/test",
    "scripts",
    "results",
    "figures",
    "paper",
    "models",
    "logs"
)

foreach ($folder in $folders) {
    New-Item -ItemType Directory -Force -Path "$folder" | Out-Null
    Write-Host "Created: $folder" -ForegroundColor Green
}

Write-Host "
Folder structure created successfully!" -ForegroundColor Cyan
Write-Host "Current structure:" -ForegroundColor Yellow
tree /f
