$regPath = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run"
$python  = "C:\Users\taylo\AppData\Local\Programs\Python\Python313\python.exe"
$script  = "C:\PhoebeLocal\PhoebeBot\phoebe.py"

Set-ItemProperty -Path $regPath -Name "PhoebePC" -Value "`"$python`" `"$script`""
Write-Host "Done. PhoebePC will launch at next login." -ForegroundColor Green
