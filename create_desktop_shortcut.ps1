$wsh = New-Object -ComObject WScript.Shell
$desktopPath = [Environment]::GetFolderPath("Desktop")
$shortcutPath = Join-Path $desktopPath "Ocular AI - Blink & Yawn Tracker.lnk"
$targetScript = "c:\Users\USER\Desktop\useless3.0\Launch_App.bat"
$workingDir = "c:\Users\USER\Desktop\useless3.0"

$shortcut = $wsh.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $targetScript
$shortcut.WorkingDirectory = $workingDir
$shortcut.WindowStyle = 7 # Minimized/hidden cmd window
$shortcut.Description = "Ocular AI - Eye Blink, Yawn & Ergonomics Tracker"
$shortcut.Save()

Write-Output "Successfully created Desktop shortcut at: $shortcutPath"
