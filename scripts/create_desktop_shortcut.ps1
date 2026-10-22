$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$python = Join-Path $root '.desktop-venv/Scripts/pythonw.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw 'Masaüstü Python ortamı bulunamadı.' }
$shell = New-Object -ComObject WScript.Shell
$target = Join-Path $root 'Deepfake Analiz Studio.lnk'
if (Test-Path -LiteralPath $target) { throw 'Mevcut kısayol korundu; üzerine yazılmadı.' }
$shortcut = $shell.CreateShortcut($target)
$shortcut.TargetPath = $python
$shortcut.Arguments = '-B "' + (Join-Path $root 'scripts/studio_app.py') + '"'
$shortcut.WorkingDirectory = $root
$shortcut.IconLocation = (Join-Path $root 'assets/desktop/deepfake.ico') + ',0'
$shortcut.Description = 'Deepfake Analiz — Fotoğraf ve Video'
$shortcut.WindowStyle = 1
$shortcut.Save()
Write-Output $target
