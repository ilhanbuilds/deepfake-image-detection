Option Explicit
Dim files, shell, root, python, entry, quote
Set files = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")
root = files.GetParentFolderName(WScript.ScriptFullName)
python = files.BuildPath(root, ".desktop-venv\Scripts\pythonw.exe")
entry = files.BuildPath(root, "scripts\studio_app.py")
quote = Chr(34)
If Not files.FileExists(python) Then
    MsgBox "Masaustu ortami bulunamadi. README.md kurulum adimlarini uygulayin.", 16, "Deepfake Analiz"
    WScript.Quit 1
End If
shell.CurrentDirectory = root
shell.Run quote & python & quote & " -B " & quote & entry & quote, 1, False
